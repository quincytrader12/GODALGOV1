"""Position tracking and UI state.

The engine reasons in *target weights* -- a continuous number that drifts up and
down. A human reasons in *positions*: an entry, an exit, and what it made. The UI
needs the second, so this module reconstructs discrete round trips from the
stream of fills.

That reconstruction is the whole job, and it is not cosmetic. A weight that goes
0.3 -> 0.5 -> 0.2 -> 0 is one position that was scaled into and out of, not four
events. Getting this wrong produces a display that disagrees with the broker,
which is worse than no display: it invites trading decisions against numbers
that are not real.

Two rules keep it honest:

* **Realised P&L is computed on the closing portion only**, at the weighted
  average entry. Marking the whole position at the exit price would book profit
  on the part still open.
* **Unrealised P&L is always marked to the current price**, never to the last
  fill. A position last filled an hour ago is not worth what it was then.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

__all__ = [
    "Neuron", "PositionTracker", "TerminalHealth", "UISnapshot", "WatchedSymbol",
]


@dataclass
class Neuron:
    """One position, as the UI draws it.

    Named for the display metaphor -- each is a node in the cluster -- but it is
    an ordinary position record and is the single source of truth for what the
    UI shows about a trade.
    """

    id: str
    symbol: str
    side: str
    """``long`` or ``short``."""

    quantity: float
    entry_price: float
    opened_at: datetime
    strategy: str = "blend"
    """Which book drove the entry: momentum, mean_reversion, or blend."""

    regime: str = "indeterminate"
    conviction: float = 0.0
    exit_price: float | None = None
    closed_at: datetime | None = None
    realised_pnl: float = 0.0
    fees: float = 0.0
    mark_price: float | None = None

    @property
    def is_open(self) -> bool:
        return self.closed_at is None

    @property
    def unrealised_pnl(self) -> float:
        """Marked to the current price, not to the last fill."""
        if not self.is_open or self.mark_price is None:
            return 0.0
        direction = 1.0 if self.side == "long" else -1.0
        return direction * self.quantity * (self.mark_price - self.entry_price)

    @property
    def net_pnl(self) -> float:
        """Total P&L including fees. Fees are part of the result, not a footnote."""
        return self.realised_pnl + self.unrealised_pnl - self.fees

    @property
    def notional(self) -> float:
        return abs(self.quantity) * (self.mark_price or self.entry_price)

    @property
    def state(self) -> str:
        """Display state, which drives colour.

        ``open`` is its own state rather than being coloured by running P&L: an
        open position has not made or lost anything yet, and colouring it by an
        unrealised number invites treating a paper gain as a result.
        """
        if self.is_open:
            return "open"
        return "profit" if self.net_pnl >= 0 else "loss"

    @property
    def return_pct(self) -> float:
        cost = abs(self.quantity) * self.entry_price
        return self.net_pnl / cost if cost > 0 else 0.0

    @property
    def duration_seconds(self) -> float:
        end = self.closed_at or datetime.now(UTC)
        return (end - self.opened_at).total_seconds()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "mark_price": self.mark_price,
            "opened_at": self.opened_at.isoformat(),
            "closed_at": self.closed_at.isoformat() if self.closed_at else None,
            "duration_seconds": round(self.duration_seconds, 1),
            "realised_pnl": round(self.realised_pnl, 4),
            "unrealised_pnl": round(self.unrealised_pnl, 4),
            "net_pnl": round(self.net_pnl, 4),
            "fees": round(self.fees, 6),
            "return_pct": round(self.return_pct, 6),
            "notional": round(self.notional, 2),
            "state": self.state,
            "is_open": self.is_open,
            "strategy": self.strategy,
            "regime": self.regime,
            "conviction": round(self.conviction, 4),
        }


@dataclass
class PositionTracker:
    """Reconstructs discrete positions from a stream of fills.

    Maintains one open position per symbol. A fill in the same direction scales
    it and re-averages the entry; an opposing fill reduces it, realising P&L on
    the closed portion; a fill that crosses through flat closes the position and
    opens a new one in the opposite direction.
    """

    open_positions: dict[str, Neuron] = field(default_factory=dict)
    closed_positions: list[Neuron] = field(default_factory=list)
    max_closed: int = 500
    """Cap on retained history, so a long-running session does not grow without
    bound. The journal keeps the durable record."""

    def on_fill(
        self,
        symbol: str,
        signed_quantity: float,
        price: float,
        fee: float = 0.0,
        *,
        moment: datetime | None = None,
        strategy: str = "blend",
        regime: str = "indeterminate",
        conviction: float = 0.0,
    ) -> Neuron | None:
        """Apply a fill. Returns the position if this fill closed one.

        Args:
            symbol: Market traded.
            signed_quantity: Positive for a buy, negative for a sell.
            price: Fill price.
            fee: Fee paid on this fill, in quote currency.
            moment: Fill time; defaults to now.
            strategy: Which book drove it, recorded on a newly opened position.
            regime: Regime at entry.
            conviction: Signal strength at entry.
        """
        if signed_quantity == 0 or price <= 0:
            return None

        moment = moment or datetime.now(UTC)
        existing = self.open_positions.get(symbol)

        if existing is None:
            self.open_positions[symbol] = self._open(
                symbol, signed_quantity, price, fee, moment, strategy, regime, conviction
            )
            return None

        existing_signed = existing.quantity if existing.side == "long" else -existing.quantity

        # Same direction: scale in and re-average the entry.
        if (existing_signed > 0) == (signed_quantity > 0):
            total = abs(existing_signed) + abs(signed_quantity)
            existing.entry_price = (
                abs(existing_signed) * existing.entry_price
                + abs(signed_quantity) * price
            ) / total
            existing.quantity = total
            existing.fees += fee
            existing.mark_price = price
            return None

        # Opposing fill: realise P&L on the portion that closes.
        closing = min(abs(signed_quantity), abs(existing_signed))
        direction = 1.0 if existing.side == "long" else -1.0
        existing.realised_pnl += direction * closing * (price - existing.entry_price)
        existing.fees += fee
        existing.mark_price = price

        remaining = abs(existing_signed) - closing
        if remaining > 1e-12:
            existing.quantity = remaining
            return None

        # Fully closed.
        existing.quantity = closing
        existing.exit_price = price
        existing.closed_at = moment
        self._retire(existing)
        del self.open_positions[symbol]

        # A fill larger than the position flips direction: the surplus opens a
        # new position rather than being discarded.
        surplus = abs(signed_quantity) - closing
        if surplus > 1e-12:
            self.open_positions[symbol] = self._open(
                symbol,
                surplus if signed_quantity > 0 else -surplus,
                price, 0.0, moment, strategy, regime, conviction,
            )
        return existing

    def mark(self, symbol: str, price: float) -> None:
        """Update the mark price for an open position."""
        position = self.open_positions.get(symbol)
        if position is not None and price > 0:
            position.mark_price = price

    def neurons(self, limit: int | None = None) -> list[Neuron]:
        """Positions to render: every open one, then the newest closed.

        Open positions are never dropped -- those are live risk, and a display
        that silently omits one is worse than no display. The limit only trims
        closed history, and the caller is told the true total so the cap is
        visible rather than a silent truncation.
        """
        live = list(self.open_positions.values())
        history = list(reversed(self.closed_positions))
        if limit is None:
            return live + history
        return live + history[: max(0, limit - len(live))]

    @property
    def total_tracked(self) -> int:
        return len(self.open_positions) + len(self.closed_positions)

    @property
    def realised_total(self) -> float:
        return sum(p.net_pnl for p in self.closed_positions)

    @property
    def unrealised_total(self) -> float:
        return sum(p.net_pnl for p in self.open_positions.values())

    @property
    def win_rate(self) -> float:
        if not self.closed_positions:
            return 0.0
        wins = sum(1 for p in self.closed_positions if p.net_pnl > 0)
        return wins / len(self.closed_positions)

    @property
    def profit_factor(self) -> float:
        """Gross profit over gross loss.

        ``inf`` when there are no losses yet -- reported honestly rather than
        clamped, because a finite-looking value would imply a track record the
        sample does not contain.
        """
        gains = sum(p.net_pnl for p in self.closed_positions if p.net_pnl > 0)
        losses = -sum(p.net_pnl for p in self.closed_positions if p.net_pnl < 0)
        if losses <= 0:
            return float("inf") if gains > 0 else 0.0
        return gains / losses

    def _open(
        self, symbol: str, signed_quantity: float, price: float, fee: float,
        moment: datetime, strategy: str, regime: str, conviction: float,
    ) -> Neuron:
        return Neuron(
            id=uuid.uuid4().hex[:12],
            symbol=symbol,
            side="long" if signed_quantity > 0 else "short",
            quantity=abs(signed_quantity),
            entry_price=price,
            opened_at=moment,
            strategy=strategy,
            regime=regime,
            conviction=conviction,
            fees=fee,
            mark_price=price,
        )

    def _retire(self, position: Neuron) -> None:
        self.closed_positions.append(position)
        if len(self.closed_positions) > self.max_closed:
            del self.closed_positions[: len(self.closed_positions) - self.max_closed]


@dataclass(frozen=True, slots=True)
class TerminalHealth:
    """What the health beam displays.

    A single 0-1 score, plus the components, so a degraded beam can be
    explained rather than merely observed. Each component is a real operational
    signal, not decoration.
    """

    connected: bool
    data_age_seconds: float
    halted: bool
    halt_reason: str | None
    reconnects: int
    errors: int
    decisions_run: int
    drawdown: float

    @property
    def score(self) -> float:
        """Overall health in [0, 1]. Multiplicative, so any one failure shows."""
        if self.halted:
            return 0.0
        score = 1.0 if self.connected else 0.2

        # Stale data is the failure that looks most like normality, so it is
        # weighted heavily.
        if self.data_age_seconds > 300:
            score *= 0.2
        elif self.data_age_seconds > 60:
            score *= 0.6

        if self.errors:
            score *= max(0.3, 1.0 - 0.05 * self.errors)
        if self.reconnects:
            score *= max(0.5, 1.0 - 0.02 * self.reconnects)
        score *= max(0.2, 1.0 - self.drawdown * 2.0)
        return round(max(0.0, min(1.0, score)), 3)

    @property
    def label(self) -> str:
        if self.halted:
            return "HALTED"
        score = self.score
        if score >= 0.85:
            return "NOMINAL"
        if score >= 0.6:
            return "DEGRADED"
        if score >= 0.3:
            return "IMPAIRED"
        return "CRITICAL"

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "label": self.label,
            "connected": self.connected,
            "data_age_seconds": round(self.data_age_seconds, 1),
            "halted": self.halted,
            "halt_reason": self.halt_reason,
            "reconnects": self.reconnects,
            "errors": self.errors,
            "decisions_run": self.decisions_run,
            "drawdown": round(self.drawdown, 4),
        }


@dataclass(frozen=True, slots=True)
class WatchedSymbol:
    """One row of the watchlist: an instrument the bot is tracking.

    Carries only what the panel renders. Deliberately not the full ticker --
    a ccxt ticker is ~30 fields including a nested `info` blob of raw venue
    JSON, and shipping that for every symbol once a second is a websocket
    frame two orders of magnitude larger than the panel needs.
    """

    symbol: str
    price: float = 0.0
    change_pct: float = 0.0
    """24h change, as a fraction. The venue reports this; it is not derived
    from our own history, which would be wrong for the first hour of a run."""

    quote_volume: float = 0.0
    spread_bps: float = 0.0
    held: bool = False
    """Whether a position is currently open in it."""

    active: bool = False
    """Whether it is the symbol the engine is currently deciding on."""

    stale: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "price": round(self.price, 8),
            "change_pct": round(self.change_pct, 5),
            "quote_volume": round(self.quote_volume, 2),
            "spread_bps": round(self.spread_bps, 2),
            "held": self.held,
            "active": self.active,
            "stale": self.stale,
        }


@dataclass
class UISnapshot:
    """Everything the front end renders in one frame."""

    timestamp: datetime
    neurons: list[Neuron]
    health: TerminalHealth
    equity: float
    starting_equity: float
    realised_pnl: float
    unrealised_pnl: float
    win_rate: float
    profit_factor: float
    open_count: int
    closed_count: int
    mode: str
    symbol: str
    rendered_count: int = 0
    total_count: int = 0
    regime: str = "indeterminate"
    conviction: float = 0.0
    target_weight: float = 0.0
    current_weight: float = 0.0
    last_price: float = 0.0
    watchlist: list[WatchedSymbol] = field(default_factory=list)
    """Instruments the bot is tracking, best-ranked first."""

    build: str = ""
    """Which build is serving. Visible so "the fix is not working" and "an
    older copy is still running" can be told apart."""

    has_keys: bool = False
    """Whether any credential is stored. Distinguishes "no key" from "key not
    tested yet", which the lamps must not conflate."""

    venue: dict[str, Any] = field(default_factory=dict)
    """Per-check connection state, rendered as the status lamps."""

    events: list[dict[str, Any]] = field(default_factory=list)
    """Most recent activity. Carried in the snapshot rather than polled
    separately so the log cannot drift out of step with what it describes."""

    book: dict[str, Any] | None = None
    """The allocation, reduced to its headline numbers.

    Effective breadth travels beside the position count deliberately: the
    position count is the number that looks like diversification and this is
    the one that is.
    """

    def to_dict(self) -> dict[str, Any]:
        pf = self.profit_factor
        return {
            "timestamp": self.timestamp.isoformat(),
            "neurons": [n.to_dict() for n in self.neurons],
            "health": self.health.to_dict(),
            "pnl": {
                "equity": round(self.equity, 2),
                "starting_equity": round(self.starting_equity, 2),
                "total_pnl": round(self.equity - self.starting_equity, 2),
                "total_return": round(
                    (self.equity / self.starting_equity - 1.0)
                    if self.starting_equity > 0 else 0.0, 6
                ),
                "realised": round(self.realised_pnl, 2),
                "unrealised": round(self.unrealised_pnl, 2),
                "win_rate": round(self.win_rate, 4),
                # inf is not valid JSON; the front end renders null as a dash.
                "profit_factor": None if pf == float("inf") else round(pf, 3),
                "open_count": self.open_count,
                "closed_count": self.closed_count,
                "rendered_count": self.rendered_count,
                "total_count": self.total_count,
            },
            "book": _book_headline(self.book),
            "brain": {
                "mode": self.mode,
                "symbol": self.symbol,
                "regime": self.regime,
                "conviction": round(self.conviction, 4),
                "target_weight": round(self.target_weight, 5),
                "current_weight": round(self.current_weight, 5),
                "last_price": round(self.last_price, 8),
            },
            "watchlist": [w.to_dict() for w in self.watchlist],
            "build": self.build,
            "has_keys": self.has_keys,
            "venue": self.venue,
            "events": self.events,
        }


def _book_headline(book: dict[str, Any] | None) -> dict[str, Any] | None:
    """The allocation reduced to what the header renders.

    The full book carries a line and an explanation per position; sending all
    of it once a second would be a frame far larger than the panel needs. The
    detail is one request away at /api/book.
    """
    if not book:
        return None
    if book.get("state") != "ready":
        return {"state": book.get("state"), "detail": book.get("detail", "")}

    forward = book.get("forward") or {}
    return {
        "state": "ready",
        "positions": book.get("positions", 0),
        "effective_breadth": round(float(book.get("effective_breadth", 0.0)), 2),
        "gross": round(float(book.get("gross", 0.0)), 4),
        "cash_weight": round(float(book.get("cash_weight", 0.0)), 4),
        "drawdown_scalar": round(float(book.get("drawdown_scalar", 1.0)), 3),
        "stressed_loss": round(float(book.get("stressed_loss", 0.0)), 4),
        "forward_days": forward.get("days", 0),
        "forward_summary": (forward.get("sharpe") or {}).get("summary", ""),
    }
