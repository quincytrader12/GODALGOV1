"""Trading journal and daily rollup.

Append-only. The journal is the durable record of what the bot actually did, and
it outlives the in-memory position tracker, which caps its history.

Append-only matters for the same reason it does in the promotion ledger: a
record that can be rewritten is not evidence. If a bad day could be edited out,
the journal would stop being useful for exactly the question it exists to answer.

Days roll over on **UTC date**, deliberately. Crypto has no session close, so any
boundary is arbitrary; an arbitrary but fixed one is what makes daily numbers
comparable. A boundary that moved with the operator's timezone would silently
change what "today" meant.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from godalgo.ui.state import Neuron

__all__ = ["DailySummary", "TradingJournal"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DailySummary:
    """One trading day, rolled up."""

    day: date
    trades: int
    wins: int
    losses: int
    gross_profit: float
    gross_loss: float
    fees: float
    net_pnl: float
    best_trade: float
    worst_trade: float
    symbols: tuple[str, ...]
    equity_start: float
    equity_end: float

    @property
    def win_rate(self) -> float:
        return self.wins / self.trades if self.trades else 0.0

    @property
    def profit_factor(self) -> float:
        return self.gross_profit / self.gross_loss if self.gross_loss > 0 else float("inf")

    @property
    def return_pct(self) -> float:
        return (self.equity_end / self.equity_start - 1.0) if self.equity_start > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        pf = self.profit_factor
        return {
            "day": self.day.isoformat(),
            "trades": self.trades,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": round(self.win_rate, 4),
            "gross_profit": round(self.gross_profit, 2),
            "gross_loss": round(self.gross_loss, 2),
            "fees": round(self.fees, 4),
            "net_pnl": round(self.net_pnl, 2),
            "profit_factor": None if pf == float("inf") else round(pf, 3),
            "best_trade": round(self.best_trade, 2),
            "worst_trade": round(self.worst_trade, 2),
            "symbols": list(self.symbols),
            "equity_start": round(self.equity_start, 2),
            "equity_end": round(self.equity_end, 2),
            "return_pct": round(self.return_pct, 6),
        }

    def as_telegram(self) -> str:
        """Format for a chat message.

        Plain text with light markup. Numbers first, prose minimal -- this is
        read on a phone, often at a glance.
        """
        sign = "+" if self.net_pnl >= 0 else ""
        mark = "\U0001f7e2" if self.net_pnl >= 0 else "\U0001f534"
        pf = self.profit_factor
        pf_text = "inf" if pf == float("inf") else f"{pf:.2f}"
        lines = [
            f"{mark} *GODALGO daily* — {self.day.isoformat()}",
            "",
            f"Net P&L      {sign}{self.net_pnl:,.2f}",
            f"Return       {sign}{self.return_pct:.2%}",
            f"Equity       {self.equity_end:,.2f}",
            "",
            f"Trades       {self.trades}  ({self.wins}W / {self.losses}L)",
            f"Win rate     {self.win_rate:.0%}",
            f"Profit factor {pf_text}",
            f"Fees         {self.fees:,.4f}",
            "",
            f"Best         {self.best_trade:+,.2f}",
            f"Worst        {self.worst_trade:+,.2f}",
        ]
        if self.symbols:
            lines.append(f"Symbols      {', '.join(self.symbols)}")
        return "\n".join(lines)


@dataclass
class TradingJournal:
    """Append-only JSONL journal of closed positions and daily summaries."""

    path: Path = field(default_factory=lambda: Path("data/journal.jsonl"))
    summary_path: Path = field(default_factory=lambda: Path("data/daily_summaries.jsonl"))
    _current_day: date | None = field(default=None, init=False)
    _day_equity_start: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.summary_path = Path(self.summary_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, position: Neuron, equity: float | None = None) -> None:
        """Append one closed position."""
        entry = position.to_dict()
        entry["recorded_at"] = datetime.now(UTC).isoformat()
        if equity is not None:
            entry["equity_after"] = round(equity, 2)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, sort_keys=True) + "\n")

    def entries(self, day: date | None = None) -> list[dict[str, Any]]:
        """Read the journal, optionally filtered to one UTC day."""
        if not self.path.exists():
            return []
        rows = []
        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    # A partial final line can survive a hard kill. Skip it
                    # rather than losing the whole journal to one bad row.
                    logger.warning("skipping malformed journal line")
                    continue
                if day is not None and not _closed_on(row, day):
                    continue
                rows.append(row)
        return rows

    def summarise(self, day: date, equity_start: float, equity_end: float) -> DailySummary:
        """Roll up one UTC day."""
        rows = self.entries(day)
        pnls = [float(r.get("net_pnl") or 0.0) for r in rows]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        return DailySummary(
            day=day,
            trades=len(rows),
            wins=len(wins),
            losses=len(losses),
            gross_profit=sum(wins),
            gross_loss=-sum(losses),
            fees=sum(float(r.get("fees") or 0.0) for r in rows),
            net_pnl=sum(pnls),
            best_trade=max(pnls) if pnls else 0.0,
            worst_trade=min(pnls) if pnls else 0.0,
            symbols=tuple(sorted({str(r.get("symbol")) for r in rows if r.get("symbol")})),
            equity_start=equity_start,
            equity_end=equity_end,
        )

    def write_summary(self, summary: DailySummary) -> None:
        with self.summary_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(summary.to_dict(), sort_keys=True) + "\n")

    def summaries(self, limit: int = 30) -> list[dict[str, Any]]:
        if not self.summary_path.exists():
            return []
        with self.summary_path.open(encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
        return rows[-limit:]

    def check_rollover(self, equity: float, now: datetime | None = None) -> DailySummary | None:
        """Detect a UTC day boundary and roll up the day that just ended.

        Returns the completed day's summary, or None if the day has not changed.
        Called on every tick; the first call only establishes the baseline, since
        rolling up a day the process did not observe would report a partial day
        as a whole one.
        """
        now = now or datetime.now(UTC)
        today = now.date()

        if self._current_day is None:
            self._current_day = today
            self._day_equity_start = equity
            return None

        if today == self._current_day:
            return None

        finished = self.summarise(self._current_day, self._day_equity_start, equity)
        self.write_summary(finished)

        self._current_day = today
        self._day_equity_start = equity
        return finished


def _closed_on(row: dict[str, Any], day: date) -> bool:
    closed = row.get("closed_at")
    if not closed:
        return False
    try:
        return datetime.fromisoformat(closed).date() == day
    except ValueError:
        return False
