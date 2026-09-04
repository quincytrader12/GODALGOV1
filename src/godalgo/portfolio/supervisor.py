"""Multi-symbol supervision: running one bot across a rotating universe.

A ``LiveEngine`` trades one instrument and knows nothing about any other. That
is deliberate -- it keeps the decision path simple -- but it means nothing in
the system bounds what happens when several engines run at once. Each would
size against full equity, and three engines at "20% of equity" is 60% of the
account, not 20%.

The supervisor owns everything that is only meaningful across symbols:

* **Aggregate exposure.** A per-symbol cap says nothing about the total. Gross
  exposure is bounded here, and each engine's share is allocated from it.
* **Buying power.** Divided among active engines rather than promised in full
  to each, which is how a multi-symbol bot ends up rejected for insufficient
  balance on its fourth order.
* **Universe rotation.** Re-scan periodically, add what qualifies, retire what
  no longer does.
* **Retiring safely.** A symbol dropping out of the universe must be *flattened*,
  not merely un-watched. Stopping the engine that owns a position leaves that
  position open with nothing managing its stop -- the worst outcome available,
  and the easy mistake to make here.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

from godalgo.data.scanner import MarketScanner, ScanResult
from godalgo.execution.engine import LiveEngine

__all__ = ["PortfolioSupervisor", "SupervisorConfig", "SupervisorState"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SupervisorConfig:
    """Portfolio-level limits and rotation cadence."""

    max_gross_exposure: float = 1.0
    """Ceiling on the sum of absolute weights across every symbol.

    The limit a per-symbol cap cannot express. Without it, N engines each
    within their own bound are collectively unbounded.
    """

    max_concurrent: int = 4
    """Most symbols traded at once.

    Each additional symbol divides attention, buying power and -- because
    crypto co-moves -- adds less diversification than it appears to.
    """

    rescan_hours: float = 6.0
    min_hold_hours: float = 2.0
    """Minimum time a symbol stays in the universe once admitted.

    Without it a symbol hovering at the selection threshold is admitted and
    retired repeatedly, paying entry and exit costs each cycle for nothing.
    """

    reserve_fraction: float = 0.15
    """Share of buying power held back and never allocated.

    Covers fees, funding, and the gap between sizing and fill. An account at
    100% deployed cannot act on anything, including an exit.
    """

    flatten_on_retire: bool = True
    """Flatten a position when its symbol leaves the universe.

    Off only for a supervisor running alongside manual management. Leaving it
    off means retired positions keep their stops but nothing re-evaluates them.
    """

    def __post_init__(self) -> None:
        if not 0 < self.max_gross_exposure <= 10:
            raise ValueError("max_gross_exposure must be in (0, 10]")
        if self.max_concurrent < 1:
            raise ValueError("max_concurrent must be at least 1")
        if not 0 <= self.reserve_fraction < 1:
            raise ValueError("reserve_fraction must be in [0, 1)")


@dataclass
class SupervisorState:
    """Portfolio state, for the UI and diagnosis."""

    active: dict[str, datetime] = field(default_factory=dict)
    """Symbol to admission time."""

    last_scan_at: datetime | None = None
    last_scan: ScanResult | None = None
    rotations: int = 0
    retired_open: int = 0
    """Retirements that had to flatten a live position."""

    blocked_reason: str | None = None

    def snapshot(self) -> dict[str, object]:
        return {
            "active": sorted(self.active),
            "count": len(self.active),
            "last_scan_at": self.last_scan_at.isoformat() if self.last_scan_at else None,
            "rotations": self.rotations,
            "retired_open": self.retired_open,
            "blocked_reason": self.blocked_reason,
            "selected": (
                [c.to_dict() for c in self.last_scan.selected] if self.last_scan else []
            ),
            # Rejections carried alongside selections, because "nothing was
            # selected" and "nine things were examined and each refused for a
            # named reason" look identical from a list of winners and are
            # completely different facts. The second is the scanner working.
            "rejected": (
                [c.to_dict() for c in self.last_scan.rejected] if self.last_scan else []
            ),
            "scanned": self.last_scan.scanned if self.last_scan else 0,
        }


class PortfolioSupervisor:
    """Runs a scanner-driven universe across several engines.

    Args:
        config: Portfolio limits and cadence.
        scanner: Universe ranker.
        history: Callable returning bar history per symbol.
        tickers: Callable returning ticker data per symbol, for liquidity.
        make_engine: Factory building a ``LiveEngine`` for a symbol.
        equity: Callable returning current account equity.
    """

    def __init__(
        self,
        config: SupervisorConfig,
        scanner: MarketScanner,
        history: Callable[[], dict],
        tickers: Callable[[], dict],
        make_engine: Callable[[str], LiveEngine],
        equity: Callable[[], float],
    ) -> None:
        self.config = config
        self.scanner = scanner
        self.history = history
        self.tickers = tickers
        self.make_engine = make_engine
        self.equity = equity
        self.state = SupervisorState()
        self.engines: dict[str, LiveEngine] = {}

    # -- exposure ----------------------------------------------------------

    @property
    def gross_exposure(self) -> float:
        """Sum of absolute weights across every active engine."""
        return sum(abs(e.state.current_weight) for e in self.engines.values())

    def headroom(self) -> float:
        """Exposure still available before the portfolio cap binds."""
        return max(0.0, self.config.max_gross_exposure - self.gross_exposure)

    def per_symbol_budget(self) -> float:
        """Exposure each active engine may take.

        The portfolio cap divided by the concurrency limit rather than by the
        current count. Dividing by the live count would let the first engine
        admitted claim the entire book and then force every later one to trade
        at a fraction of it -- allocation by arrival order rather than by merit.
        """
        return self.config.max_gross_exposure / self.config.max_concurrent

    def buying_power_for(self, symbol: str) -> float:
        """Buying power allocated to one symbol, after the reserve."""
        total = self.equity()
        if total <= 0:
            return 0.0
        usable = total * (1.0 - self.config.reserve_fraction)
        return usable / self.config.max_concurrent

    # -- rotation ----------------------------------------------------------

    def due_for_scan(self, now: datetime | None = None) -> bool:
        now = now or datetime.now(UTC)
        if self.state.last_scan_at is None:
            return True
        elapsed = (now - self.state.last_scan_at).total_seconds() / 3600.0
        return elapsed >= self.config.rescan_hours

    def scan(self, now: datetime | None = None) -> ScanResult:
        """Run the scanner and record the result."""
        now = now or datetime.now(UTC)
        result = self.scanner.scan(self.history(), self.tickers())
        self.state.last_scan_at = now
        self.state.last_scan = result
        logger.info("universe scan:\n%s", result.summary())
        return result

    def target_universe(self, result: ScanResult, now: datetime | None = None) -> list[str]:
        """Symbols that should be active after this scan.

        Incumbents inside their minimum hold are retained even if they no
        longer rank, so a symbol sitting on the selection threshold is not
        churned in and out at full cost each cycle.
        """
        now = now or datetime.now(UTC)
        selected = [c.symbol for c in result.selected]

        protected = [
            symbol for symbol, admitted in self.state.active.items()
            if (now - admitted).total_seconds() / 3600.0 < self.config.min_hold_hours
        ]

        universe: list[str] = []
        for symbol in protected + selected:
            if symbol not in universe:
                universe.append(symbol)
        return universe[: self.config.max_concurrent]

    async def rotate(self, now: datetime | None = None) -> tuple[list[str], list[str]]:
        """Apply a scan: admit new symbols, retire departed ones.

        Returns ``(admitted, retired)``.
        """
        now = now or datetime.now(UTC)
        result = self.scan(now)
        target = set(self.target_universe(result, now))
        current = set(self.engines)

        retired = sorted(current - target)
        admitted = sorted(target - current)

        for symbol in retired:
            await self._retire(symbol)
        for symbol in admitted:
            self._admit(symbol, now)

        if admitted or retired:
            self.state.rotations += 1
            logger.info("universe rotated: +%s -%s", admitted or "none", retired or "none")
        return admitted, retired

    def _admit(self, symbol: str, now: datetime) -> None:
        if symbol in self.engines:
            return
        self.engines[symbol] = self.make_engine(symbol)
        self.state.active[symbol] = now
        logger.info("admitted %s", symbol)

    async def _retire(self, symbol: str) -> None:
        """Remove a symbol, flattening any position it still holds.

        The flatten happens before the engine is dropped. Dropping first would
        leave an open position with nothing managing its stop, which is worse
        than never having traded it.
        """
        engine = self.engines.get(symbol)
        if engine is None:
            self.state.active.pop(symbol, None)
            return

        if not engine.is_flat():
            self.state.retired_open += 1
            if self.config.flatten_on_retire:
                logger.warning("retiring %s with an open position — flattening first", symbol)
                try:
                    await engine.flatten()
                except Exception:
                    # Never drop the engine on a failed flatten: it still owns
                    # the position and its stop.
                    logger.exception("could not flatten %s; keeping it active", symbol)
                    return
            else:
                logger.warning(
                    "retiring %s with an open position and flatten_on_retire off; "
                    "the position remains open and unmanaged", symbol,
                )

        self.engines.pop(symbol, None)
        self.state.active.pop(symbol, None)
        logger.info("retired %s", symbol)

    # -- enforcement -------------------------------------------------------

    def clamp_target(self, symbol: str, desired: float) -> float:
        """Clamp one engine's desired weight against the portfolio.

        Two ceilings, the binding one wins: the per-symbol budget, and whatever
        gross exposure remains once every *other* engine is counted. Reductions
        always pass -- an exit blocked by an exposure limit would be a limit
        preventing risk from being reduced.
        """
        budget = self.per_symbol_budget()
        clamped = max(-budget, min(budget, desired))

        others = sum(
            abs(e.state.current_weight)
            for s, e in self.engines.items() if s != symbol
        )
        remaining = max(0.0, self.config.max_gross_exposure - others)

        current = self.engines[symbol].state.current_weight if symbol in self.engines else 0.0
        if abs(clamped) > remaining and abs(clamped) > abs(current):
            clamped = max(-remaining, min(remaining, clamped))

        return clamped

    async def run(self, poll_seconds: float = 900.0) -> None:
        """Background rotation loop."""
        while True:
            try:
                if self.due_for_scan():
                    await self.rotate()
                self.state.blocked_reason = None
            except asyncio.CancelledError:
                raise
            except Exception:
                # A failed scan must not disturb engines that are already
                # trading; they keep running their current symbols.
                self.state.blocked_reason = "scan failed; see logs"
                logger.exception("universe rotation failed")
            await asyncio.sleep(poll_seconds)
