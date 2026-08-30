"""Fleet driver: running the scanner-selected universe end to end.

``WebSocketDriver`` runs one symbol. This runs the book: it asks the supervisor
what should be traded, starts a driver for each admitted symbol, stops the ones
retired, and keeps doing that as the universe rotates.

The whole design rests on one decision: **each symbol keeps its own engine and
its own driver, and they know nothing about each other.** Everything portfolio-
wide -- gross exposure, buying power, which symbols are in play -- reaches them
through two hooks the supervisor owns. A single engine rewritten to hold N
symbols would have to re-derive per-symbol state everywhere, and every existing
test of the single-symbol path would stop covering what actually runs.

The hard part is not starting drivers, it is stopping them. A rotation that
cancels a driver still holding a position leaves that position open with
nothing managing its stop -- so retirement flattens first, and a driver is only
torn down once its engine reports flat or the flatten has definitively failed.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime

import ccxt.pro as ccxtpro

from godalgo.execution.driver import DriverConfig, WebSocketDriver
from godalgo.portfolio.supervisor import PortfolioSupervisor

__all__ = ["PortfolioDriver", "PortfolioDriverConfig"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PortfolioDriverConfig:
    """Fleet-level settings."""

    exchange_id: str = "binance"
    rotation_poll_seconds: float = 900.0
    """How often to check whether a rescan is due. The scan cadence itself is
    the supervisor's; this only bounds how promptly a due scan happens."""

    seed_bars: int = 2000
    """Historical bars fetched when a symbol is admitted.

    A newly admitted engine is blind until it clears warm-up. Seeding is the
    difference between trading a symbol now and trading it in several hundred
    bars' time.
    """

    startup_grace_seconds: float = 20.0
    """Time allowed for a new driver's streams to connect before its health is
    judged. Without it every admission looks like a stale-data failure."""


@dataclass
class FleetState:
    """What the fleet is doing, for the UI."""

    started_at: datetime | None = None
    admitted_total: int = 0
    retired_total: int = 0
    failed: dict[str, str] = field(default_factory=dict)

    def snapshot(self) -> dict[str, object]:
        return {
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "admitted_total": self.admitted_total,
            "retired_total": self.retired_total,
            "failed": dict(self.failed),
        }


class PortfolioDriver:
    """Runs one ``WebSocketDriver`` per symbol under a supervisor.

    Args:
        supervisor: Owns the universe, exposure, and buying power.
        config: Fleet settings.
        driver_config: Per-symbol driver settings.
        seed_history: Callable fetching historical bars for a newly admitted
            symbol. Optional; without it a new engine waits out its warm-up.
        exchange: Shared ccxt.pro exchange. One connection is used for every
            symbol -- a socket per symbol would hit connection limits and
            multiply reconnect storms.
    """

    def __init__(
        self,
        supervisor: PortfolioSupervisor,
        config: PortfolioDriverConfig | None = None,
        *,
        driver_config: DriverConfig | None = None,
        seed_history=None,
        exchange: object | None = None,
    ) -> None:
        self.supervisor = supervisor
        self.config = config or PortfolioDriverConfig()
        self.seed_history = seed_history
        self.state = FleetState()
        self._owns_exchange = exchange is None

        if exchange is not None:
            self._exchange = exchange
        else:
            if self.config.exchange_id not in ccxtpro.exchanges:
                raise ValueError(f"{self.config.exchange_id!r} is not a ccxt.pro exchange")
            self._exchange = getattr(ccxtpro, self.config.exchange_id)(
                {"enableRateLimit": True}
            )

        # The fleet shares one exchange, so a per-driver exchange id would be
        # ignored; keeping them consistent avoids a misleading log line.
        # dataclasses.replace rather than __dict__ -- DriverConfig uses slots.
        self._driver_config = replace(
            driver_config or DriverConfig(), exchange_id=self.config.exchange_id
        )

        self.drivers: dict[str, WebSocketDriver] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._stop = asyncio.Event()

    # -- lifecycle ---------------------------------------------------------

    async def run(self) -> None:
        """Run until stopped. Always tears the fleet down cleanly."""
        self.state.started_at = datetime.now(UTC)
        logger.warning(
            "portfolio driver starting on %s (max %d concurrent, gross cap %.2f)",
            self.config.exchange_id,
            self.supervisor.config.max_concurrent,
            self.supervisor.config.max_gross_exposure,
        )
        try:
            await self._rotate()      # first scan immediately rather than after a poll
            while not self._stop.is_set():
                await asyncio.sleep(self.config.rotation_poll_seconds)
                if self.supervisor.due_for_scan():
                    await self._rotate()
                self._reap()
        finally:
            # Runs on cancellation too, so the fleet is always torn down with
            # positions flattened rather than abandoned mid-rotation.
            await self._shutdown()

    async def stop(self) -> None:
        self._stop.set()

    async def _shutdown(self) -> None:
        """Stop every symbol, flattening as we go."""
        for symbol in list(self.drivers):
            await self._stop_symbol(symbol, flatten=True)
        if self._owns_exchange:
            with contextlib.suppress(Exception):
                await self._exchange.close()
        logger.info("portfolio driver stopped: %s", self.state.snapshot())

    # -- rotation ----------------------------------------------------------

    async def _rotate(self) -> None:
        """Apply a scan, then start and stop drivers to match."""
        try:
            admitted, retired = await self.supervisor.rotate()
        except Exception:
            # A failed scan must not disturb symbols already trading.
            logger.exception("scan failed; fleet unchanged")
            return

        for symbol in retired:
            await self._stop_symbol(symbol, flatten=True)
        for symbol in admitted:
            await self._start_symbol(symbol)

    async def _start_symbol(self, symbol: str) -> None:
        engine = self.supervisor.engines.get(symbol)
        if engine is None or symbol in self.drivers:
            return

        # Wire the portfolio hooks. Without these each engine sizes against the
        # full account and the fleet is collectively unbounded.
        engine.target_clamp = self.supervisor.clamp_target
        engine.buying_power = self.supervisor.buying_power_for

        if self.seed_history is not None:
            try:
                bars = self.seed_history(symbol, self.config.seed_bars)
                if bars is not None and len(bars):
                    engine.seed_history(bars)
            except Exception:
                # A symbol that cannot be seeded still trades, just later.
                logger.exception("could not seed %s; it will wait out warm-up", symbol)

        driver = WebSocketDriver(engine, self._driver_config, exchange=self._exchange)
        self.drivers[symbol] = driver
        self._tasks[symbol] = asyncio.create_task(driver.run(), name=f"driver:{symbol}")
        self.state.admitted_total += 1
        logger.info("started %s", symbol)

    async def _stop_symbol(self, symbol: str, *, flatten: bool) -> None:
        """Stop a symbol's driver, flattening before teardown.

        Order matters and is the reason this is not just a task cancel:
        cancelling first would drop the only thing managing the position's stop.
        """
        driver = self.drivers.get(symbol)
        engine = self.supervisor.engines.get(symbol)

        if flatten and engine is not None and not engine.is_flat():
            try:
                await engine.flatten()
            except Exception:
                logger.exception(
                    "could not flatten %s; leaving its driver running so the "
                    "position keeps its stop", symbol,
                )
                self.state.failed[symbol] = "flatten failed"
                return

        if driver is not None:
            with contextlib.suppress(Exception):
                await driver.stop()
        task = self._tasks.pop(symbol, None)
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

        self.drivers.pop(symbol, None)
        self.state.retired_total += 1
        logger.info("stopped %s", symbol)

    def _reap(self) -> None:
        """Notice drivers that have exited on their own.

        A driver returns only on a terminal condition -- a halt, or reconnection
        exhausted. Left unnoticed the symbol would sit in the universe looking
        active while nothing traded it.
        """
        for symbol, task in list(self._tasks.items()):
            if not task.done():
                continue

            # Cancelled tasks must be tested before exception(), which raises
            # CancelledError rather than returning it -- and CancelledError
            # derives from BaseException, so a suppress(Exception) around it
            # does not catch it and the reaper itself dies.
            if task.cancelled():
                reason = "cancelled"
            else:
                reason = "halted"
                exc = task.exception()
                if exc is not None:
                    reason = f"failed: {exc}"
            self.state.failed[symbol] = reason
            logger.error("driver for %s exited (%s)", symbol, reason)
            self._tasks.pop(symbol, None)
            self.drivers.pop(symbol, None)

    # -- reporting ---------------------------------------------------------

    def snapshot(self) -> dict[str, object]:
        """Fleet state for the UI."""
        return {
            "fleet": self.state.snapshot(),
            "supervisor": self.supervisor.state.snapshot(),
            "gross_exposure": round(self.supervisor.gross_exposure, 4),
            "headroom": round(self.supervisor.headroom(), 4),
            "symbols": {
                symbol: {
                    **engine.state.snapshot(),
                    "stats": self.drivers[symbol].stats.snapshot()
                    if symbol in self.drivers else None,
                }
                for symbol, engine in self.supervisor.engines.items()
            },
        }
