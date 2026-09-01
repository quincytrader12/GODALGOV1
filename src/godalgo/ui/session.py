"""The trading loop, running inside the terminal process.

Until this existed the terminal was a *viewer*: it could switch modes, but the
mode switch swapped a broker nothing was attached to, so pressing LIVE changed
a label and traded nothing. The actual bot lived behind ``python -m godalgo
live``, which is not what someone running a double-clicked executable is going
to do.

So the terminal now owns an engine. The design constraints are the ones the
rest of the system already established, and none of them are relaxed here:

* **The UI still cannot place an order.** It starts and stops a session; every
  order decision belongs to ``LiveEngine`` exactly as it does headless. There
  is still one path to the exchange.
* **The engine is an autonomous loop, not a remote control.** Nothing in the
  interface can push a target weight, size, or side into it.
* **A display failure cannot break trading.** The bridge is fed through an
  observer whose exceptions are swallowed inside the engine.
* **Mode changes go through ``ModeController``**, which flattens first and
  refuses a switch it cannot flatten for.

Dry run is a real session. It runs the whole pipeline -- data, regime, signals,
sizing, risk, routing -- and the broker discards the orders at the last step.
That is the point: the operator sees exactly what the bot would have done,
against live prices, before deciding to fund anything.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from godalgo.execution.types import TradingMode

if TYPE_CHECKING:
    from godalgo.ui.server import UIBridge

__all__ = ["TradingSession"]

logger = logging.getLogger(__name__)

_SEED_BARS = 1200
"""History fetched before a symbol can trade.

Below its warm-up the engine produces no signal, so a session that starts with
nothing seeded is a session that does nothing for several hundred bars. The
operator reads that as "it isn't working", and they are not wrong.
"""


@dataclass
class TradingSession:
    """Owns the running engine and keeps the bridge fed.

    Args:
        bridge: Where fills, equity and decision state are published.
        symbol: Instrument to trade.
        bar_seconds: Bar interval driving the decision cadence.
        exchange_id: Venue for market data.
    """

    bridge: UIBridge
    symbol: str = "BTC/USDT"
    bar_seconds: int = 60
    exchange_id: str = "binance"

    _task: asyncio.Task | None = field(default=None, init=False, repr=False)
    _engine: Any = field(default=None, init=False, repr=False)
    _driver: Any = field(default=None, init=False, repr=False)

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def status(self) -> dict[str, Any]:
        state = self._engine.state.snapshot() if self._engine is not None else {}
        return {
            "running": self.running,
            "symbol": self.symbol,
            "bar_seconds": self.bar_seconds,
            "warmed_up": self.warmed_up,
            "bars": self._engine.bars.n_complete if self._engine is not None else 0,
            "engine": state,
        }

    @property
    def warmed_up(self) -> bool:
        """Whether the engine has enough history to produce a signal at all.

        Reported because "warming up" and "seeing no opportunity" look
        identical from outside and mean completely different things.
        """
        if self._engine is None:
            return False
        return self._engine.bars.n_complete > self._engine._warmup

    # -- lifecycle ---------------------------------------------------------

    async def start(self, broker: Any) -> None:
        """Start trading against ``broker``. Restarts if already running."""
        await self.stop()

        from godalgo.execution.driver import DriverConfig, WebSocketDriver
        from godalgo.execution.engine import LiveEngine, LiveEngineConfig
        from godalgo.strategies.mean_reversion import MeanReversionStrategy
        from godalgo.strategies.momentum import MomentumStrategy

        mode = _mode_of(broker)
        config = LiveEngineConfig(
            symbol=self.symbol, bar_seconds=self.bar_seconds, mode=mode,
        )
        engine = LiveEngine(
            broker, MomentumStrategy(), MeanReversionStrategy(), config
        )
        engine.on_fill = self._on_fill
        self._engine = engine

        await self._seed(engine)

        self._driver = WebSocketDriver(
            engine, DriverConfig(exchange_id=self.exchange_id)
        )
        self._task = asyncio.create_task(self._run())
        self.bridge.events.record(
            "warn" if mode is TradingMode.LIVE else "info", "engine",
            f"trading session started in {mode.value}",
            f"{self.symbol} on {self.bar_seconds}s bars",
        )

    async def stop(self) -> None:
        """Stop the session, letting the engine flatten on its way out."""
        task, self._task = self._task, None
        if task is None:
            return

        driver = self._driver
        if driver is not None:
            with contextlib.suppress(Exception):
                await driver.stop()

        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task
        self.bridge.events.info("engine", "trading session stopped")

    async def _run(self) -> None:
        try:
            await asyncio.gather(self._driver.run(), self._publish())
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # A dead engine must be visible. Silently losing the trading loop
            # while the interface keeps rendering is the worst failure this
            # program has, because everything still looks alive.
            logger.exception("trading session failed")
            self.bridge.events.error(
                "engine", "trading session stopped unexpectedly", str(exc)[:200]
            )
            self.bridge.halted = True
            self.bridge.halt_reason = str(exc)[:200]

    async def _seed(self, engine: Any) -> None:
        """Fetch history so the engine can signal now rather than in hours."""
        from godalgo.data.feed import OHLCVFeed

        def fetch():
            feed = OHLCVFeed(exchange_id=self.exchange_id)
            return feed.fetch(self.symbol, _timeframe(self.bar_seconds),
                              limit=_SEED_BARS)

        try:
            # Off the event loop: OHLCVFeed is synchronous, and blocking here
            # would stall the websocket stream and the UI with it.
            bars = await asyncio.to_thread(fetch)
        except Exception as exc:  # noqa: BLE001 - seeding is best-effort
            self.bridge.events.warn(
                "engine", "could not seed history",
                f"{type(exc).__name__}: the bot will warm up from live bars, "
                f"which takes far longer",
            )
            return

        if bars is None or bars.empty:
            return
        engine.seed_history(bars)
        ready = engine.bars.n_complete > engine._warmup
        self.bridge.events.record(
            "good" if ready else "warn", "engine",
            f"seeded {engine.bars.n_complete} bars",
            "ready to signal" if ready
            else f"still below the {engine._warmup} bar warm-up",
        )

    # -- publishing --------------------------------------------------------

    def _on_fill(self, symbol: str, signed_qty: float, price: float, fee: float) -> None:
        self.bridge.record_fill(symbol, signed_qty, price, fee)
        self.bridge.events.good(
            "engine",
            f"{'bought' if signed_qty > 0 else 'sold'} {abs(signed_qty):.6g} {symbol}",
            f"at {price:,.4f}",
        )

    async def _publish(self) -> None:
        """Copy engine state onto the bridge, once a second.

        Polled rather than pushed: the engine already owns this state, and a
        callback per field would make the display a participant in the
        decision loop rather than an observer of it.
        """
        while True:
            engine = self._engine
            if engine is not None:
                state = engine.state
                self.bridge.equity = state.equity or self.bridge.equity
                self.bridge.peak_equity = max(self.bridge.peak_equity, self.bridge.equity)
                self.bridge.target_weight = state.target_weight
                self.bridge.current_weight = state.current_weight
                self.bridge.decisions_run = state.bars_seen
                self.bridge.halted = state.halted
                self.bridge.halt_reason = state.halt_reason
                # Mark open positions to the watchlist's live prices, so
                # unrealised P&L moves between fills rather than only at
                # them.
                for symbol, price in self.bridge.prices.items():
                    self.bridge.tracker.mark(symbol, price)
            await asyncio.sleep(1.0)


def _mode_of(broker: Any) -> TradingMode:
    """Infer the mode from the broker actually in hand.

    Read from the broker rather than passed alongside it, so the label can
    never disagree with where the orders are really going.
    """
    name = type(broker).__name__
    if name == "LiveBroker":
        return TradingMode.LIVE
    if name == "PaperBroker":
        return TradingMode.PAPER
    return TradingMode.DRY_RUN


def _timeframe(bar_seconds: int) -> str:
    """Nearest ccxt timeframe at or below the bar interval.

    Seeding with a coarser timeframe than the engine trades would hand it bars
    that are not the bars it is about to see.
    """
    for seconds, name in (
        (60, "1m"), (180, "3m"), (300, "5m"), (900, "15m"), (1800, "30m"),
        (3600, "1h"), (14400, "4h"), (86400, "1d"),
    ):
        if bar_seconds <= seconds:
            return name
    return "1d"
