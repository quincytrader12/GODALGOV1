"""ccxt.pro WebSocket driver.

Connects a live market-data stream to ``LiveEngine``. This is the component that
turns the engine from something you call into something that runs.

## Concurrency shape

Four independent tasks, deliberately not one loop:

* **trade loop** -- ``watch_trades`` -> ``engine.ingest_tick`` (cheap, sync).
  Signals the decision task when a bar completes.
* **book loop** -- ``watch_order_book`` -> ``engine.on_book`` (cheap, sync).
* **decision loop** -- awaits the bar signal, runs the expensive decision.
* **watchdog** -- timer-driven staleness check.

The split exists because the decision is slow: regime classification refits an
ADF test and a Hurst estimate, which takes far longer than the gap between
trades on an active market. Running it inline in the socket loop would stall
reads and drop ticks precisely when the market is moving fastest.

The decision signal is a **single-slot** latch, not a queue. If bars complete
faster than decisions finish, the right behaviour is to act on the newest state
once, not to work through a backlog of stale decisions -- a queued decision
computed from three bars ago would trade on information the market has already
moved past.

## Reconnection

WebSockets drop. ccxt.pro reconnects internally for some venues but surfaces
errors for others, so both loops retry with exponential backoff and jitter, and
the watchdog independently flattens if data stops for long enough. Reconnection
logic and the staleness watchdog are separate on purpose: a reconnect loop that
silently fails forever looks identical to a quiet market unless something else
is watching the clock.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import random
from dataclasses import dataclass
from datetime import UTC, datetime

import ccxt.pro as ccxtpro

from godalgo.execution.broker import BookSnapshot
from godalgo.execution.engine import LiveEngine

__all__ = ["DriverConfig", "WebSocketDriver"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DriverConfig:
    """Stream and reconnection settings."""

    exchange_id: str = "binance"
    order_book_depth: int = 5
    """Levels to request. We only use the touch, but venues often reject depth
    values outside a supported set, and a small depth reduces bandwidth."""

    watchdog_interval: float = 10.0
    reconnect_base_delay: float = 1.0
    reconnect_max_delay: float = 60.0
    max_reconnect_attempts: int = 0
    """0 means retry indefinitely. Any positive value halts the engine once
    exhausted, which is safer than a bot that reconnects forever while holding a
    position nobody is watching."""

    autopilot_poll_seconds: float = 300.0
    """Seconds between autopilot checks. The search itself runs far less often;
    this only controls how promptly an approved swap lands once the book is
    flat."""

    decision_timeout: float = 30.0
    """Seconds a single decision may take before it is abandoned.

    A decision wedged on a hanging broker call would otherwise stop every
    subsequent bar from being acted on, silently freezing the bot in whatever
    position it last held.
    """


@dataclass
class DriverStats:
    """Observability counters."""

    trades_seen: int = 0
    books_seen: int = 0
    bars_completed: int = 0
    decisions_run: int = 0
    decisions_skipped: int = 0
    reconnects: int = 0
    errors: int = 0
    started_at: datetime | None = None

    def snapshot(self) -> dict[str, object]:
        uptime = (
            (datetime.now(UTC) - self.started_at).total_seconds()
            if self.started_at else 0.0
        )
        return {
            "uptime_s": round(uptime, 1),
            "trades": self.trades_seen,
            "books": self.books_seen,
            "bars": self.bars_completed,
            "decisions": self.decisions_run,
            "skipped": self.decisions_skipped,
            "reconnects": self.reconnects,
            "errors": self.errors,
        }


class WebSocketDriver:
    """Drives a ``LiveEngine`` from ccxt.pro streams.

    Args:
        engine: The engine to feed. Its ``config.symbol`` is the market watched.
        config: Stream and reconnection settings.
        exchange: An existing ccxt.pro exchange, for tests. One is constructed
            from ``config.exchange_id`` when omitted.
    """

    def __init__(
        self,
        engine: LiveEngine,
        config: DriverConfig | None = None,
        *,
        exchange: object | None = None,
        autopilot: object | None = None,
    ) -> None:
        self.engine = engine
        self.config = config or DriverConfig()
        self.autopilot = autopilot
        self.stats = DriverStats()
        self._owns_exchange = exchange is None

        if exchange is not None:
            self._exchange = exchange
        else:
            if self.config.exchange_id not in ccxtpro.exchanges:
                raise ValueError(
                    f"{self.config.exchange_id!r} is not a ccxt.pro exchange"
                )
            klass = getattr(ccxtpro, self.config.exchange_id)
            # Public market data needs no credentials, and a data driver that
            # cannot authenticate cannot place an order by accident.
            self._exchange = klass({
                # Without this ccxt ignores the system proxy; see ui/feed.py.
                "enableRateLimit": True, "aiohttp_trust_env": True,
            })

        self._bar_ready = asyncio.Event()
        self._stop = asyncio.Event()
        self._tasks: list[asyncio.Task] = []

    # -- lifecycle --------------------------------------------------------

    async def run(self) -> None:
        """Run until stopped, the engine halts, or reconnection is exhausted.

        Always tears down cleanly: on any exit path the engine is halted (which
        cancels resting orders and flattens) before the socket is closed.
        Leaving a position open behind a dead driver is the worst available
        outcome, so it is handled in ``finally`` rather than on the happy path.
        """
        symbol = self.engine.config.symbol
        self.stats.started_at = datetime.now(UTC)
        self._verify_support()

        logger.info(
            "driver starting: %s on %s (mode=%s)",
            symbol, self.config.exchange_id, self.engine.config.mode.value,
        )

        self._tasks = [
            asyncio.create_task(self._trade_loop(), name="trades"),
            asyncio.create_task(self._book_loop(), name="book"),
            asyncio.create_task(self._decision_loop(), name="decisions"),
            asyncio.create_task(self._watchdog_loop(), name="watchdog"),
        ]
        if self.autopilot is not None:
            self._tasks.append(
                asyncio.create_task(
                    self.autopilot.run(self.config.autopilot_poll_seconds),
                    name="autopilot",
                )
            )

        try:
            # Any task returning means a terminal condition -- a halt, or
            # reconnection exhausted. None of them exit on success.
            done, _ = await asyncio.wait(
                self._tasks, return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                if (exc := task.exception()) is not None:
                    logger.error("task %s failed: %s", task.get_name(), exc)
        except asyncio.CancelledError:
            logger.info("driver cancelled")
            raise
        finally:
            await self._shutdown()

    async def stop(self) -> None:
        """Request a graceful stop."""
        self._stop.set()

    async def _shutdown(self) -> None:
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []

        if not self.engine.state.halted:
            await self.engine.halt("driver shutting down")

        if self._owns_exchange:
            with contextlib.suppress(Exception):
                await self._exchange.close()
        logger.info("driver stopped: %s", self.stats.snapshot())

    def _verify_support(self) -> None:
        """Fail fast if the venue cannot serve the streams we need.

        Checked up front rather than on first use: discovering mid-session that
        the book stream was never available means the bot has been pricing off
        nothing.
        """
        has = getattr(self._exchange, "has", {}) or {}
        missing = [
            name for name in ("watchTrades", "watchOrderBook") if not has.get(name)
        ]
        if missing:
            raise ValueError(
                f"{self.config.exchange_id} does not support {', '.join(missing)}"
            )

    # -- stream loops -----------------------------------------------------

    async def _trade_loop(self) -> None:
        """Feed trades into the aggregator; latch when a bar completes."""
        symbol = self.engine.config.symbol

        async for trades in self._resilient(
            lambda: self._exchange.watch_trades(symbol), "trades"
        ):
            completed = False
            for trade in trades:
                price = trade.get("price")
                if not price or price <= 0:
                    continue
                moment = (
                    datetime.fromtimestamp(trade["timestamp"] / 1000.0, tz=UTC)
                    if trade.get("timestamp") else None
                )
                self.stats.trades_seen += 1
                if self.engine.ingest_tick(float(price), float(trade.get("amount") or 0.0), moment):
                    completed = True

            if completed:
                self.stats.bars_completed += 1
                # Latch rather than enqueue: if the decision task is still busy,
                # the next run should use the newest state, not replay a stale bar.
                if self._bar_ready.is_set():
                    self.stats.decisions_skipped += 1
                self._bar_ready.set()

    async def _book_loop(self) -> None:
        """Feed top of book to the engine."""
        symbol = self.engine.config.symbol

        async for book in self._resilient(
            lambda: self._exchange.watch_order_book(symbol, self.config.order_book_depth),
            "book",
        ):
            bids, asks = book.get("bids") or [], book.get("asks") or []
            if not bids or not asks:
                continue

            bid, bid_size = float(bids[0][0]), float(bids[0][1])
            ask, ask_size = float(asks[0][0]), float(asks[0][1])
            if bid <= 0 or ask <= 0 or ask < bid:
                # Crossed or empty books appear transiently on some venues
                # during fast markets. Skip rather than trade off them.
                continue

            timestamp = (
                datetime.fromtimestamp(book["timestamp"] / 1000.0, tz=UTC)
                if book.get("timestamp") else datetime.now(UTC)
            )
            self.stats.books_seen += 1
            self.engine.on_book(
                BookSnapshot(
                    symbol=symbol, bid=bid, ask=ask, timestamp=timestamp,
                    bid_size=bid_size, ask_size=ask_size,
                )
            )

    async def _resilient(self, watch, label: str):
        """Wrap a ccxt.pro watch call with reconnection and backoff.

        Yields each payload. Returns -- ending the loop and therefore the run --
        when stopped, when the engine halts, or when attempts are exhausted.
        """
        attempt = 0
        while not self._stop.is_set() and not self.engine.state.halted:
            try:
                payload = await watch()
                attempt = 0          # a success resets the backoff
                yield payload
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 -- supervisor loop, see below
                # Deliberately blind. ccxt.pro surfaces ccxt.BaseError, raw
                # websocket errors, OSError, and venue-specific types that share
                # no common base. This is a reconnect supervisor: crashing it on
                # an unanticipated exception type would leave a live position
                # with nothing watching it, which is strictly worse than
                # retrying. logger.warning rather than logger.exception because
                # disconnects are expected and a traceback per drop is noise.
                attempt += 1
                self.stats.errors += 1
                self.stats.reconnects += 1

                if (
                    self.config.max_reconnect_attempts
                    and attempt >= self.config.max_reconnect_attempts
                ):
                    logger.error(
                        "%s stream: giving up after %d attempts: %s", label, attempt, exc
                    )
                    await self.engine.halt(f"{label} stream unrecoverable: {exc}")
                    return

                delay = min(
                    self.config.reconnect_base_delay * 2 ** (attempt - 1),
                    self.config.reconnect_max_delay,
                )
                # Jitter so that two streams failing together do not reconnect
                # in lockstep and hammer the venue on the same schedule.
                delay *= 0.5 + random.random()
                logger.warning(
                    "%s stream error (attempt %d): %s -- retrying in %.1fs",
                    label, attempt, exc, delay,
                )
                await asyncio.sleep(delay)

    # -- decision and watchdog -------------------------------------------

    async def _decision_loop(self) -> None:
        """Run the engine's decision whenever a bar has completed."""
        while not self._stop.is_set() and not self.engine.state.halted:
            try:
                await asyncio.wait_for(self._bar_ready.wait(), timeout=1.0)
            except TimeoutError:
                continue

            self._bar_ready.clear()
            try:
                await asyncio.wait_for(
                    self.engine.on_bar_close(), timeout=self.config.decision_timeout
                )
                self.stats.decisions_run += 1
            except TimeoutError:
                # A wedged decision must not freeze every later bar.
                self.stats.errors += 1
                logger.error(
                    "decision exceeded %.0fs; halting rather than trading on a "
                    "stalled pipeline", self.config.decision_timeout,
                )
                await self.engine.halt("decision timeout")
            except asyncio.CancelledError:
                raise
            except Exception:
                self.stats.errors += 1
                logger.exception("decision failed")
                await self.engine.halt("decision raised")

    async def _watchdog_loop(self) -> None:
        """Independent staleness check.

        Deliberately not driven by incoming data: a watchdog that only runs when
        data arrives cannot detect data not arriving, which is the only failure
        it exists to catch.
        """
        while not self._stop.is_set() and not self.engine.state.halted:
            await asyncio.sleep(self.config.watchdog_interval)
            try:
                if await self.engine.check_staleness():
                    return
            except asyncio.CancelledError:
                raise
            except Exception:
                self.stats.errors += 1
                logger.exception("watchdog check failed")
