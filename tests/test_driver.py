"""WebSocket driver: concurrency, reconnection, and failure handling.

Driven by a fake ccxt.pro exchange. Real venues are unreachable from this
sandbox, but the logic worth testing here is not the venue's -- it is ours: does
a slow decision stall ingestion, does a dropped socket reconnect, does an
unrecoverable stream halt the engine rather than spin forever.
"""

import asyncio
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from godalgo.execution.broker import PaperBroker
from godalgo.execution.driver import DriverConfig, WebSocketDriver
from godalgo.execution.engine import LiveEngine, LiveEngineConfig
from godalgo.strategies.mean_reversion import MeanReversionStrategy
from godalgo.strategies.momentum import MomentumStrategy


class FakeExchange:
    """Minimal ccxt.pro stand-in.

    ``trade_script`` and ``book_script`` are lists of payloads or exceptions;
    an exception is raised instead of yielded, which is how stream failures and
    reconnection are exercised.
    """

    def __init__(self, trade_script=None, book_script=None, *, supports=True):
        self.has = {"watchTrades": supports, "watchOrderBook": supports}
        self._trades = list(trade_script or [])
        self._books = list(book_script or [])
        self.closed = False
        self.trade_calls = 0
        self.book_calls = 0

    async def watch_trades(self, symbol):
        self.trade_calls += 1
        if not self._trades:
            await asyncio.sleep(3600)          # idle rather than end the stream
        item = self._trades.pop(0)
        if isinstance(item, Exception):
            raise item
        await asyncio.sleep(0)
        return item

    async def watch_order_book(self, symbol, depth=5):
        self.book_calls += 1
        if not self._books:
            await asyncio.sleep(3600)
        item = self._books.pop(0)
        if isinstance(item, Exception):
            raise item
        await asyncio.sleep(0)
        return item

    async def close(self):
        self.closed = True


def make_engine(bar_seconds=60, seed_bars=1200):
    broker = PaperBroker(starting_equity=100_000.0)
    config = LiveEngineConfig(
        symbol="BTC/USDT", bar_seconds=bar_seconds, regime_window=250,
        reconcile_every=1e9,
    )
    engine = LiveEngine(broker, MomentumStrategy(), MeanReversionStrategy(), config)

    idx = pd.date_range("2024-01-01", periods=seed_bars, freq="1min", tz="UTC")
    price = pd.Series(range(seed_bars), index=idx).astype(float) * 0.01 + 100.0
    engine.seed_history(
        pd.DataFrame(
            {"open": price, "high": price * 1.001, "low": price * 0.999,
             "close": price, "volume": 1.0},
            index=idx,
        )
    )
    return engine, broker


def trade(price, ts):
    return [{"price": price, "amount": 0.1, "timestamp": int(ts.timestamp() * 1000)}]


def book(bid=100.0, ask=100.01, ts=None):
    return {
        "bids": [[bid, 5.0]], "asks": [[ask, 5.0]],
        "timestamp": int((ts or datetime.now(UTC)).timestamp() * 1000),
    }


async def run_briefly(driver, seconds=0.4):
    task = asyncio.create_task(driver.run())
    await asyncio.sleep(seconds)
    await driver.stop()
    with pytest.raises((asyncio.CancelledError, Exception)):
        task.cancel()
        await task
    return task


# --- startup ---------------------------------------------------------------

def test_unsupported_exchange_fails_fast():
    """Discovering mid-session that the book stream never existed is too late."""
    engine, _ = make_engine()
    driver = WebSocketDriver(engine, DriverConfig(), exchange=FakeExchange(supports=False))
    with pytest.raises(ValueError, match="does not support"):
        asyncio.run(driver.run())


def test_unknown_ccxtpro_exchange_rejected():
    engine, _ = make_engine()
    with pytest.raises(ValueError, match="not a ccxt.pro exchange"):
        WebSocketDriver(engine, DriverConfig(exchange_id="definitely_not_real"))


# --- ingestion -------------------------------------------------------------

def test_trades_and_books_are_ingested():
    async def run():
        engine, _ = make_engine()
        base = datetime(2024, 1, 2, tzinfo=UTC)
        trades = [trade(100.0 + i, base + timedelta(seconds=i * 30)) for i in range(6)]
        exchange = FakeExchange(trades, [book() for _ in range(6)])
        driver = WebSocketDriver(engine, DriverConfig(watchdog_interval=1e6), exchange=exchange)

        task = asyncio.create_task(driver.run())
        await asyncio.sleep(0.3)
        await driver.stop()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

        assert driver.stats.trades_seen == 6
        assert driver.stats.books_seen == 6
        assert engine._book is not None
        # 30s ticks across 60s bars must complete bars.
        assert driver.stats.bars_completed >= 2

    asyncio.run(run())


def test_crossed_and_empty_books_are_skipped():
    """Venues emit these transiently in fast markets; trading off them is a gift."""
    async def run():
        engine, _ = make_engine()
        bad = [
            {"bids": [], "asks": [[100.0, 1.0]], "timestamp": None},
            {"bids": [[101.0, 1.0]], "asks": [[100.0, 1.0]], "timestamp": None},
            book(100.0, 100.01),
        ]
        exchange = FakeExchange([], bad)
        driver = WebSocketDriver(engine, DriverConfig(watchdog_interval=1e6), exchange=exchange)

        task = asyncio.create_task(driver.run())
        await asyncio.sleep(0.3)
        await driver.stop()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

        assert driver.stats.books_seen == 1
        assert engine._book.bid == 100.0

    asyncio.run(run())


# --- reconnection ----------------------------------------------------------

def test_stream_errors_reconnect_and_resume():
    async def run():
        engine, _ = make_engine()
        base = datetime(2024, 1, 2, tzinfo=UTC)
        script = [
            ConnectionResetError("socket dropped"),
            trade(100.0, base),
            ConnectionResetError("dropped again"),
            trade(101.0, base + timedelta(seconds=90)),
        ]
        exchange = FakeExchange(script, [book() for _ in range(4)])
        driver = WebSocketDriver(
            engine,
            DriverConfig(reconnect_base_delay=0.01, reconnect_max_delay=0.02,
                         watchdog_interval=1e6),
            exchange=exchange,
        )

        task = asyncio.create_task(driver.run())
        await asyncio.sleep(0.5)

        # Checked before shutdown: _shutdown halts deliberately, so asserting
        # afterwards would test the teardown rather than the reconnect.
        assert not engine.state.halted, "a recoverable drop must not halt"
        assert driver.stats.reconnects >= 2
        assert driver.stats.trades_seen == 2      # resumed after each drop

        await driver.stop()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())


def test_exhausted_reconnects_halt_the_engine():
    """A bot that retries forever while holding a position is not safe, it is unattended."""
    async def run():
        engine, _ = make_engine()
        exchange = FakeExchange([ConnectionResetError("down")] * 10, [])
        driver = WebSocketDriver(
            engine,
            DriverConfig(reconnect_base_delay=0.001, reconnect_max_delay=0.002,
                         max_reconnect_attempts=3, watchdog_interval=1e6),
            exchange=exchange,
        )
        await asyncio.wait_for(driver.run(), timeout=5.0)
        assert engine.state.halted
        assert "unrecoverable" in engine.state.halt_reason

    asyncio.run(run())


# --- decision latch --------------------------------------------------------

def test_decisions_latch_rather_than_queue():
    """A backlog of stale decisions is worse than one decision on fresh state."""
    async def run():
        engine, _ = make_engine()
        base = datetime(2024, 1, 2, tzinfo=UTC)

        slow_calls = {"n": 0}

        async def slow_decision():
            slow_calls["n"] += 1
            await asyncio.sleep(0.15)

        engine.on_bar_close = slow_decision

        trades = [trade(100.0 + i, base + timedelta(seconds=i * 70)) for i in range(6)]
        exchange = FakeExchange(trades, [book() for _ in range(6)])
        driver = WebSocketDriver(engine, DriverConfig(watchdog_interval=1e6), exchange=exchange)

        task = asyncio.create_task(driver.run())
        await asyncio.sleep(0.4)
        await driver.stop()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

        # More bars completed than decisions ran: the surplus was coalesced.
        assert driver.stats.bars_completed > slow_calls["n"]
        assert driver.stats.decisions_skipped > 0

    asyncio.run(run())


def test_wedged_decision_halts_instead_of_freezing():
    async def run():
        engine, _ = make_engine()
        base = datetime(2024, 1, 2, tzinfo=UTC)

        async def hang():
            await asyncio.sleep(3600)

        engine.on_bar_close = hang

        trades = [trade(100.0 + i, base + timedelta(seconds=i * 70)) for i in range(3)]
        exchange = FakeExchange(trades, [book() for _ in range(3)])
        driver = WebSocketDriver(
            engine,
            DriverConfig(decision_timeout=0.1, watchdog_interval=1e6),
            exchange=exchange,
        )
        await asyncio.wait_for(driver.run(), timeout=5.0)
        assert engine.state.halted
        assert "timeout" in engine.state.halt_reason

    asyncio.run(run())


# --- shutdown --------------------------------------------------------------

def test_shutdown_halts_engine_and_closes_socket():
    """Leaving a position behind a dead driver is the worst available outcome."""
    async def run():
        engine, _ = make_engine()
        exchange = FakeExchange([], [book()])
        driver = WebSocketDriver(
            engine,
            DriverConfig(max_reconnect_attempts=1, reconnect_base_delay=0.001,
                         watchdog_interval=1e6),
            exchange=exchange,
        )
        driver._owns_exchange = True

        task = asyncio.create_task(driver.run())
        await asyncio.sleep(0.2)
        await driver.stop()
        await asyncio.wait_for(task, timeout=5.0)

        assert engine.state.halted
        assert exchange.closed

    asyncio.run(run())


def test_stats_snapshot_is_serialisable():
    engine, _ = make_engine()
    driver = WebSocketDriver(engine, DriverConfig(), exchange=FakeExchange())
    snapshot = driver.stats.snapshot()
    assert set(snapshot) >= {"trades", "books", "bars", "decisions", "reconnects"}
