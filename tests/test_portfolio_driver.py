"""Fleet driver.

The tests that matter are the teardown paths. Starting drivers is easy;
stopping one that still holds a position without orphaning that position is
where a multi-symbol bot goes wrong.
"""

import asyncio
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from godalgo.data.scanner import MarketScanner, ScanCriteria
from godalgo.execution.broker import PaperBroker
from godalgo.execution.driver import DriverConfig
from godalgo.execution.engine import LiveEngine, LiveEngineConfig
from godalgo.execution.portfolio_driver import PortfolioDriver, PortfolioDriverConfig
from godalgo.portfolio.supervisor import PortfolioSupervisor, SupervisorConfig
from godalgo.strategies.mean_reversion import MeanReversionStrategy
from godalgo.strategies.momentum import MomentumStrategy

N = 800
IDX = pd.date_range("2024-01-01", periods=N, freq="1h", tz="UTC")
T0 = datetime(2024, 6, 1, tzinfo=UTC)


def trending(seed=7, phi=0.35, sigma=0.010):
    rng = np.random.default_rng(seed)
    r = np.zeros(N)
    for i in range(1, N):
        r[i] = phi * r[i - 1] + rng.normal(0, sigma)
    px = 100 * np.exp(np.cumsum(r))
    return pd.DataFrame(
        {"open": px, "high": px * 1.001, "low": px * 0.999, "close": px, "volume": 1.0},
        index=IDX,
    )


class FakeExchange:
    """ccxt.pro stand-in that idles rather than ending its streams."""

    def __init__(self):
        self.has = {"watchTrades": True, "watchOrderBook": True}
        self.closed = False

    async def watch_trades(self, symbol):
        await asyncio.sleep(3600)

    async def watch_order_book(self, symbol, depth=5):
        await asyncio.sleep(3600)

    async def close(self):
        self.closed = True


def build(symbols=("BTC/USDT",), **supervisor_kw):
    history = {s: trending(seed=7 + i) for i, s in enumerate(symbols)}
    tickers = {s: {"quoteVolume": 5e7, "bid": 99.99, "ask": 100.01} for s in symbols}
    broker = PaperBroker(starting_equity=100_000.0)

    def make_engine(symbol):
        return LiveEngine(
            broker, MomentumStrategy(), MeanReversionStrategy(),
            LiveEngineConfig(symbol=symbol, bar_seconds=60),
        )

    supervisor = PortfolioSupervisor(
        SupervisorConfig(**{"max_concurrent": 3, **supervisor_kw}),
        MarketScanner(ScanCriteria(max_candidates=3)),
        history=lambda: history,
        tickers=lambda: tickers,
        make_engine=make_engine,
        equity=lambda: 100_000.0,
    )
    driver = PortfolioDriver(
        supervisor,
        PortfolioDriverConfig(rotation_poll_seconds=0.05),
        driver_config=DriverConfig(watchdog_interval=1e6),
        seed_history=lambda symbol, n: history.get(symbol),
        exchange=FakeExchange(),
    )
    return driver, supervisor, history


# --- startup ---------------------------------------------------------------

def test_fleet_starts_a_driver_per_selected_symbol():
    async def run():
        driver, supervisor, _ = build()
        task = asyncio.create_task(driver.run())
        await asyncio.sleep(0.3)
        started = set(driver.drivers)
        await driver.stop()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        return started, set(supervisor.engines)

    started, engines = asyncio.run(run())
    assert started, "no drivers started"
    assert started <= engines


def test_engines_receive_the_portfolio_hooks():
    """Without these each engine sizes against the full account and the fleet
    is collectively unbounded."""
    async def run():
        driver, supervisor, _ = build()
        task = asyncio.create_task(driver.run())
        await asyncio.sleep(0.3)
        engines = list(supervisor.engines.values())
        await driver.stop()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        return engines

    engines = asyncio.run(run())
    assert engines
    for engine in engines:
        assert engine.target_clamp is not None
        assert engine.buying_power is not None


def test_new_engines_are_seeded_so_they_are_not_blind():
    async def run():
        driver, supervisor, _ = build()
        task = asyncio.create_task(driver.run())
        await asyncio.sleep(0.3)
        counts = [e.bars.n_complete for e in supervisor.engines.values()]
        await driver.stop()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        return counts

    assert all(c > 0 for c in asyncio.run(run()))


def test_unknown_exchange_is_rejected():
    _, supervisor, _ = build()
    with pytest.raises(ValueError, match="not a ccxt.pro exchange"):
        PortfolioDriver(supervisor, PortfolioDriverConfig(exchange_id="nope"))


# --- teardown --------------------------------------------------------------

def test_stopping_a_symbol_flattens_before_teardown():
    """Cancelling first drops the only thing managing the position's stop."""
    async def run():
        driver, supervisor, _ = build()
        task = asyncio.create_task(driver.run())
        await asyncio.sleep(0.3)
        symbol = next(iter(driver.drivers))
        engine = supervisor.engines[symbol]
        engine.state.current_weight = 0.4          # pretend we hold something
        flattened = {"called": False}

        async def record():
            flattened["called"] = True
            engine.state.current_weight = 0.0

        engine.flatten = record
        await driver._stop_symbol(symbol, flatten=True)
        result = (flattened["called"], symbol in driver.drivers)
        await driver.stop()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        return result

    called, still_running = asyncio.run(run())
    assert called is True
    assert still_running is False


def test_a_failed_flatten_keeps_the_driver_running():
    """The driver is what manages the position's stop; dropping it orphans the
    position."""
    async def run():
        driver, supervisor, _ = build()
        task = asyncio.create_task(driver.run())
        await asyncio.sleep(0.3)
        symbol = next(iter(driver.drivers))
        engine = supervisor.engines[symbol]
        engine.state.current_weight = 0.4

        async def explode():
            raise RuntimeError("venue down")

        engine.flatten = explode
        await driver._stop_symbol(symbol, flatten=True)
        result = (symbol in driver.drivers, driver.state.failed.get(symbol))
        await driver.stop()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        return result

    still_running, failure = asyncio.run(run())
    assert still_running is True, "driver dropped while holding a position"
    assert failure == "flatten failed"


def test_shutdown_stops_everything_and_closes_the_socket():
    async def run():
        driver, _, _ = build()
        task = asyncio.create_task(driver.run())
        await asyncio.sleep(0.3)
        driver._owns_exchange = True
        await driver.stop()
        await asyncio.sleep(0.2)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        return driver.drivers, driver._exchange.closed

    remaining, closed = asyncio.run(run())
    assert remaining == {}
    assert closed is True


def test_a_failed_scan_leaves_the_fleet_untouched():
    """Symbols already trading must not be disturbed by a scan problem."""
    async def run():
        driver, supervisor, _ = build()
        task = asyncio.create_task(driver.run())
        await asyncio.sleep(0.3)
        before = set(driver.drivers)

        def explode():
            raise RuntimeError("feed down")

        supervisor.history = explode
        await driver._rotate()
        after = set(driver.drivers)
        await driver.stop()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        return before, after

    before, after = asyncio.run(run())
    assert after == before


def test_reap_notices_an_exited_driver():
    """An unnoticed exit leaves a symbol looking active with nothing trading it."""
    async def run():
        driver, _, _ = build()
        task = asyncio.create_task(driver.run())
        await asyncio.sleep(0.3)
        symbol = next(iter(driver.drivers))
        driver._tasks[symbol].cancel()
        await asyncio.sleep(0.05)
        driver._reap()
        result = (symbol in driver.drivers, symbol in driver.state.failed)
        await driver.stop()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        return result

    still_listed, recorded = asyncio.run(run())
    assert still_listed is False
    assert recorded is True


# --- exposure --------------------------------------------------------------

def test_clamp_binds_across_symbols_through_the_hook():
    async def run():
        driver, supervisor, _ = build(
            symbols=("BTC/USDT",), max_gross_exposure=0.5, max_concurrent=2,
        )
        task = asyncio.create_task(driver.run())
        await asyncio.sleep(0.3)
        symbol = next(iter(supervisor.engines))
        engine = supervisor.engines[symbol]
        clamped = engine.target_clamp(symbol, 5.0)
        await driver.stop()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        return clamped

    # Per-symbol budget is the gross cap divided by the concurrency limit.
    assert asyncio.run(run()) == pytest.approx(0.25)


def test_snapshot_is_serialisable():
    import json

    async def run():
        driver, _, _ = build()
        task = asyncio.create_task(driver.run())
        await asyncio.sleep(0.3)
        snap = driver.snapshot()
        await driver.stop()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        return snap

    json.dumps(asyncio.run(run()))
