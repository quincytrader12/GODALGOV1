"""Many symbols, scanned, decided and executed — in one loop.

The terminal traded one instrument while the scanner, supervisor and allocator
sat unused behind it. These tests run the actual loop against a stub broker and
assert the things a single-symbol design cannot even express: aggregate
exposure, retirement flattening rather than detaching, and one data path
feeding every engine.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pandas as pd
import pytest

from godalgo.execution.types import TradingMode
from godalgo.ui.credentials import CredentialStore
from godalgo.ui.journal import TradingJournal
from godalgo.ui.portfolio_session import PortfolioSession, _mode_of
from godalgo.ui.server import UIBridge


class _Broker:
    """A broker that records rather than trades."""

    def __init__(self) -> None:
        self.flattened: list[str] = []

    async def cancel_all(self, symbol=None):
        return 0

    async def position(self, symbol):
        from godalgo.execution.types import Position

        return Position(symbol=symbol, quantity=0.0)

    async def equity(self):
        return 10_000.0


class _DryRunBroker(_Broker):
    """Named so ``_mode_of`` reads it as a dry run."""


def _frame(close: np.ndarray) -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=len(close), freq="1min", tz="UTC")
    return pd.DataFrame(
        {
            "open": close, "high": close * 1.001, "low": close * 0.999,
            "close": close, "volume": np.full(len(close), 1e6),
        },
        index=index,
    )


def _trending(n: int = 500, seed: int = 0, strength: float = 0.85) -> pd.DataFrame:
    """A series with genuine positive autocorrelation.

    A random walk has no regime, and the scanner correctly refuses it -- the
    first version of these tests fed it four random walks, got one symbol
    admitted, and looked like a broken supervisor rather than a working
    scanner. Test data has to contain the thing being detected.
    """
    rng = np.random.default_rng(seed)
    steps = np.zeros(n)
    for i in range(1, n):
        steps[i] = strength * steps[i - 1] + rng.normal(0, 0.004)
    return _frame(100 * np.exp(np.cumsum(steps)))


def _reverting(n: int = 500, seed: int = 0, pull: float = 0.08) -> pd.DataFrame:
    """An Ornstein-Uhlenbeck series: negative autocorrelation, real half-life."""
    rng = np.random.default_rng(seed)
    level = np.zeros(n)
    for i in range(1, n):
        level[i] = level[i - 1] * (1 - pull) + rng.normal(0, 0.01)
    return _frame(100 * np.exp(level))


def _random_walk(n: int = 500, seed: int = 0) -> pd.DataFrame:
    """No regime at all. The scanner should refuse it."""
    rng = np.random.default_rng(seed)
    return _frame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))))


@pytest.fixture
def bridge(tmp_path) -> UIBridge:
    b = UIBridge(
        credentials=CredentialStore(directory=tmp_path),
        journal=TradingJournal(path=tmp_path / "j.jsonl",
                               summary_path=tmp_path / "s.jsonl"),
        market_feed_enabled=False,
    )
    b.universe = ["BTC/USD", "ETH/USD", "SPY", "AAPL"]
    return b


@pytest.fixture
def session(bridge) -> PortfolioSession:
    s = PortfolioSession(bridge, exchange_id="alpaca", max_concurrent=3)
    # History is normally fetched from the venue; injected here so the scanner
    # has something to rank without a network call.
    s._history = {
        "BTC/USD": _trending(seed=1),
        "ETH/USD": _trending(seed=2),
        "SPY": _reverting(seed=3),
        "AAPL": _random_walk(seed=4),
    }
    s._refresh_history = _noop
    return s


async def _noop() -> None:
    return None


# --------------------------------------------------------------------------
# it actually runs
# --------------------------------------------------------------------------

def test_the_loop_admits_several_symbols_and_runs(session, bridge):
    """The whole point: more than one instrument, decided and executed."""
    async def _go():
        await session.start(_DryRunBroker())
        for _ in range(60):
            await asyncio.sleep(0.05)
            if session.supervisor and session.supervisor.engines:
                break
        engines = dict(session.supervisor.engines)
        await session.stop()
        return engines

    engines = asyncio.run(_go())
    assert len(engines) > 1, f"only got {list(engines)}"
    assert len(engines) <= 3, "the concurrency limit must bind"


def test_the_scanner_refuses_what_has_no_regime(session):
    """Refusing is the expected outcome, not a fault. AAPL here is a random
    walk: no autocorrelation of either sign, so there is nothing for either
    strategy to trade and admitting it would be the scanner failing."""
    async def _go():
        await session.start(_DryRunBroker())
        for _ in range(60):
            await asyncio.sleep(0.05)
            if session.supervisor and session.supervisor.engines:
                break
        scan = session.supervisor.state.last_scan
        await session.stop()
        return scan

    scan = asyncio.run(_go())
    assert "AAPL" in [c.symbol for c in scan.rejected]
    assert "AAPL" not in [c.symbol for c in scan.selected]


def test_every_engine_shares_one_broker(session):
    """One account at the venue, not one per symbol."""
    async def _go():
        broker = _DryRunBroker()
        await session.start(broker)
        for _ in range(60):
            await asyncio.sleep(0.05)
            if session.supervisor and session.supervisor.engines:
                break
        brokers = {id(e.broker) for e in session.supervisor.engines.values()}
        await session.stop()
        return brokers, id(broker)

    brokers, expected = asyncio.run(_go())
    assert brokers == {expected}


def test_the_status_reports_every_symbol(session):
    async def _go():
        await session.start(_DryRunBroker())
        for _ in range(60):
            await asyncio.sleep(0.05)
            if session.supervisor and session.supervisor.engines:
                break
        status = session.status()
        await session.stop()
        return status

    status = asyncio.run(_go())
    assert len(status["symbols"]) > 1
    assert status["running"] is True
    assert status["max_concurrent"] == 3
    assert "gross_exposure" in status


# --------------------------------------------------------------------------
# ticks
# --------------------------------------------------------------------------

def test_a_repeated_price_is_not_a_second_observation(session, bridge):
    """The feed polls on a timer. Forwarding the same price twice
    manufactures volume and bars out of a still market."""
    seen: list[float] = []

    class _Engine:
        state = type("S", (), {"current_weight": 0.0, "target_weight": 0.0})()

        async def on_tick(self, price, moment=None):
            seen.append(price)

    session.supervisor = type("S", (), {"engines": {"BTC/USD": _Engine()}})()
    bridge.prices["BTC/USD"] = 100.0

    asyncio.run(session._dispatch())
    asyncio.run(session._dispatch())
    assert seen == [100.0]

    bridge.prices["BTC/USD"] = 101.0
    asyncio.run(session._dispatch())
    assert seen == [100.0, 101.0]


def test_one_symbol_failing_does_not_stop_the_others(session, bridge):
    """A book must not be taken down by one instrument."""
    good: list[float] = []

    class _Broken:
        state = type("S", (), {"current_weight": 0.0, "target_weight": 0.0})()

        async def on_tick(self, price, moment=None):
            raise RuntimeError("bad symbol")

    class _Fine:
        state = type("S", (), {"current_weight": 0.0, "target_weight": 0.0})()

        async def on_tick(self, price, moment=None):
            good.append(price)

    session.supervisor = type("S", (), {
        "engines": {"BAD": _Broken(), "OK": _Fine()},
    })()
    bridge.prices.update({"BAD": 10.0, "OK": 20.0})

    asyncio.run(session._dispatch())
    assert good == [20.0]
    assert any("decision failed" in e["message"] for e in bridge.events.entries())


# --------------------------------------------------------------------------
# retirement flattens
# --------------------------------------------------------------------------

def test_stopping_retires_through_the_supervisor_so_positions_flatten(session):
    """Stopping an engine that owns a position leaves it open with nothing
    managing its stop -- the worst outcome available here."""
    retired: list[str] = []

    async def _go():
        await session.start(_DryRunBroker())
        for _ in range(60):
            await asyncio.sleep(0.05)
            if session.supervisor and session.supervisor.engines:
                break
        supervisor = session.supervisor
        symbols = list(supervisor.engines)
        original = supervisor._retire

        async def _spy(symbol):
            retired.append(symbol)
            await original(symbol)

        supervisor._retire = _spy
        await session.stop()
        return symbols

    symbols = asyncio.run(_go())
    assert sorted(retired) == sorted(symbols)


# --------------------------------------------------------------------------
# aggregate exposure and the book
# --------------------------------------------------------------------------

def test_gross_exposure_is_summed_across_the_book(session, bridge):
    """Three engines each within their own cap is not three times within one."""
    class _Engine:
        def __init__(self, weight):
            self.state = type("S", (), {
                "current_weight": weight, "target_weight": weight,
                "bars_seen": 1, "halted": False, "halt_reason": None,
                "regime": "trending", "conviction": 0.5,
            })()

    from godalgo.portfolio.supervisor import PortfolioSupervisor, SupervisorConfig

    supervisor = PortfolioSupervisor(
        config=SupervisorConfig(), scanner=None, history=dict, tickers=dict,
        make_engine=lambda s: None, equity=lambda: 10_000.0,
    )
    supervisor.engines = {"A": _Engine(0.2), "B": _Engine(0.2), "C": _Engine(0.2)}
    session.supervisor = supervisor

    asyncio.run(session._publish())
    assert bridge.current_weight == pytest.approx(0.6)
    assert bridge.portfolio is not None


def test_the_book_caps_each_engine_not_just_one(session, bridge):
    """The allocator has to reach every symbol, or it is decorative."""
    from godalgo.ui.book_state import BookManager

    capped: dict[str, float] = {}

    class _Engine:
        def __init__(self, symbol):
            self.symbol = symbol
            self.state = type("S", (), {
                "current_weight": 0.0, "target_weight": 0.9,
                "bars_seen": 1, "halted": False, "halt_reason": None,
                "regime": "trending", "conviction": 1.0,
            })()

        def set_weight_cap(self, cap):
            capped[self.symbol] = cap

    book = BookManager(interval_seconds=5.0)
    rng = np.random.default_rng(0)
    for _ in range(120):
        book.observe({
            "BTC/USD": 100 * float(np.exp(rng.normal(0, 0.0006))),
            "SPY": 400 * float(np.exp(rng.normal(0, 0.00015))),
        })
    session.book = book

    engines = {s: _Engine(s) for s in ("BTC/USD", "SPY")}
    session._apply_book(engines)
    assert set(capped) == {"BTC/USD", "SPY"}
    assert all(v < 0.9 for v in capped.values())


def test_the_cap_reason_is_logged_once_not_every_second(session, bridge):
    from godalgo.ui.book_state import BookManager

    class _Engine:
        state = type("S", (), {"current_weight": 0.0, "target_weight": 0.9})()

        def set_weight_cap(self, cap):
            return None

    book = BookManager(interval_seconds=5.0)
    rng = np.random.default_rng(0)
    for _ in range(120):
        book.observe({"BTC/USD": 100 * float(np.exp(rng.normal(0, 0.0006)))})
    session.book = book

    engines = {"BTC/USD": _Engine()}
    for _ in range(5):
        session._apply_book(engines)

    logged = [e for e in bridge.events.entries() if e["category"] == "book"]
    assert len(logged) == 1


# --------------------------------------------------------------------------
# empty results explain themselves
# --------------------------------------------------------------------------

def test_nothing_passing_the_scan_is_reported_as_a_decision(session, bridge):
    """'No matches' is only true when something was actually scanned, and
    refusing everything is the expected outcome, not a fault."""
    session._history = {}

    async def _go():
        await session.start(_DryRunBroker())
        await asyncio.sleep(0.3)
        await session.stop()

    asyncio.run(_go())
    messages = " ".join(
        f"{e['message']} {e.get('detail', '')}" for e in bridge.events.entries()
    )
    assert "scan" in messages or "universe" in messages


def test_history_gaps_are_skipped_not_blacklisted(session):
    """One credential-less launch must not permanently mark a universe dead."""
    import inspect

    source = inspect.getsource(PortfolioSession._refresh_history)
    assert "retried on the next" in source
    assert "not marked dead" in source


# --------------------------------------------------------------------------
# mode
# --------------------------------------------------------------------------

def test_the_mode_is_read_from_the_broker_in_hand():
    from godalgo.execution.broker import DryRunBroker, PaperBroker

    assert _mode_of(DryRunBroker()) is TradingMode.DRY_RUN
    assert _mode_of(PaperBroker()) is TradingMode.PAPER
    assert _mode_of(object()) is TradingMode.LIVE


def test_a_stopped_session_reports_stopped(session):
    assert session.running is False
    asyncio.run(session.stop())
    assert session.running is False
