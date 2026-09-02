"""The allocator actually deciding something.

It was a tested library that nothing consulted, which is the position the mode
switch was in before it was wired: every part correct, none of them connected.
Isolation is what let that ship, so these tests run from the entry point
inward.

The invariant that matters most: the book may only ever REDUCE. An allocator
that could enlarge a position the strategy did not ask for would be taking a
view of its own, which is not what a sizing layer is for.
"""

from __future__ import annotations

import numpy as np

from godalgo.portfolio.book import BookLimits
from godalgo.research.forward import ForwardRecord
from godalgo.ui.book_state import BookManager


def _walk(n: int = 300, start: float = 100.0, vol: float = 0.0006,
          seed: int = 0) -> list[float]:
    """A price path at the feed's five-second cadence.

    The per-tick figure is small on purpose. An earlier version of these tests
    used 1% moves per tick, which annualises to several hundred percent, and
    the allocator correctly refused to hold any of it -- the test data was
    unrealistic, and it hid a real annualisation bug behind a plausible
    looking empty book.
    """
    rng = np.random.default_rng(seed)
    return list(start * np.exp(np.cumsum(rng.normal(0, vol, n))))


def _feed(manager: BookManager, prices: dict[str, list[float]]) -> None:
    for i in range(max(len(v) for v in prices.values())):
        manager.observe({s: v[i] for s, v in prices.items() if i < len(v)})


# --------------------------------------------------------------------------
# the cap can only reduce
# --------------------------------------------------------------------------

def test_the_book_never_enlarges_a_position():
    manager = BookManager()
    _feed(manager, {"BTC/USD": _walk(), "SPY": _walk(vol=0.00015, seed=2)})
    manager.rebuild(["BTC/USD", "SPY"], equity=10_000, peak_equity=10_000)

    permitted, _ = manager.permitted_weight("BTC/USD", 0.001)
    assert permitted == 0.001, "a small request must pass through untouched"


def test_an_oversized_request_is_cut_to_the_book():
    manager = BookManager()
    _feed(manager, {"BTC/USD": _walk(), "SPY": _walk(vol=0.00015, seed=2)})
    manager.rebuild(["BTC/USD", "SPY"], equity=10_000, peak_equity=10_000)

    permitted, why = manager.permitted_weight("BTC/USD", 0.95)
    assert permitted < 0.95
    assert "BTC/USD" in why


def test_the_cut_keeps_the_direction():
    manager = BookManager()
    _feed(manager, {"BTC/USD": _walk()})
    manager.rebuild(["BTC/USD"], equity=10_000, peak_equity=10_000)
    permitted, _ = manager.permitted_weight("BTC/USD", -0.95)
    assert permitted < 0


def test_an_uncomputed_book_does_not_flatten_the_engine():
    """A book that has not run must not silently stop the bot trading."""
    permitted, why = BookManager().permitted_weight("BTC/USD", 0.5)
    assert permitted == 0.5
    assert "no allocation computed yet" in why


def test_a_symbol_cut_from_the_book_gets_zero_with_a_reason():
    manager = BookManager()
    _feed(manager, {"BTC/USD": _walk()})
    manager.rebuild(["BTC/USD"], equity=10_000, peak_equity=10_000)
    permitted, why = manager.permitted_weight("NOT_IN_BOOK", 0.5)
    assert permitted == 0.0
    assert "not in the book" in why


# --------------------------------------------------------------------------
# volatility, measured rather than assumed
# --------------------------------------------------------------------------

def test_a_quiet_symbol_measures_lower_volatility():
    manager = BookManager()
    _feed(manager, {"CALM": _walk(vol=0.0001, seed=1),
                    "WILD": _walk(vol=0.002, seed=1)})
    assert manager.volatility("CALM") < manager.volatility("WILD")


def test_too_little_history_is_none_rather_than_a_guess():
    manager = BookManager()
    for price in _walk(5):
        manager.observe({"X": price})
    assert manager.volatility("X") is None


def test_an_unmeasurable_symbol_falls_back_to_the_median_not_an_invention():
    """Inverse volatility on a noisy volatility is a random weighting with a
    respectable name."""
    manager = BookManager()
    _feed(manager, {"A": _walk(seed=1), "B": _walk(seed=2)})
    manager.observe({"NEW": 100.0})
    candidates = {c.symbol: c for c in manager.candidates(["A", "B", "NEW"])}
    others = sorted([candidates["A"].volatility, candidates["B"].volatility])
    assert others[0] <= candidates["NEW"].volatility <= others[1]


def test_history_is_bounded():
    """An unbounded buffer on a once-a-second loop is a slow leak."""
    manager = BookManager()
    manager._max_history = 50
    for price in _walk(500):
        manager.observe({"X": price})
    assert len(manager._history["X"]) == 50


# --------------------------------------------------------------------------
# the forward record drives sizing
# --------------------------------------------------------------------------

def test_allocation_scales_with_the_forward_record_not_the_backtest(tmp_path):
    """A strategy's allocation is a function of out-of-sample track length."""
    import datetime as dt

    prices = {"BTC/USD": _walk(), "SPY": _walk(vol=0.00015, seed=2)}

    fresh = BookManager(forward=ForwardRecord(tmp_path / "a.jsonl"))
    _feed(fresh, prices)
    fresh.rebuild(["BTC/USD", "SPY"], equity=100_000, peak_equity=100_000)

    record = ForwardRecord(tmp_path / "b.jsonl")
    for i in range(200):
        record.record_day(strategy="m", symbol="SPY", pnl=1.0, ret=0.001,
                          day=dt.date(2026, 1, 1) + dt.timedelta(days=i))
    proven = BookManager(forward=record)
    _feed(proven, prices)
    proven.rebuild(["BTC/USD", "SPY"], equity=100_000, peak_equity=100_000)

    assert proven.result.gross > fresh.result.gross * 2


# --------------------------------------------------------------------------
# the drawdown ladder reaches the engine
# --------------------------------------------------------------------------

def test_a_drawdown_reduces_what_the_engine_may_hold():
    manager = BookManager()
    _feed(manager, {"BTC/USD": _walk(), "SPY": _walk(vol=0.00015, seed=2)})

    manager.rebuild(["BTC/USD", "SPY"], equity=10_000, peak_equity=10_000)
    calm, _ = manager.permitted_weight("BTC/USD", 0.95)

    manager.rebuild(["BTC/USD", "SPY"], equity=8_800, peak_equity=10_000)
    hurt, _ = manager.permitted_weight("BTC/USD", 0.95)

    assert hurt < calm


def test_a_deep_drawdown_flattens_the_book():
    manager = BookManager()
    _feed(manager, {"BTC/USD": _walk()})
    manager.rebuild(["BTC/USD"], equity=8_000, peak_equity=10_000)
    assert manager.result.gross == 0.0
    permitted, why = manager.permitted_weight("BTC/USD", 0.5)
    assert permitted == 0.0


# --------------------------------------------------------------------------
# it must never break the loop it runs on
# --------------------------------------------------------------------------

def test_a_failing_rebuild_returns_the_last_book_rather_than_raising():
    manager = BookManager(limits=BookLimits())
    _feed(manager, {"BTC/USD": _walk()})
    good = manager.rebuild(["BTC/USD"], equity=10_000, peak_equity=10_000)

    manager.candidates = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    again = manager.rebuild(["BTC/USD"], equity=10_000, peak_equity=10_000)
    assert again is good


def test_the_display_payload_distinguishes_not_yet_from_nothing_qualified():
    """'No matches' is only true when something was actually scanned."""
    payload = BookManager().to_dict()
    assert payload["state"] == "not_computed"
    assert "not yet" in payload["detail"]


# --------------------------------------------------------------------------
# from the entry point inward
# --------------------------------------------------------------------------

def test_the_terminal_is_built_with_a_book_and_a_record(tmp_path, monkeypatch):
    from godalgo.ui import credentials as credentials_module
    from godalgo.ui.server import build_terminal

    monkeypatch.setattr(credentials_module, "_DEFAULT_DIR", tmp_path)
    bridge = build_terminal()
    assert bridge.book is not None, "the allocator must be attached"
    assert bridge.forward is not None, "the forward record must be attached"
    assert bridge.session.book is bridge.book, "the session must consult it"


def test_the_snapshot_carries_the_book_headline(tmp_path, monkeypatch):
    from godalgo.ui import credentials as credentials_module
    from godalgo.ui.server import build_terminal

    monkeypatch.setattr(credentials_module, "_DEFAULT_DIR", tmp_path)
    bridge = build_terminal()
    body = bridge.snapshot().to_dict()
    assert "book" in body
    assert body["book"]["state"] == "not_computed"


def test_the_headline_is_small_enough_to_send_every_second(tmp_path):
    """The full book carries a line and an explanation per position. Sending
    all of it once a second is a frame far larger than the panel needs."""
    import json

    from godalgo.ui.state import _book_headline

    manager = BookManager()
    _feed(manager, {f"S{i}": _walk(seed=i) for i in range(12)})
    manager.rebuild([f"S{i}" for i in range(12)], equity=1e6, peak_equity=1e6)

    full = len(json.dumps(manager.to_dict()))
    headline = len(json.dumps(_book_headline(manager.to_dict())))
    assert headline < full / 2
    assert headline < 500


def test_the_book_endpoints_answer(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from godalgo.ui import credentials as credentials_module
    from godalgo.ui.server import build_terminal, create_app

    monkeypatch.setattr(credentials_module, "_DEFAULT_DIR", tmp_path)
    bridge = build_terminal()
    bridge.market_feed_enabled = False
    bridge.session = None
    with TestClient(create_app(bridge)) as client:
        assert client.get("/api/book").json()["state"] == "not_computed"
        forward = client.get("/api/forward").json()
        assert list(forward)[0] == "out_of_sample"
