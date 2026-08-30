"""Per-position stops and risk-based sizing.

The ratchet test is the important one. A stop that can widen is not a stop, and
loosening one as price approaches is the single most common way a bounded loss
becomes an account-ending one.
"""

import pytest

from godalgo.portfolio.sizing import GrowthConfig, risk_based_size
from godalgo.risk.stops import ExitReason, StopConfig, StopManager


def mgr(**kw):
    return StopManager(StopConfig(**kw))


# --- initial stop ----------------------------------------------------------

def test_initial_stop_sits_at_the_configured_atr_distance():
    m = mgr(initial_atr=2.0)
    s = m.open("BTC/USDT", "long", entry_price=100.0, atr=2.0)
    assert s.stop_price == pytest.approx(96.0)


def test_short_stop_sits_above_entry():
    m = mgr(initial_atr=2.0)
    s = m.open("BTC/USDT", "short", entry_price=100.0, atr=2.0)
    assert s.stop_price == pytest.approx(104.0)


def test_a_stop_cannot_be_placed_without_volatility():
    """Defaulting to a percentage would silently swap in a different risk model."""
    m = mgr()
    with pytest.raises(ValueError, match="valid ATR"):
        m.open("BTC/USDT", "long", 100.0, atr=0.0)


def test_initial_stop_fires():
    m = mgr(initial_atr=2.0)
    m.open("BTC/USDT", "long", 100.0, 2.0)
    assert m.update("BTC/USDT", 99.0) is None
    assert m.update("BTC/USDT", 95.5) is ExitReason.INITIAL_STOP


# --- break-even ------------------------------------------------------------

def test_stop_moves_to_break_even_after_enough_favourable_travel():
    m = mgr(breakeven_trigger_atr=1.5, breakeven_offset_atr=0.15)
    m.open("BTC/USDT", "long", 100.0, 2.0)
    m.update("BTC/USDT", 101.0)
    assert m.positions["BTC/USDT"].at_breakeven is False
    m.update("BTC/USDT", 103.0)          # 1.5 ATR
    state = m.positions["BTC/USDT"]
    assert state.at_breakeven is True
    assert state.stop_price > 100.0, "break-even must clear entry, not sit on it"


def test_break_even_offset_covers_costs():
    """A stop exactly at entry exits at a loss once fees are counted."""
    m = mgr(breakeven_offset_atr=0.2)
    m.open("BTC/USDT", "long", 100.0, 2.0)
    m.update("BTC/USDT", 104.0)
    assert m.positions["BTC/USDT"].stop_price == pytest.approx(100.4)


def test_break_even_exit_is_labelled_distinctly():
    m = mgr(trail_activate_atr=99.0)     # keep trailing out of it
    m.open("BTC/USDT", "long", 100.0, 2.0)
    m.update("BTC/USDT", 103.0)
    assert m.update("BTC/USDT", 100.0) is ExitReason.BREAK_EVEN


# --- trailing --------------------------------------------------------------

def test_trailing_only_starts_clear_of_the_noise_band():
    """Trailing from entry converts winners into scratches."""
    m = mgr(trail_activate_atr=2.0)
    m.open("BTC/USDT", "long", 100.0, 2.0)
    m.update("BTC/USDT", 101.0)
    assert m.positions["BTC/USDT"].trailing is False
    m.update("BTC/USDT", 104.0)
    assert m.positions["BTC/USDT"].trailing is True


def test_trailing_stop_follows_price_up():
    m = mgr(trail_atr=2.5, trail_activate_atr=2.0)
    m.open("BTC/USDT", "long", 100.0, 2.0)
    m.update("BTC/USDT", 104.0)
    first = m.stop_for("BTC/USDT")
    m.update("BTC/USDT", 110.0)
    assert m.stop_for("BTC/USDT") > first


def test_the_ratchet_holds():
    """A stop must never move against the position.

    Widening a stop because price is approaching it is the mechanism by which
    a small controlled loss becomes an unbounded one.
    """
    m = mgr()
    m.open("BTC/USDT", "long", 100.0, 2.0)
    m.update("BTC/USDT", 112.0)
    high = m.stop_for("BTC/USDT")
    for price in (108.0, 105.0, 102.0):
        m.update("BTC/USDT", price)
        assert m.stop_for("BTC/USDT") >= high, "stop loosened"


def test_ratchet_holds_for_shorts():
    m = mgr()
    m.open("BTC/USDT", "short", 100.0, 2.0)
    m.update("BTC/USDT", 88.0)
    low = m.stop_for("BTC/USDT")
    for price in (92.0, 95.0, 98.0):
        m.update("BTC/USDT", price)
        assert m.stop_for("BTC/USDT") <= low, "stop loosened"


def test_trailing_stop_exit_is_labelled():
    m = mgr(trail_atr=2.0, trail_activate_atr=2.0)
    m.open("BTC/USDT", "long", 100.0, 2.0)
    m.update("BTC/USDT", 110.0)
    assert m.update("BTC/USDT", 104.0) is ExitReason.TRAILING_STOP


# --- optional exits --------------------------------------------------------

def test_take_profit_fires():
    m = mgr(take_profit_atr=4.0)
    m.open("BTC/USDT", "long", 100.0, 2.0)
    assert m.update("BTC/USDT", 108.5) is ExitReason.TAKE_PROFIT


def test_take_profit_must_exceed_the_risk():
    with pytest.raises(ValueError, match="must exceed initial_atr"):
        StopConfig(initial_atr=2.0, take_profit_atr=1.0)


def test_time_stop_fires():
    m = mgr(max_bars=3)
    m.open("BTC/USDT", "long", 100.0, 2.0)
    for _ in range(2):
        assert m.update("BTC/USDT", 100.5) is None
    assert m.update("BTC/USDT", 100.5) is ExitReason.TIME_STOP


def test_closing_drops_tracking():
    m = mgr()
    m.open("BTC/USDT", "long", 100.0, 2.0)
    assert m.close("BTC/USDT") is not None
    assert m.update("BTC/USDT", 1.0) is None


# --- risk-based sizing -----------------------------------------------------

def test_every_trade_risks_the_same_regardless_of_stop_width():
    """The point of sizing from the stop rather than from notional."""
    g = GrowthConfig(risk_per_trade=0.01)
    wide = risk_based_size(10_000, 100.0, 96.0, growth=g)
    tight = risk_based_size(10_000, 100.0, 99.0, growth=g)
    assert wide * 4.0 == pytest.approx(tight * 1.0)
    assert wide * abs(100 - 96) == pytest.approx(100.0)
    assert tight * abs(100 - 99) == pytest.approx(100.0)


def test_sizing_compounds_with_equity():
    g = GrowthConfig()
    small = risk_based_size(10_000, 100.0, 99.0, growth=g)
    large = risk_based_size(20_000, 100.0, 99.0, growth=g)
    assert large == pytest.approx(small * 2)


def test_buying_power_caps_the_order():
    g = GrowthConfig(max_buying_power_fraction=0.95)
    capped = risk_based_size(10_000, 100.0, 99.0, buying_power=5_000, growth=g)
    assert capped == pytest.approx(47.5)


def test_no_buying_power_means_no_trade():
    assert risk_based_size(10_000, 100.0, 99.0, buying_power=0.0) == 0.0


def test_an_undefined_stop_returns_no_position():
    """No stop distance means unbounded risk; sizing it would be a guess."""
    assert risk_based_size(10_000, 100.0, 100.0) == 0.0


def test_drawdown_reduces_risk_beyond_compounding():
    g = GrowthConfig(risk_per_trade=0.01, drawdown_derisk=True)
    assert g.effective_risk(0.0) > g.effective_risk(0.10) > g.effective_risk(0.30)


def test_derisking_has_a_floor():
    """The system must keep enough size to trade its way back."""
    g = GrowthConfig(risk_per_trade=0.01, derisk_floor=0.35)
    assert g.effective_risk(0.9) == pytest.approx(0.01 * 0.35)


def test_risk_ceiling_is_enforced():
    with pytest.raises(ValueError):
        GrowthConfig(risk_per_trade=0.05, max_risk_per_trade=0.02)


def test_conviction_scales_within_the_cap():
    g = GrowthConfig()
    full = risk_based_size(10_000, 100.0, 99.0, growth=g, conviction=1.0)
    half = risk_based_size(10_000, 100.0, 99.0, growth=g, conviction=0.5)
    assert half == pytest.approx(full * 0.5)


# --- engine integration ----------------------------------------------------

def _engine():
    from godalgo.execution.broker import PaperBroker
    from godalgo.execution.engine import LiveEngine, LiveEngineConfig
    from godalgo.strategies.mean_reversion import MeanReversionStrategy
    from godalgo.strategies.momentum import MomentumStrategy

    return LiveEngine(
        PaperBroker(starting_equity=10_000.0),
        MomentumStrategy(), MeanReversionStrategy(),
        LiveEngineConfig(symbol="BTC/USDT", bar_seconds=60),
    )


def _bars(n=400, price=100.0):
    import numpy as np
    import pandas as pd

    idx = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    px = np.full(n, price) + np.linspace(0, 4, n)
    return pd.DataFrame(
        {"open": px, "high": px * 1.004, "low": px * 0.996, "close": px, "volume": 1.0},
        index=idx,
    )


def test_engine_places_a_stop_once_it_holds_a_position():
    engine = _engine()
    engine.state.current_weight = 0.3
    engine._check_stop(_bars())
    assert "BTC/USDT" in engine.stops.positions


def test_engine_drops_the_stop_when_flat():
    engine = _engine()
    engine.state.current_weight = 0.3
    engine._check_stop(_bars())
    engine.state.current_weight = 0.0
    engine._check_stop(_bars())
    assert "BTC/USDT" not in engine.stops.positions


def test_a_direction_flip_replaces_the_stop():
    """Regression, found by running a paper session.

    A stop belongs to the position open when it was placed. After a flip it
    sits on the wrong side of the market and can never fire, leaving the new
    position completely unprotected. Weight does not always pass cleanly
    through zero between trades, so a flat check alone misses this -- in the
    session that surfaced it, one stop was carried across 17 round trips and
    the win rate was 6%.
    """
    engine = _engine()
    bars = _bars()

    engine.state.current_weight = 0.3
    engine._check_stop(bars)
    first = engine.stops.positions["BTC/USDT"]
    assert first.side == "long"

    engine.state.current_weight = -0.3      # flip without passing through flat
    engine._check_stop(bars)
    second = engine.stops.positions["BTC/USDT"]
    assert second.side == "short", "stale long stop survived a flip to short"
    assert second is not first
    assert second.bars_held == 0, "carried the old position's age"


def test_a_long_stop_sits_below_price_and_a_short_stop_above():
    """The property a flipped stop violates: it can never fire."""
    engine = _engine()
    bars = _bars()
    price = float(bars["close"].iloc[-1])

    engine.state.current_weight = 0.5
    engine._check_stop(bars)
    assert engine.stops.stop_for("BTC/USDT") < price

    engine.state.current_weight = -0.5
    engine._check_stop(bars)
    assert engine.stops.stop_for("BTC/USDT") > price


def test_risk_cap_is_derived_from_the_stop_distance():
    engine = _engine()
    engine.state.equity = 10_000.0
    cap = engine._risk_capped_weight(_bars())
    assert cap is not None and cap > 0
