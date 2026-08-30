"""Session / overnight-drift overlay.

The decisive property is negative: with 24 clock buckets, noise alone produces a
bucket that looks like a real effect. The estimator must refuse it.
"""

import numpy as np
import pandas as pd
import pytest

from godalgo.core.types import Regime
from godalgo.features.session import (
    Bucketing,
    SessionConfig,
    bucket_of,
    fit_session_profile,
)
from godalgo.portfolio.allocator import blend_signals


def hourly_index(days=400):
    return pd.date_range("2023-01-01", periods=24 * days, freq="1h", tz="UTC")


def test_pure_noise_is_shrunk_to_nothing():
    """The whole point of the shrinkage.

    Some hour will always look best across 24 buckets. If that survives, the
    overlay is a machine for trading sampling error.
    """
    idx = hourly_index()
    rng = np.random.default_rng(0)
    noise = pd.Series(rng.normal(0, 0.004, len(idx)), index=idx)

    profile = fit_session_profile(noise)
    assert np.abs(profile.shrunk_means).max() < 1e-9
    assert profile.scale == 0.0
    assert profile.tilt(idx[100]) == 0.0
    # And the raw estimate really did look like something.
    assert np.abs(profile.raw_means).max() > 1e-4


def test_a_real_effect_survives_shrinkage():
    idx = hourly_index()
    rng = np.random.default_rng(1)
    values = rng.normal(0, 0.004, len(idx))
    us_hours = np.array([14 <= t.hour <= 21 for t in idx])
    values[us_hours] += 0.0008

    profile = fit_session_profile(pd.Series(values, index=idx))
    assert profile.scale > 0
    assert profile.shrinkage.max() > 0.2

    us_tilt = np.mean([profile.tilt(t) for t in idx if 14 <= t.hour <= 21])
    other_tilt = np.mean([profile.tilt(t) for t in idx if not 14 <= t.hour <= 21])
    assert us_tilt > other_tilt
    assert us_tilt > 0


def test_shrinkage_is_conservative_never_amplifying():
    """Shrunk estimates must sit inside the raw ones, never beyond them."""
    idx = hourly_index()
    rng = np.random.default_rng(2)
    values = rng.normal(0, 0.004, len(idx))
    values[np.array([t.hour == 3 for t in idx])] += 0.001

    profile = fit_session_profile(pd.Series(values, index=idx))
    centred_raw = profile.raw_means - profile.raw_means.mean()
    assert np.abs(profile.shrunk_means).max() <= np.abs(centred_raw).max() + 1e-12


def test_thin_buckets_are_not_trusted():
    idx = pd.date_range("2024-01-01", periods=48, freq="1h", tz="UTC")
    rng = np.random.default_rng(3)
    profile = fit_session_profile(
        pd.Series(rng.normal(0, 0.01, 48), index=idx),
        SessionConfig(min_observations=30),
    )
    assert profile.scale == 0.0


@pytest.mark.parametrize(
    ("bucketing", "expected"),
    [(Bucketing.HOUR_OF_DAY, 24), (Bucketing.HOUR_OF_WEEK, 168), (Bucketing.SESSION, 3)],
)
def test_bucket_indices_stay_in_range(bucketing, expected):
    idx = hourly_index(days=30)
    buckets = {bucket_of(t, bucketing) for t in idx}
    assert max(buckets) < expected
    assert min(buckets) >= 0


def test_empty_input_is_handled():
    empty = pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"))
    profile = fit_session_profile(empty)
    assert profile.scale == 0.0


def test_non_datetime_index_is_rejected():
    """Bucketing a non-datetime index would silently produce meaningless groups."""
    with pytest.raises(TypeError, match="DatetimeIndex"):
        fit_session_profile(pd.Series([0.1, 0.2, 0.3]))


# --- integration with the allocator ---------------------------------------

def _frame(n=10, tilt=0.0):
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return (
        pd.Series(0.5, index=idx),
        pd.Series(0.0, index=idx),
        pd.Series([Regime.TRENDING] * n, index=idx),
        pd.Series(1.0, index=idx),
        pd.Series(tilt, index=idx),
    )


def test_zero_tilt_weight_leaves_the_blend_untouched():
    mom, rev, reg, conf, tilt = _frame(tilt=1.0)
    without = blend_signals(mom, rev, reg, conf)
    with_zero = blend_signals(mom, rev, reg, conf, session_tilt=tilt, tilt_weight=0.0)
    pd.testing.assert_series_equal(without["combined"], with_zero["combined"])


def test_tilt_moves_the_blend_in_its_own_direction():
    mom, rev, reg, conf, tilt = _frame(tilt=1.0)
    base = blend_signals(mom, rev, reg, conf)["combined"].iloc[-1]
    tilted = blend_signals(
        mom, rev, reg, conf, session_tilt=tilt, tilt_weight=0.5
    )["combined"].iloc[-1]
    assert tilted > base

    _, _, _, _, negative = _frame(tilt=-1.0)
    down = blend_signals(
        mom, rev, reg, conf, session_tilt=negative, tilt_weight=0.5
    )["combined"].iloc[-1]
    assert down < base


def test_tilt_cannot_raise_risk_beyond_the_authorised_allocation():
    """The calendar must not restore exposure the regime allocator withheld."""
    n = 10
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    mom = pd.Series(0.0, index=idx)
    rev = pd.Series(0.0, index=idx)
    indeterminate = pd.Series([Regime.INDETERMINATE] * n, index=idx)
    conf = pd.Series(0.0, index=idx)
    tilt = pd.Series(1.0, index=idx)

    blended = blend_signals(
        mom, rev, indeterminate, conf, session_tilt=tilt, tilt_weight=1.0
    )
    authorised = (blended["w_momentum"] + blended["w_reversion"]).iloc[-1]
    assert blended["combined"].iloc[-1] <= authorised + 1e-12


def test_misaligned_tilt_index_is_rejected():
    mom, rev, reg, conf, _ = _frame()
    bad = pd.Series(0.5, index=pd.date_range("2025-01-01", periods=10, freq="1h", tz="UTC"))
    with pytest.raises(ValueError, match="not aligned"):
        blend_signals(mom, rev, reg, conf, session_tilt=bad, tilt_weight=0.5)


# --- edge gate -------------------------------------------------------------

def test_edge_gate_blocks_entries_that_cannot_pay_costs():
    from godalgo.portfolio.sizing import apply_edge_gate

    idx = pd.date_range("2024-01-01", periods=5, freq="1h", tz="UTC")
    target = pd.Series([0.0, 0.5, 0.5, 0.5, 0.5], index=idx)
    weak = pd.Series([0.0, 1.0, 1.0, 1.0, 1.0], index=idx)
    gated = apply_edge_gate(target, weak, round_trip_cost_bps=22.0, min_edge_multiple=1.5)
    assert (gated == 0.0).all()


def test_edge_gate_allows_entries_that_clear_costs():
    from godalgo.portfolio.sizing import apply_edge_gate

    idx = pd.date_range("2024-01-01", periods=3, freq="1h", tz="UTC")
    target = pd.Series([0.0, 0.5, 0.5], index=idx)
    strong = pd.Series([0.0, 100.0, 100.0], index=idx)
    gated = apply_edge_gate(target, strong, 22.0, 1.5)
    assert gated.iloc[-1] == pytest.approx(0.5)


def test_edge_gate_never_blocks_an_exit():
    from godalgo.portfolio.sizing import apply_edge_gate

    idx = pd.date_range("2024-01-01", periods=3, freq="1h", tz="UTC")
    target = pd.Series([0.5, 0.5, 0.0], index=idx)
    edge = pd.Series([100.0, 0.0, 0.0], index=idx)
    gated = apply_edge_gate(target, edge, 22.0, 1.5)
    assert gated.iloc[-1] == 0.0


def test_edge_gate_disable_is_explicit():
    from godalgo.portfolio.sizing import apply_edge_gate

    idx = pd.date_range("2024-01-01", periods=3, freq="1h", tz="UTC")
    target = pd.Series([0.0, 0.5, 0.5], index=idx)
    weak = pd.Series([0.0, 0.1, 0.1], index=idx)
    pd.testing.assert_series_equal(apply_edge_gate(target, weak, 22.0, 0.0), target)
    with pytest.raises(ValueError, match="0.0 \\(disabled\\) or >= 1.0"):
        apply_edge_gate(target, weak, 22.0, 0.5)
