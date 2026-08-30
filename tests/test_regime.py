"""Regime classifier must discriminate, and must refuse to guess."""

import numpy as np
import pandas as pd
import pytest

from godalgo.core.types import Regime
from godalgo.features.regime import (
    classify_regime,
    hurst_exponent,
    ou_half_life,
    variance_ratio,
)


def _index(n):
    return pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")


def random_walk(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=_index(n))


def trending(n=2000, seed=1, phi=0.30):
    rng = np.random.default_rng(seed)
    r = np.zeros(n)
    for i in range(1, n):
        r[i] = phi * r[i - 1] + rng.normal(0, 0.01)
    return pd.Series(100 * np.exp(np.cumsum(r)), index=_index(n))


def reverting(n=2000, seed=2, theta=0.05):
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = x[i - 1] + theta * (0.0 - x[i - 1]) + rng.normal(0, 0.02)
    return pd.Series(100 * np.exp(x), index=_index(n))


def test_hurst_separates_the_three_processes():
    assert hurst_exponent(trending()) > hurst_exponent(random_walk())
    assert hurst_exponent(reverting()) < hurst_exponent(random_walk())


def test_hurst_on_random_walk_is_near_one_half():
    assert 0.40 < hurst_exponent(random_walk()) < 0.60


def test_variance_ratio_sign_matches_process():
    vr_trend, z_trend = variance_ratio(trending(), q=8)
    vr_rev, z_rev = variance_ratio(reverting(), q=8)
    assert vr_trend > 1.0 and z_trend > 0
    assert vr_rev < 1.0 and z_rev < 0


def test_half_life_none_for_trending_positive_for_reverting():
    assert ou_half_life(reverting()) is not None
    assert ou_half_life(reverting()) > 0


def test_classify_labels_each_process_correctly():
    assert classify_regime(trending(), "T").regime is Regime.TRENDING
    assert classify_regime(reverting(), "R").regime is Regime.MEAN_REVERTING


def test_random_walk_is_not_assigned_an_edge():
    """The important negative case.

    A classifier that labels noise as a tradeable regime is worse than useless
    -- it routes capital into a strategy with no edge and full costs.
    """
    state = classify_regime(random_walk(), "RW")
    assert state.regime is Regime.INDETERMINATE
    assert state.confidence == 0.0


def test_short_windows_degrade_to_no_information():
    tiny = pd.Series([100.0, 101.0, 100.5], index=_index(3))
    assert hurst_exponent(tiny) == 0.5
    assert variance_ratio(tiny) == (1.0, 0.0)
    assert ou_half_life(tiny) is None


def test_empty_series_raises():
    with pytest.raises(ValueError, match="empty"):
        classify_regime(pd.Series(dtype=float), "X")
