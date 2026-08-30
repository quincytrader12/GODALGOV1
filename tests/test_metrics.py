"""The anti-overfitting statistics must actually penalise overfitting."""

import numpy as np
import pandas as pd
import pytest

from godalgo.backtest.metrics import (
    compute_stats,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    max_drawdown,
    probability_of_backtest_overfitting,
    sharpe_ratio,
)


def test_expected_max_sharpe_grows_with_trial_count():
    """More trials, higher bar. This is the correction's whole purpose."""
    few = expected_max_sharpe(10, 0.01)
    many = expected_max_sharpe(1000, 0.01)
    assert many > few > 0


def test_deflated_sharpe_falls_as_trials_rise():
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.0008, 0.01, 2000))
    honest = deflated_sharpe_ratio(returns, n_trials=1, sharpe_variance=0.0)
    searched = deflated_sharpe_ratio(returns, n_trials=5000, sharpe_variance=0.02)
    assert honest > searched


def test_deflated_sharpe_rejects_pure_noise():
    rng = np.random.default_rng(1)
    noise = pd.Series(rng.normal(0.0, 0.01, 1500))
    assert deflated_sharpe_ratio(noise, n_trials=100, sharpe_variance=0.01) < 0.95


def test_pbo_near_half_for_pure_noise():
    """Selecting among noise generalises no better than chance."""
    rng = np.random.default_rng(2)
    trials = pd.DataFrame(rng.normal(0, 0.01, (2000, 12)))
    pbo = probability_of_backtest_overfitting(trials, n_splits=8)
    assert 0.25 < pbo < 0.75


def test_pbo_low_when_one_candidate_genuinely_dominates():
    """A real, persistent edge must survive the CSCV splits."""
    rng = np.random.default_rng(3)
    trials = pd.DataFrame(rng.normal(0, 0.01, (2000, 8)))
    trials[0] = rng.normal(0.004, 0.01, 2000)   # consistently superior column
    pbo = probability_of_backtest_overfitting(trials, n_splits=8)
    assert pbo < 0.20


def test_max_drawdown_matches_a_hand_computed_case():
    equity = pd.Series([1.0, 1.25, 0.75, 1.0])
    assert max_drawdown(equity) == pytest.approx(0.40)


def test_sharpe_of_constant_series_is_zero():
    assert sharpe_ratio(pd.Series([0.01] * 50), 365) == 0.0


def test_hit_rate_ignores_flat_bars():
    """Regression: flat bars were being scored as losses."""
    returns = pd.Series([0.0, 0.0, 0.01, -0.005, 0.0, 0.02])
    weights = pd.Series([0.0, 0.0, 1.0, 1.0, 0.0, 1.0])
    stats = compute_stats(returns, 365, weights)
    assert stats.hit_rate == pytest.approx(2 / 3)
    assert stats.time_in_market == pytest.approx(0.5)
