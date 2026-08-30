"""No-lookahead invariants.

The single most valuable property in this codebase. A lookahead bug does not
crash -- it produces a wonderful backtest that cannot be traded, and it
contaminates every statistic the promotion gate relies on.
"""

import numpy as np
import pandas as pd

from godalgo.strategies.mean_reversion import MeanReversionStrategy
from godalgo.strategies.momentum import MomentumStrategy


def bars(n=1500, seed=3):
    rng = np.random.default_rng(seed)
    price = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {"open": price, "high": price * 1.003, "low": price * 0.997,
         "close": price, "volume": 1.0},
        index=idx,
    )


def _assert_prefix_stable(strategy, data, cut):
    """A signal computed on a prefix must equal the same bars of the full run.

    If future bars influence past signals, truncating the data changes the past
    -- which is exactly what this compares.
    """
    full = strategy.generate(data)
    prefix = strategy.generate(data.iloc[:cut])
    pd.testing.assert_series_equal(
        full.iloc[:cut], prefix, check_names=False, rtol=1e-9, atol=1e-12
    )


def test_momentum_signal_does_not_use_future_bars():
    _assert_prefix_stable(MomentumStrategy(), bars(), cut=1000)


def test_mean_reversion_signal_does_not_use_future_bars():
    _assert_prefix_stable(MeanReversionStrategy(), bars(), cut=1000)


def test_signals_are_bounded_and_never_nan():
    data = bars()
    for strategy in (MomentumStrategy(), MeanReversionStrategy()):
        signal = strategy.generate(data)
        assert signal.notna().all(), f"{strategy.name} emitted NaN"
        assert signal.abs().max() <= 1.0 + 1e-12, f"{strategy.name} exceeded [-1, 1]"


def test_warmup_bars_are_flat():
    data = bars()
    for strategy in (MomentumStrategy(), MeanReversionStrategy()):
        signal = strategy.generate(data)
        assert (signal.iloc[: strategy.warmup] == 0.0).all()
