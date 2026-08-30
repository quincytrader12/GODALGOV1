"""Indicator primitives shared by both strategies.

Every function here is strictly causal: the value at bar ``t`` uses only bars
``<= t``. That property is what makes the backtest honest, so it is worth
stating as a hard rule -- a single lookahead here silently inflates every
downstream metric, including the ones the self-improvement gate relies on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "annualisation_factor",
    "atr",
    "ewma_volatility",
    "log_returns",
    "realised_volatility",
    "rolling_zscore",
]


def annualisation_factor(bars_per_day: float, trading_days: float = 365.0) -> float:
    """Square-root-of-time scaler from per-bar to annual units.

    Crypto trades continuously, so the default is 365 calendar days rather than
    the 252 equity trading days. Using 252 on a 24/7 market understates
    annualised volatility by ~20% and correspondingly overstates Sharpe.
    """
    return float(np.sqrt(bars_per_day * trading_days))


def log_returns(prices: pd.Series) -> pd.Series:
    """Log returns. Additive across time, which keeps compounding exact."""
    return np.log(prices).diff()


def realised_volatility(returns: pd.Series, window: int) -> pd.Series:
    """Rolling sample standard deviation of returns, in per-bar units."""
    window = int(window)
    return returns.rolling(window, min_periods=max(2, window // 2)).std(ddof=1)


def ewma_volatility(returns: pd.Series, halflife: float) -> pd.Series:
    """Exponentially weighted volatility.

    Preferred over a flat rolling window for position sizing: vol regimes shift
    faster than a rectangular window can follow, and a stale vol estimate sizes
    positions using conditions that no longer hold.
    """
    return returns.ewm(halflife=halflife, min_periods=2).std(bias=False)


def rolling_zscore(series: pd.Series, window: int, min_std: float = 1e-12) -> pd.Series:
    """Standardise against a trailing window.

    ``min_std`` guards the flat-series case, where a near-zero denominator would
    otherwise produce enormous z-scores out of rounding noise -- and those turn
    straight into maximum-size positions.
    """
    window = int(window)
    min_periods = max(2, window // 2)
    mean = series.rolling(window, min_periods=min_periods).mean()
    std = series.rolling(window, min_periods=min_periods).std(ddof=1)
    return (series - mean) / std.clip(lower=min_std)


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
) -> pd.Series:
    """Average true range (Wilder 1978), used for stop placement.

    True range accounts for overnight/inter-bar gaps, which a simple high-low
    range misses -- and gaps are exactly when stops matter most.
    """
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
