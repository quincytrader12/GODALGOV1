"""Position sizing.

Sizing is kept strictly separate from signal generation. A strategy says "I want
to be long with conviction 0.6"; this layer decides what 0.6 is worth in equity
terms given current volatility and the risk budget. Fusing the two is how
strategies end up implicitly carrying their own, mutually inconsistent, risk
models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["apply_turnover_buffer", "fractional_kelly", "volatility_target_scalar"]


def volatility_target_scalar(
    returns: pd.Series,
    target_annual_vol: float,
    bars_per_day: float,
    *,
    halflife: float = 30.0,
    max_leverage: float = 3.0,
    trading_days: float = 365.0,
) -> pd.Series:
    """Scale positions so realised portfolio vol tracks a constant target.

    Volatility is far more forecastable than return -- it is strongly
    autocorrelated, while returns are close to unforecastable at these horizons.
    Targeting it turns an unstable risk profile into a roughly stationary one,
    which is what makes drawdowns comparable across regimes and makes the
    performance statistics in the promotion gate mean the same thing in every
    period they are measured over.

    Args:
        returns: Per-bar log returns of the traded asset.
        target_annual_vol: Desired annualised volatility, e.g. 0.20 for 20%.
        bars_per_day: Bars per calendar day for the timeframe in use.
        halflife: EWMA half-life for the volatility estimate, in bars.
        max_leverage: Hard cap on the scalar. Without it, a quiet period drives
            the estimate toward zero and the scalar toward infinity -- precisely
            before the vol expansion that ends the quiet period.
        trading_days: Days per year. 365 for crypto's continuous market.

    Returns:
        A non-negative scalar per bar, shifted one bar forward so that sizing at
        ``t`` uses only volatility observable through ``t-1``.
    """
    if target_annual_vol <= 0:
        raise ValueError("target_annual_vol must be positive")

    bar_vol = returns.ewm(halflife=halflife, min_periods=2).std(bias=False)
    annual_vol = bar_vol * np.sqrt(bars_per_day * trading_days)

    scalar = (target_annual_vol / annual_vol.replace(0.0, np.nan)).clip(
        lower=0.0, upper=max_leverage
    )
    # Shift: the vol estimate for bar t is only complete at the close of bar t,
    # so it can only size the position entered at t+1.
    return scalar.shift(1).fillna(0.0)


def fractional_kelly(
    edge: float,
    variance: float,
    fraction: float = 0.25,
    max_weight: float = 1.0,
) -> float:
    """Kelly-optimal weight, deliberately under-bet.

    Full Kelly (``edge / variance``) maximises long-run log wealth *given known
    parameters*. We do not know the parameters -- we estimate the edge from a
    finite backtest, and Kelly is brutally sensitive to that estimate. Over-
    estimating the edge by 2x under full Kelly produces a strategy with negative
    expected log growth.

    Quarter Kelly is the usual practitioner compromise: roughly 44% of the growth
    rate of full Kelly at roughly a quarter of the variance drag, and it stays
    positive-growth even when the edge estimate is off by a factor of two.

    Args:
        edge: Expected per-bar excess return.
        variance: Per-bar return variance. Must be positive.
        fraction: Kelly multiplier. Keep at or below 0.5.
        max_weight: Cap on the returned weight.

    Returns:
        Weight as a fraction of equity, clipped to ``[-max_weight, max_weight]``.
    """
    if variance <= 0 or not np.isfinite(variance):
        return 0.0
    if not np.isfinite(edge):
        return 0.0
    weight = fraction * edge / variance
    return float(np.clip(weight, -max_weight, max_weight))


def apply_turnover_buffer(
    target: pd.Series,
    buffer: float = 0.10,
) -> pd.Series:
    """Suppress position changes smaller than ``buffer``.

    A no-trade band. Every rebalance pays the spread, so a continuously varying
    target weight leaks money through changes too small to matter. Holding the
    previous weight until the target has moved materially cuts turnover sharply
    at negligible tracking error.

    Args:
        target: Desired weight per bar.
        buffer: Minimum absolute change in weight required to trade.

    Returns:
        The buffered weight series.
    """
    if buffer <= 0:
        return target

    values = target.to_numpy(dtype=float)
    held = np.zeros_like(values)
    current = 0.0
    for i, desired in enumerate(values):
        if not np.isfinite(desired):
            desired = 0.0
        # Always honour a full exit: refusing to flatten because the step is
        # small is how a small position becomes a permanent one.
        if desired == 0.0 or abs(desired - current) >= buffer:
            current = float(desired)
        held[i] = current

    return pd.Series(held, index=target.index, name=target.name)
