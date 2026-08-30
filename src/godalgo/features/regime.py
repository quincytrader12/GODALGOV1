"""Regime classification: is this series trending or mean-reverting?

This module is the hinge of the whole system. Momentum and mean reversion are
opposite bets on autocorrelation -- momentum needs positive serial correlation,
reversion needs negative. Running both blindly means one is always paying the
other. So instead of averaging two strategies, we estimate which regime holds
and let that estimate drive the allocation.

Three independent tests, deliberately not variations on one idea:

* **Hurst exponent** (Hurst 1951; Mandelbrot & Van Ness 1968). H > 0.5 means a
  series diffuses faster than a random walk -- trending. H < 0.5 means it
  diffuses slower -- reverting. H = 0.5 is a random walk, no edge either way.
* **Variance ratio test** (Lo & MacKinlay 1988). Same intuition, but with an
  actual sampling distribution, so we get a significance level rather than a
  point estimate. Uses the heteroskedasticity-robust statistic, which matters
  for crypto where volatility clustering is severe.
* **Augmented Dickey-Fuller + OU half-life** (Ornstein-Uhlenbeck). ADF asks
  whether a unit root can be rejected; half-life says how fast reversion
  happens, which is what makes a reversion trade tradeable or not.

The tests are not weighted equally, and the asymmetry is deliberate. The
variance ratio is the *primary* discriminator because it is the only one of the
three with a sampling distribution -- it can say "significant at 5%" rather than
merely reporting a number. The Hurst exponent is a point estimate with a known
small-sample bias and no error bars, so it serves as a **veto**: it can block a
regime call by pointing the other way, but it cannot be required to clear an
arbitrary band before a properly significant test result counts.

Getting this backwards -- gating on the weaker statistic -- makes the classifier
refuse to call regimes it has strong evidence for, and the allocator then holds
minimal risk almost all the time.

A regime is only declared when no test contradicts the others. Disagreement
returns INDETERMINATE, which the allocator treats as a reason to hold less risk
-- not as a coin flip.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from godalgo.core.types import Regime, RegimeState

__all__ = [
    "adf_pvalue",
    "classify_regime",
    "hurst_exponent",
    "ou_half_life",
    "variance_ratio",
]


def hurst_exponent(
    prices: pd.Series | np.ndarray,
    min_lag: int = 2,
    max_lag: int = 64,
) -> float:
    """Estimate the Hurst exponent by generalised variance scaling.

    For a self-similar process, ``Std[x(t + tau) - x(t)] ~ tau**H``. Regressing
    log dispersion on log lag recovers H as the slope.

    We use this rather than classical rescaled-range (R/S) analysis because R/S
    carries a well-documented small-sample bias that pushes H above 0.5 even on
    genuine random walks -- which would make everything look like a momentum
    opportunity. The variance-scaling estimator is far better behaved on the
    few-hundred-bar windows we actually run on.

    Args:
        prices: Price level series. Log-transformed internally; must be positive.
        min_lag: Shortest lag in bars. Below 2 the differences are too noisy.
        max_lag: Longest lag. Capped at len(prices) // 4 so the longest lag still
            has enough non-overlapping samples to estimate a dispersion.

    Returns:
        Estimated H. ~0.5 random walk, >0.5 trending, <0.5 mean reverting.
        Returns 0.5 (the no-information value) if the window is too short.
    """
    x = np.asarray(prices, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 4 * min_lag or np.any(x <= 0):
        return 0.5

    log_p = np.log(x)
    max_lag = min(max_lag, log_p.size // 4)
    if max_lag <= min_lag:
        return 0.5

    lags = np.arange(min_lag, max_lag + 1)
    dispersions = np.empty(lags.size, dtype=float)
    for i, lag in enumerate(lags):
        diffs = log_p[lag:] - log_p[:-lag]
        dispersions[i] = np.std(diffs, ddof=1) if diffs.size > 1 else np.nan

    ok = np.isfinite(dispersions) & (dispersions > 0)
    if ok.sum() < 3:
        return 0.5

    slope, _ = np.polyfit(np.log(lags[ok]), np.log(dispersions[ok]), 1)
    # Clip to the theoretically meaningful range; estimates outside it are noise.
    return float(np.clip(slope, 0.0, 1.0))


def variance_ratio(
    prices: pd.Series | np.ndarray,
    q: int = 8,
) -> tuple[float, float]:
    """Lo-MacKinlay variance ratio test with heteroskedasticity-robust z-stat.

    Under a random walk, the variance of q-period returns is exactly q times the
    variance of 1-period returns, so VR(q) = 1. VR > 1 implies positive serial
    correlation (trend persistence); VR < 1 implies negative (reversion).

    The robust statistic is used because the homoskedastic version rejects the
    random-walk null far too often when volatility clusters -- which it always
    does in crypto. Using the naive version here would manufacture regimes that
    are really just vol regimes.

    Args:
        prices: Price level series, positive.
        q: Aggregation horizon in bars.

    Returns:
        ``(vr, z)``. ``z`` is asymptotically standard normal under the null, so
        ``|z| > 1.96`` is rejection at 5%. Returns ``(1.0, 0.0)`` -- the
        no-information result -- when the window is too short to test.
    """
    x = np.asarray(prices, dtype=float)
    x = x[np.isfinite(x)]
    if q < 2 or x.size < 2 * q + 2 or np.any(x <= 0):
        return 1.0, 0.0

    p = np.log(x)
    n = p.size - 1              # number of single-period returns
    r = np.diff(p)
    mu = (p[-1] - p[0]) / n

    var_1 = np.sum((r - mu) ** 2) / (n - 1)
    if var_1 <= 0:
        return 1.0, 0.0

    # Overlapping q-period returns, with the Lo-MacKinlay unbiased denominator.
    m = q * (n - q + 1) * (1.0 - q / n)
    if m <= 0:
        return 1.0, 0.0
    q_returns = p[q:] - p[:-q]
    var_q = np.sum((q_returns - q * mu) ** 2) / m

    vr = float(var_q / var_1)

    # Heteroskedasticity-robust variance of VR (Lo & MacKinlay 1988, eq. 18).
    eps2 = (r - mu) ** 2
    denom = np.sum(eps2) ** 2
    if denom <= 0:
        return vr, 0.0

    theta = 0.0
    for j in range(1, q):
        delta_j = np.sum(eps2[j:] * eps2[:-j]) / denom
        theta += (2.0 * (q - j) / q) ** 2 * delta_j

    if theta <= 0:
        return vr, 0.0
    z = (vr - 1.0) / np.sqrt(theta)
    return vr, float(z)


def ou_half_life(prices: pd.Series | np.ndarray) -> float | None:
    """Half-life of mean reversion under an Ornstein-Uhlenbeck fit.

    Discretised OU is ``dy_t = a + b * y_{t-1} + e_t``. A negative ``b`` means
    the series is pulled back toward its mean, and the time to close half the
    gap is ``-ln(2) / b``.

    Half-life is the number that decides whether a reversion signal is
    *tradeable*, as distinct from merely *present*. A statistically perfect
    reversion with a 400-bar half-life will be eaten by funding and fees long
    before it pays.

    Returns:
        Half-life in bars, or ``None`` if the series is not mean reverting
        (``b >= 0``) or the window is too short.
    """
    y = np.asarray(prices, dtype=float)
    y = y[np.isfinite(y)]
    if y.size < 20 or np.any(y <= 0):
        return None

    log_y = np.log(y)
    lagged = log_y[:-1]
    delta = np.diff(log_y)

    design = np.column_stack([np.ones_like(lagged), lagged])
    try:
        coef, *_ = np.linalg.lstsq(design, delta, rcond=None)
    except np.linalg.LinAlgError:
        return None

    b = coef[1]
    if b >= 0 or not np.isfinite(b):
        return None
    return float(-np.log(2.0) / b)


def adf_pvalue(prices: pd.Series | np.ndarray) -> float | None:
    """p-value of the Augmented Dickey-Fuller test on log prices.

    Null hypothesis is a unit root (non-stationary). A small p-value rejects the
    random walk in favour of stationarity, i.e. evidence for mean reversion.

    Returns ``None`` when the test cannot be computed on the given window.
    """
    y = np.asarray(prices, dtype=float)
    y = y[np.isfinite(y)]
    if y.size < 30 or np.any(y <= 0):
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = adfuller(np.log(y), autolag="AIC")
        return float(result[1])
    except (ValueError, np.linalg.LinAlgError):
        return None


def classify_regime(
    prices: pd.Series,
    symbol: str,
    *,
    vr_q: int = 8,
    hurst_band: float = 0.05,
    z_threshold: float = 1.645,
    max_half_life: int = 200,
) -> RegimeState:
    """Combine the three tests into a single regime call with a confidence.

    Decision rule -- a significant variance ratio proposes the regime, and the
    Hurst estimate may veto it:

    * TRENDING: ``VR > 1`` with a significant z-stat, and ``H`` not contradicting
      (``H > 0.5 - band``).
    * MEAN_REVERTING: ``VR < 1`` with a significant z-stat, ``H`` not
      contradicting (``H < 0.5 + band``), and a half-life short enough to trade.
    * INDETERMINATE: anything else.

    ``hurst_band`` is a *contradiction tolerance*, not an entry threshold. The
    Hurst estimator has real sampling error, so a reading marginally on the
    wrong side of 0.5 is not evidence against a z-stat of 10 -- but a reading
    well onto the wrong side is, and that is what the band sizes.

    Args:
        prices: Price series for one symbol, oldest first.
        symbol: Symbol label, carried through to the result.
        vr_q: Aggregation horizon for the variance ratio test.
        hurst_band: How far onto the *opposing* side of 0.5 the Hurst estimate
            may sit before it vetoes an otherwise significant regime call.
        z_threshold: |z| needed to call the variance ratio significant. This is
            the actual decision threshold of the classifier.
            Default 1.645 is the 10% two-sided / 5% one-sided level.
        max_half_life: Reversion slower than this (in bars) is not tradeable.

    Returns:
        A ``RegimeState``. ``confidence`` in [0, 1] scales allocator risk and is
        driven by how far the evidence sits past the decision thresholds.
    """
    if prices.empty:
        raise ValueError("cannot classify regime on an empty price series")

    timestamp = prices.index[-1]
    h = hurst_exponent(prices)
    vr, z = variance_ratio(prices, q=vr_q)
    hl = ou_half_life(prices)
    adf_p = adf_pvalue(prices)

    significant = abs(z) >= z_threshold
    trending = vr > 1.0 and significant and h > 0.5 - hurst_band
    reverting = (
        vr < 1.0
        and significant
        and h < 0.5 + hurst_band
        and hl is not None
        and hl <= max_half_life
    )

    if trending:
        regime = Regime.TRENDING
    elif reverting:
        regime = Regime.MEAN_REVERTING
    else:
        regime = Regime.INDETERMINATE

    confidence = _confidence(regime, h, z, hurst_band, z_threshold, adf_p)

    return RegimeState(
        timestamp=timestamp,
        symbol=symbol,
        regime=regime,
        hurst=h,
        variance_ratio=vr,
        vr_zscore=z,
        half_life=hl,
        adf_pvalue=adf_p,
        confidence=confidence,
    )


def _confidence(
    regime: Regime,
    hurst: float,
    z: float,
    hurst_band: float,
    z_threshold: float,
    adf_p: float | None,
) -> float:
    """Blend evidence strength into a [0, 1] risk scaler.

    Weighted to match the decision rule: the variance ratio's z-statistic is the
    evidence that actually established the regime, so it carries most of the
    weight. Hurst corroborates.

    * ``z`` excess past the significance threshold, saturating at 3x  (weight 0.7)
    * distance of H from 0.5 on the supporting side, saturating at 0.2 (weight 0.3)

    For mean reversion a third term rewards a confirming ADF rejection. ADF is
    only informative in that direction, so it is not applied to trends.
    """
    if regime is Regime.INDETERMINATE:
        return 0.0

    z_excess = max(0.0, abs(z) - z_threshold)
    z_score = min(1.0, z_excess / (2.0 * z_threshold))

    # Only Hurst movement in the direction that supports the call counts.
    supporting = (hurst - 0.5) if regime is Regime.TRENDING else (0.5 - hurst)
    hurst_score = min(1.0, max(0.0, supporting) / 0.20)

    score = 0.7 * z_score + 0.3 * hurst_score

    if regime is Regime.MEAN_REVERTING and adf_p is not None:
        # Reward stationarity evidence, but never let a single test dominate.
        adf_score = min(1.0, max(0.0, (0.10 - adf_p) / 0.10))
        score = 0.75 * score + 0.25 * adf_score

    return float(np.clip(score, 0.0, 1.0))
