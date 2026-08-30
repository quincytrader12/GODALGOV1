"""Performance statistics, including the ones that survive multiple testing.

Ordinary Sharpe is not a usable promotion criterion for a system that searches
its own parameter space. If you evaluate N configurations and keep the best, the
maximum Sharpe you observe is an order statistic, and its expectation is well
above zero even when every configuration is worthless. Selecting on raw Sharpe
therefore promotes noise, reliably, and does so more confidently the harder the
system searches.

Two corrections from Bailey & Lopez de Prado address this directly:

* **Deflated Sharpe Ratio** (Bailey & Lopez de Prado 2014) asks whether an
  observed Sharpe exceeds what the *best of N trials* would produce by luck
  alone, while also correcting for the non-normality of returns. It returns a
  probability, so it can be thresholded.

* **Probability of Backtest Overfitting** (Bailey, Borwein, Lopez de Prado & Zhu
  2017), estimated by combinatorially symmetric cross-validation. It asks a
  different and complementary question: across many in-sample/out-of-sample
  splits, how often does the configuration that looked best in-sample land below
  median out-of-sample? A high PBO means the *selection procedure* is broken,
  independent of any single result.

The promotion gate in ``godalgo.evolve`` requires both.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "PerformanceStats",
    "compute_stats",
    "deflated_sharpe_ratio",
    "expected_max_sharpe",
    "max_drawdown",
    "probability_of_backtest_overfitting",
    "sharpe_ratio",
]

_EULER_MASCHERONI = 0.5772156649015329


@dataclass(frozen=True, slots=True)
class PerformanceStats:
    """Summary of one equity curve."""

    n_observations: int
    total_return: float
    annual_return: float
    annual_volatility: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    hit_rate: float
    """Share of *active* bars that were profitable.

    Measured over bars holding a position, not all bars. A flat bar returns
    exactly 0.0, and counting those as losses makes a selective strategy look
    catastrophically wrong when it is merely idle.
    """

    time_in_market: float
    """Share of bars holding a non-zero position."""

    turnover: float
    skew: float
    kurtosis: float
    """Non-excess (normal = 3.0), matching the deflated Sharpe convention."""

    def to_dict(self) -> dict[str, float | int]:
        return {
            "n_observations": self.n_observations,
            "total_return": self.total_return,
            "annual_return": self.annual_return,
            "annual_volatility": self.annual_volatility,
            "sharpe": self.sharpe,
            "sortino": self.sortino,
            "max_drawdown": self.max_drawdown,
            "calmar": self.calmar,
            "hit_rate": self.hit_rate,
            "time_in_market": self.time_in_market,
            "turnover": self.turnover,
            "skew": self.skew,
            "kurtosis": self.kurtosis,
        }


def sharpe_ratio(returns: pd.Series | np.ndarray, periods_per_year: float) -> float:
    """Annualised Sharpe ratio with a zero risk-free rate.

    Zero risk-free is the right convention for a crypto perp book, where the
    financing cost is funding and is already charged in the return series rather
    than being an opportunity cost sitting outside it.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2:
        return 0.0
    sd = r.std(ddof=1)
    if sd <= 0:
        return 0.0
    return float(r.mean() / sd * np.sqrt(periods_per_year))


def max_drawdown(equity: pd.Series | np.ndarray) -> float:
    """Largest peak-to-trough decline, as a positive fraction."""
    e = np.asarray(equity, dtype=float)
    e = e[np.isfinite(e)]
    if e.size == 0:
        return 0.0
    running_peak = np.maximum.accumulate(e)
    drawdowns = 1.0 - e / running_peak
    return float(np.max(drawdowns))


def expected_max_sharpe(n_trials: int, sharpe_variance: float) -> float:
    """Expected maximum of N independent Sharpe estimates under a zero-skill null.

    Uses the extreme-value approximation from Bailey & Lopez de Prado (2014):

        E[max SR_N] ~ sigma * [(1 - g) * Z^-1(1 - 1/N) + g * Z^-1(1 - 1/(N*e))]

    with ``g`` the Euler-Mascheroni constant. This is the benchmark an observed
    Sharpe has to beat before it counts as evidence of anything.

    Args:
        n_trials: Number of configurations evaluated during selection. Counting
            this honestly is essential -- undercounting trials is the single
            easiest way to make a worthless strategy pass the gate.
        sharpe_variance: Variance of the Sharpe estimates across those trials,
            in the same (per-observation) units as the observed Sharpe.

    Returns:
        The expected maximum Sharpe, in per-observation units.
    """
    if n_trials < 2 or sharpe_variance <= 0 or not np.isfinite(sharpe_variance):
        return 0.0
    sigma = np.sqrt(sharpe_variance)
    n = float(n_trials)
    term_1 = (1.0 - _EULER_MASCHERONI) * stats.norm.ppf(1.0 - 1.0 / n)
    term_2 = _EULER_MASCHERONI * stats.norm.ppf(1.0 - 1.0 / (n * np.e))
    return float(sigma * (term_1 + term_2))


def deflated_sharpe_ratio(
    returns: pd.Series | np.ndarray,
    n_trials: int,
    sharpe_variance: float,
) -> float:
    """Probability that the observed Sharpe reflects genuine skill.

    Corrects for two distinct biases at once:

    * **Selection bias** -- the benchmark is the expected maximum over
      ``n_trials``, not zero.
    * **Non-normality** -- negative skew and fat tails, both endemic to crypto,
      inflate the apparent significance of a Sharpe estimate. The denominator
      penalises exactly those.

    Args:
        returns: Per-bar strategy returns.
        n_trials: Number of configurations evaluated in selection.
        sharpe_variance: Variance of Sharpe estimates across those trials, in
            per-observation units.

    Returns:
        A probability in [0, 1]. Values above ~0.95 are the usual bar for
        treating a result as more than sampling noise.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    t = r.size
    if t < 3:
        return 0.0

    sd = r.std(ddof=1)
    if sd <= 0:
        return 0.0

    # Per-observation Sharpe: the DSR formula is stated in these units, and
    # mixing in an annualised value here silently corrupts the result.
    sr = r.mean() / sd
    sr_star = expected_max_sharpe(n_trials, sharpe_variance)

    skew = float(stats.skew(r, bias=False))
    kurt = float(stats.kurtosis(r, fisher=False, bias=False))  # non-excess

    denominator_sq = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr**2
    if denominator_sq <= 0 or not np.isfinite(denominator_sq):
        return 0.0

    z = (sr - sr_star) * np.sqrt(t - 1) / np.sqrt(denominator_sq)
    if not np.isfinite(z):
        return 0.0
    return float(stats.norm.cdf(z))


def probability_of_backtest_overfitting(
    trial_returns: pd.DataFrame,
    n_splits: int = 10,
) -> float:
    """PBO via combinatorially symmetric cross-validation (CSCV).

    Procedure:

    1. Split the observation axis into ``n_splits`` contiguous, equal blocks.
    2. For every way of choosing half the blocks as in-sample (the rest being
       out-of-sample), find the configuration with the best in-sample Sharpe.
    3. Record where that configuration ranks out-of-sample.
    4. PBO is the fraction of splits where the in-sample winner landed below the
       out-of-sample median.

    The symmetry is what makes this work: because in-sample and out-of-sample
    are complementary halves drawn the same way, there is no train/test
    asymmetry to explain away a bad result.

    Interpretation: PBO near 0.5 means selection is no better than picking at
    random -- the search is fitting noise. Below ~0.2 the selection procedure is
    carrying real information.

    Args:
        trial_returns: Frame of per-bar returns, one column per configuration.
            Needs at least two columns; PBO is a statement about *choosing
            between* configurations.
        n_splits: Number of blocks. Must be even. Larger is more thorough and
            costs C(n, n/2) evaluations -- 10 gives 252, 16 gives 12870.

    Returns:
        Estimated probability in [0, 1]. Returns 0.5 -- the uninformative value
        -- when the input is too small to evaluate.
    """
    if n_splits % 2 != 0:
        raise ValueError("n_splits must be even so that halves are the same size")

    matrix = trial_returns.to_numpy(dtype=float)
    n_obs, n_configs = matrix.shape
    if n_configs < 2 or n_obs < n_splits * 2:
        return 0.5

    block_size = n_obs // n_splits
    blocks = [
        matrix[i * block_size : (i + 1) * block_size] for i in range(n_splits)
    ]

    logits: list[float] = []
    half = n_splits // 2

    for is_indices in combinations(range(n_splits), half):
        oos_indices = [i for i in range(n_splits) if i not in is_indices]

        is_data = np.vstack([blocks[i] for i in is_indices])
        oos_data = np.vstack([blocks[i] for i in oos_indices])

        is_sharpe = _column_sharpe(is_data)
        oos_sharpe = _column_sharpe(oos_data)
        if not np.any(np.isfinite(is_sharpe)) or not np.any(np.isfinite(oos_sharpe)):
            continue

        best = int(np.nanargmax(is_sharpe))

        # Relative rank of the in-sample winner within the OOS distribution.
        finite = np.isfinite(oos_sharpe)
        ranks = stats.rankdata(np.where(finite, oos_sharpe, -np.inf))
        omega = ranks[best] / (n_configs + 1)

        # Bounded away from 0 and 1 so the logit stays finite.
        omega = float(np.clip(omega, 1e-6, 1 - 1e-6))
        logits.append(float(np.log(omega / (1.0 - omega))))

    if not logits:
        return 0.5

    # PBO = P(logit <= 0) = P(winner ranks at or below the OOS median).
    return float(np.mean(np.asarray(logits) <= 0.0))


def _column_sharpe(data: np.ndarray) -> np.ndarray:
    """Per-observation Sharpe for each column; NaN where undefined.

    Not annualised -- PBO only ever compares and ranks these, and the
    annualisation factor is a positive constant that cancels out of any ranking.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.nanmean(data, axis=0)
        sd = np.nanstd(data, axis=0, ddof=1)
        out = np.where(sd > 0, mean / sd, np.nan)
    return out


def compute_stats(
    returns: pd.Series,
    periods_per_year: float,
    weights: pd.Series | None = None,
) -> PerformanceStats:
    """Summarise an equity curve.

    Args:
        returns: Per-bar arithmetic returns, net of costs.
        periods_per_year: Bars per year, for annualisation.
        weights: Position weights per bar. Supplied only to measure turnover.

    Returns:
        A ``PerformanceStats``. All fields are 0.0 for a degenerate input rather
        than NaN, so downstream comparisons never silently propagate nulls.
    """
    r = returns.dropna()
    n = int(r.size)
    if n < 2:
        return PerformanceStats(
            n_observations=n,
            total_return=0.0,
            annual_return=0.0,
            annual_volatility=0.0,
            sharpe=0.0,
            sortino=0.0,
            max_drawdown=0.0,
            calmar=0.0,
            hit_rate=0.0,
            time_in_market=0.0,
            turnover=0.0,
            skew=0.0,
            kurtosis=3.0,
        )

    equity = (1.0 + r).cumprod()
    total_return = float(equity.iloc[-1] - 1.0)
    years = n / periods_per_year

    # Geometric annualisation. The arithmetic mean overstates realised growth
    # whenever volatility is material, which for crypto it always is.
    annual_return = (
        float(equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 and equity.iloc[-1] > 0 else 0.0
    )
    annual_vol = float(r.std(ddof=1) * np.sqrt(periods_per_year))

    downside = r[r < 0]
    downside_vol = float(downside.std(ddof=1) * np.sqrt(periods_per_year)) if downside.size > 1 else 0.0
    sortino = float(annual_return / downside_vol) if downside_vol > 0 else 0.0

    mdd = max_drawdown(equity)
    calmar = float(annual_return / mdd) if mdd > 0 else 0.0

    turnover = 0.0
    time_in_market = 1.0
    active = pd.Series(True, index=r.index)
    if weights is not None and weights.size > 1:
        turnover = float(weights.diff().abs().sum() / years) if years > 0 else 0.0
        active = weights.reindex(r.index).fillna(0.0) != 0.0
        time_in_market = float(active.mean())

    active_returns = r[active]
    hit_rate = float((active_returns > 0).mean()) if active_returns.size else 0.0

    return PerformanceStats(
        n_observations=n,
        total_return=total_return,
        annual_return=annual_return,
        annual_volatility=annual_vol,
        sharpe=sharpe_ratio(r, periods_per_year),
        sortino=sortino,
        max_drawdown=mdd,
        calmar=calmar,
        hit_rate=hit_rate,
        time_in_market=time_in_market,
        turnover=turnover,
        skew=float(stats.skew(r, bias=False)),
        kurtosis=float(stats.kurtosis(r, fisher=False, bias=False)),
    )
