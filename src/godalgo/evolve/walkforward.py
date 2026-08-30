"""Purged walk-forward evaluation.

The only defensible way to measure a self-tuning system: fit parameters on a
training window, evaluate them on the window that follows, roll forward, and
judge the strategy on the concatenated out-of-sample record. Every reported
number then comes from data the search had not seen when it chose.

Two refinements from Lopez de Prado, *Advances in Financial Machine Learning*
(2018), both of which matter here specifically because our features are built
from trailing windows:

**Purging.** A signal at the first test bar is computed from a window reaching
back into the training period. Train and test therefore share information even
though their index ranges do not overlap. Purging drops the tail of the training
window that the test set's features reach into.

**Embargo.** Serial correlation means bars just after the test window still carry
information about it. An embargo gap prevents the *next* fold's training set
from starting inside that shadow.

Skip either and out-of-sample scores drift upward toward in-sample ones -- which
defeats the entire purpose of running the split.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from godalgo.backtest.engine import BacktestConfig, run_backtest
from godalgo.backtest.metrics import PerformanceStats, compute_stats
from godalgo.strategies.base import Strategy, StrategyParams

__all__ = ["Fold", "WalkForwardResult", "walk_forward_evaluate", "walk_forward_splits"]


@dataclass(frozen=True, slots=True)
class Fold:
    """One train/test split, as positional index ranges."""

    index: int
    train_start: int
    train_end: int
    """Exclusive, already purged."""

    test_start: int
    test_end: int
    """Exclusive."""

    @property
    def train_slice(self) -> slice:
        return slice(self.train_start, self.train_end)

    @property
    def test_slice(self) -> slice:
        return slice(self.test_start, self.test_end)

    def __repr__(self) -> str:
        return (
            f"Fold({self.index}: train[{self.train_start}:{self.train_end}] "
            f"test[{self.test_start}:{self.test_end}])"
        )


def walk_forward_splits(
    n_obs: int,
    train_size: int,
    test_size: int,
    *,
    purge: int = 0,
    embargo: int = 0,
    anchored: bool = False,
) -> list[Fold]:
    """Generate purged, embargoed walk-forward folds.

    Args:
        n_obs: Total observations available.
        train_size: Training window length in bars. For anchored mode this is
            the length of the *first* window; later ones grow.
        test_size: Out-of-sample window length in bars.
        purge: Bars dropped from the end of each training window. Should be at
            least the longest trailing window any feature uses -- otherwise the
            test set's warm-up leaks backwards into training.
        embargo: Bars skipped after each test window before the next fold.
        anchored: If True the training window always starts at 0 and expands
            (more data, but the distant past may no longer be representative).
            If False it rolls at fixed length (adapts faster, forgets sooner).

    Returns:
        Folds in chronological order. Empty if the data cannot accommodate even
        one fold.

    Raises:
        ValueError: If sizes are non-positive or purge/embargo are negative.
    """
    if train_size <= 0 or test_size <= 0:
        raise ValueError("train_size and test_size must be positive")
    if purge < 0 or embargo < 0:
        raise ValueError("purge and embargo must be non-negative")

    folds: list[Fold] = []
    test_start = train_size
    fold_index = 0

    while test_start + test_size <= n_obs:
        train_start = 0 if anchored else max(0, test_start - train_size)
        train_end = max(train_start, test_start - purge)

        if train_end - train_start >= max(10, purge):
            folds.append(
                Fold(
                    index=fold_index,
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_start + test_size,
                )
            )
            fold_index += 1

        test_start += test_size + embargo

    return folds


@dataclass(slots=True)
class WalkForwardResult:
    """Concatenated out-of-sample record across all folds."""

    oos_returns: pd.Series
    """Stitched OOS returns. This is the only performance record that counts."""

    oos_stats: PerformanceStats
    fold_stats: list[PerformanceStats]
    chosen_params: list[dict[str, object]]
    """The parameter set selected in each fold's training window."""

    fold_trial_returns: pd.DataFrame
    """OOS returns per candidate, for PBO. One column per candidate."""

    n_trials: int
    """Total configurations evaluated -- the N that deflated Sharpe corrects for."""

    n_skipped: int = 0
    """Candidate/fold pairs that could not be evaluated.

    Surfaced rather than swallowed. A high count means candidates are failing
    systematically -- usually a warm-up longer than the fold, or a parameter
    type error -- which silently narrows the search to whatever still runs
    while the reported statistics continue to look healthy.
    """

    @property
    def parameter_stability(self) -> dict[str, float]:
        """Coefficient of variation of each parameter across folds.

        A parameter that swings wildly from fold to fold is not being *learned*
        -- it is being refit to noise. High values here are the clearest early
        warning that a configuration will not survive live, and they are visible
        long before the out-of-sample curve turns over.
        """
        if not self.chosen_params:
            return {}
        stability: dict[str, float] = {}
        keys = self.chosen_params[0].keys()
        for key in keys:
            values = [float(p[key]) for p in self.chosen_params if isinstance(p[key], (int, float))]
            if len(values) < 2:
                continue
            mean = float(np.mean(values))
            if abs(mean) < 1e-12:
                stability[key] = 0.0
            else:
                stability[key] = float(np.std(values) / abs(mean))
        return stability


def walk_forward_evaluate(
    bars: pd.DataFrame,
    candidates: Sequence[tuple[StrategyParams, StrategyParams]],
    build: Callable[[StrategyParams, StrategyParams], tuple[Strategy, Strategy]],
    config: BacktestConfig | None = None,
    *,
    train_size: int = 4000,
    test_size: int = 1000,
    purge: int | None = None,
    embargo: int = 100,
    anchored: bool = False,
    symbol: str = "UNKNOWN",
) -> WalkForwardResult:
    """Select parameters in-sample per fold and record out-of-sample results.

    In each fold every candidate is backtested on the training window; the one
    with the best in-sample Sharpe is then run on the test window, and *that*
    test-window return stream is what gets kept.

    Args:
        bars: Full OHLCV history, oldest first.
        candidates: ``(momentum_params, reversion_params)`` pairs to search over.
        build: Factory turning a parameter pair into strategy instances.
        config: Backtest configuration.
        train_size: Training window length in bars.
        test_size: Test window length in bars.
        purge: Bars purged from each training tail. Defaults to the engine's
            warm-up requirement, which is the correct value -- that is exactly
            how far a test-window feature reaches back.
        embargo: Bars skipped between folds.
        anchored: Expanding vs rolling training window.
        symbol: Label passed through to the engine.

    Returns:
        A ``WalkForwardResult``.

    Raises:
        ValueError: If no candidates are supplied, or the data is too short for
            a single fold at the requested sizes.
    """
    if not candidates:
        raise ValueError("at least one candidate parameter set is required")

    cfg = config or BacktestConfig()

    if purge is None:
        # The engine discards `warmup` bars; a test-window signal reaches back
        # exactly that far, so that is the correct purge length.
        sample_mom, sample_rev = build(*candidates[0])
        purge = max(sample_mom.warmup, sample_rev.warmup, cfg.regime_window)

    folds = walk_forward_splits(
        len(bars),
        train_size,
        test_size,
        purge=purge,
        embargo=embargo,
        anchored=anchored,
    )
    if not folds:
        raise ValueError(
            f"{len(bars)} bars cannot produce a fold at train_size={train_size}, "
            f"test_size={test_size}, purge={purge}"
        )

    oos_chunks: list[pd.Series] = []
    fold_stats: list[PerformanceStats] = []
    chosen: list[dict[str, object]] = []
    trial_columns: dict[int, list[pd.Series]] = {i: [] for i in range(len(candidates))}
    n_trials = 0
    n_skipped = 0

    for fold in folds:
        train_bars = bars.iloc[fold.train_slice]
        test_bars = bars.iloc[fold.test_slice]

        best_sharpe = -np.inf
        best_idx = 0

        for idx, (mom_params, rev_params) in enumerate(candidates):
            try:
                mom, rev = build(mom_params, rev_params)
                train_result = run_backtest(train_bars, mom, rev, cfg, symbol)
            except ValueError:
                # Window too short for this configuration's warm-up; it simply
                # does not compete in this fold. Counted, not hidden.
                n_skipped += 1
                continue
            n_trials += 1
            if train_result.stats.sharpe > best_sharpe:
                best_sharpe = train_result.stats.sharpe
                best_idx = idx

        # Every candidate is also run OOS -- not to select, but because PBO
        # needs the full cross-section of what each choice *would* have earned.
        for idx, (mom_params, rev_params) in enumerate(candidates):
            try:
                mom, rev = build(mom_params, rev_params)
                test_result = run_backtest(test_bars, mom, rev, cfg, symbol)
            except ValueError:
                continue
            trial_columns[idx].append(test_result.returns)
            if idx == best_idx:
                oos_chunks.append(test_result.returns)
                fold_stats.append(test_result.stats)
                chosen.append(
                    {
                        **{f"mom_{k}": v for k, v in mom_params.to_dict().items()},
                        **{f"rev_{k}": v for k, v in rev_params.to_dict().items()},
                    }
                )

    if not oos_chunks:
        raise ValueError("no fold produced a usable out-of-sample result")

    oos_returns = pd.concat(oos_chunks).sort_index()

    trial_frame = pd.DataFrame(
        {
            f"cand_{idx}": pd.concat(chunks).sort_index()
            for idx, chunks in trial_columns.items()
            if chunks
        }
    ).dropna(how="all")

    return WalkForwardResult(
        oos_returns=oos_returns,
        oos_stats=compute_stats(oos_returns, cfg.periods_per_year),
        fold_stats=fold_stats,
        chosen_params=chosen,
        fold_trial_returns=trial_frame,
        n_trials=n_trials,
        n_skipped=n_skipped,
    )
