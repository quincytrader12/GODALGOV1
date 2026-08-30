"""Walk-forward split geometry and allocator behaviour."""

from itertools import pairwise

import pandas as pd
import pytest

from godalgo.core.types import Regime
from godalgo.evolve.walkforward import walk_forward_splits
from godalgo.portfolio.allocator import AllocationConfig, blend_signals, regime_weights
from godalgo.portfolio.sizing import apply_turnover_buffer


def test_train_never_overlaps_test():
    for fold in walk_forward_splits(10_000, 2000, 500, purge=200, embargo=50):
        assert fold.train_end <= fold.test_start


def test_purge_creates_the_requested_gap():
    purge = 250
    for fold in walk_forward_splits(10_000, 2000, 500, purge=purge):
        assert fold.test_start - fold.train_end == purge


def test_embargo_separates_consecutive_folds():
    embargo = 100
    folds = walk_forward_splits(10_000, 2000, 500, purge=0, embargo=embargo)
    for earlier, later in pairwise(folds):
        assert later.test_start - earlier.test_end == embargo


def test_folds_move_forward_in_time():
    folds = walk_forward_splits(10_000, 2000, 500, purge=100, embargo=50)
    starts = [f.test_start for f in folds]
    assert starts == sorted(starts)
    assert len(set(starts)) == len(starts)


def test_anchored_windows_expand_rolling_windows_do_not():
    anchored = walk_forward_splits(10_000, 2000, 500, anchored=True)
    rolling = walk_forward_splits(10_000, 2000, 500, anchored=False)
    assert all(f.train_start == 0 for f in anchored)
    assert anchored[-1].train_end - anchored[-1].train_start > anchored[0].train_end - anchored[0].train_start
    assert len({f.train_end - f.train_start for f in rolling}) == 1


def test_insufficient_data_yields_no_folds():
    assert walk_forward_splits(100, 2000, 500) == []


def test_invalid_sizes_raise():
    with pytest.raises(ValueError):
        walk_forward_splits(1000, 0, 100)
    with pytest.raises(ValueError):
        walk_forward_splits(1000, 100, 100, purge=-1)


# --- allocator -------------------------------------------------------------

def test_favoured_strategy_dominates_in_its_regime():
    mom, rev = regime_weights(Regime.TRENDING, confidence=1.0)
    assert mom > rev
    rev_mom, rev_rev = regime_weights(Regime.MEAN_REVERTING, confidence=1.0)
    assert rev_rev > rev_mom


def test_off_regime_strategy_keeps_a_floor():
    """The classifier lags regime turns; a small standing position eases handover."""
    _, rev = regime_weights(Regime.TRENDING, confidence=1.0)
    assert rev > 0.0


def test_indeterminate_regime_derisks_both_books():
    mom, rev = regime_weights(Regime.INDETERMINATE, confidence=1.0)
    trending_mom, _ = regime_weights(Regime.TRENDING, confidence=1.0)
    assert mom + rev < trending_mom


def test_weights_never_exceed_full_allocation():
    for regime in Regime:
        for confidence in (0.0, 0.25, 0.5, 0.75, 1.0):
            mom, rev = regime_weights(regime, confidence)
            assert 0.0 <= mom <= 1.0 and 0.0 <= rev <= 1.0
            assert mom + rev <= 1.0 + 1e-12


def test_blend_rejects_misaligned_indexes():
    """Silent misalignment would pair a signal with another bar's regime."""
    idx_a = pd.date_range("2024-01-01", periods=5, freq="1h", tz="UTC")
    idx_b = pd.date_range("2024-02-01", periods=5, freq="1h", tz="UTC")
    with pytest.raises(ValueError, match="not aligned"):
        blend_signals(
            pd.Series(0.5, index=idx_a),
            pd.Series(0.5, index=idx_b),
            pd.Series([Regime.TRENDING] * 5, index=idx_a),
            pd.Series(1.0, index=idx_a),
        )


def test_turnover_buffer_suppresses_small_moves_but_honours_exits():
    target = pd.Series([0.0, 0.50, 0.52, 0.54, 0.0])
    held = apply_turnover_buffer(target, buffer=0.10)
    assert held.iloc[1] == 0.50
    assert held.iloc[2] == 0.50      # +0.02 is below the band
    assert held.iloc[3] == 0.50
    assert held.iloc[4] == 0.0       # a full exit is never suppressed


def test_allocation_config_rejects_inverted_floor():
    with pytest.raises(ValueError):
        AllocationConfig(off_regime_floor=0.9)
