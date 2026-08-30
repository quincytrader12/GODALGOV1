"""Parameter space invariants."""

import numpy as np
import pytest

from godalgo.evolve.search import perturb_params, propose_candidates, sample_params
from godalgo.strategies.base import ParamSpec
from godalgo.strategies.mean_reversion import MeanReversionParams
from godalgo.strategies.momentum import MomentumParams


def test_integer_paramspec_returns_a_real_int():
    """Regression.

    Returning ``float(round(v))`` here produced values like ``36.0`` that passed
    validation and then failed inside pandas, which rejects a float
    ``min_periods``. Candidates were silently skipped and the search collapsed
    onto a single configuration while still reporting healthy statistics.
    """
    spec = ParamSpec(10, 200, integer=True)
    value = spec.clip(36.7)
    assert isinstance(value, int) and not isinstance(value, float)
    assert value == 37


def test_float_paramspec_stays_float():
    assert isinstance(ParamSpec(0.0, 1.0).clip(0.5), float)


def test_sampled_params_are_always_valid_and_integer_typed():
    rng = np.random.default_rng(0)
    for _ in range(200):
        mom = sample_params(MomentumParams(), rng)
        rev = sample_params(MeanReversionParams(), rng)
        mom.validate()
        rev.validate()
        assert isinstance(mom.fast_lookback, int)
        assert isinstance(rev.lookback, int)
        assert mom.fast_lookback < mom.slow_lookback
        assert rev.exit_zscore < rev.entry_zscore < rev.stop_zscore


def test_perturbations_stay_in_bounds():
    rng = np.random.default_rng(1)
    params = MomentumParams()
    for _ in range(200):
        params = perturb_params(params, rng, scale=0.5)
        params.validate()


def test_incumbent_is_on_the_ballot():
    """A search that cannot conclude 'keep what we have' will always churn."""
    mom, rev = MomentumParams(), MeanReversionParams()
    candidates = propose_candidates(mom, rev)
    assert candidates[0] == (mom, rev)


def test_replace_rejects_unknown_parameters():
    with pytest.raises(ValueError, match="unknown parameters"):
        MomentumParams().replace(nonexistent=1.0)


def test_out_of_bounds_values_are_clipped_not_accepted():
    assert MomentumParams().replace(fast_lookback=9999).fast_lookback == 60


def test_risk_limits_are_not_in_any_search_space():
    """The optimiser must not be able to widen its own leash."""
    for space in (MomentumParams.SPACE, MeanReversionParams.SPACE):
        for name in space:
            assert "drawdown" not in name
            assert "loss_limit" not in name
            assert "gross" not in name
