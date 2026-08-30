"""Parameter proposal generation.

The search space is a closed box. Every proposal is a point inside the
``ParamSpec`` bounds declared on the strategy, clipped and validated before it
can reach a backtest. Nothing here generates code, and nothing here can reach a
risk limit -- ``RiskLimits`` is not part of any strategy's ``SPACE``, so the
optimiser cannot widen its own leash.

Random search rather than grid search, following Bergstra & Bengio (2012):
with a fixed evaluation budget, random sampling beats a grid because most
parameters barely matter, and a grid spends most of its budget resolving the
ones that do not. It also keeps the trial count honest and explicit, which
matters because that count feeds directly into the deflated Sharpe correction.

Local perturbation search is offered alongside it for incremental refinement
around a known-good incumbent, which is the normal mode once the system has a
configuration in production.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from godalgo.strategies.base import ParamSpec, StrategyParams

__all__ = ["SearchConfig", "perturb_params", "propose_candidates", "sample_params"]


@dataclass(frozen=True, slots=True)
class SearchConfig:
    """Controls proposal generation."""

    n_candidates: int = 24
    """Configurations per search round.

    Kept modest on purpose. Every extra candidate raises the bar the deflated
    Sharpe ratio applies, so a wider search must earn its width -- this is the
    mechanism that stops the loop from brute-forcing its way to a lucky result.
    """

    perturbation_scale: float = 0.15
    """Std-dev of local moves, as a fraction of each parameter's range."""

    explore_fraction: float = 0.30
    """Share of candidates drawn uniformly rather than near the incumbent.

    Pure local search converges into whatever basin it started in. Pure random
    search never refines. The mix keeps both available.
    """

    seed: int | None = None

    def __post_init__(self) -> None:
        if self.n_candidates < 1:
            raise ValueError("n_candidates must be >= 1")
        if not 0.0 <= self.explore_fraction <= 1.0:
            raise ValueError("explore_fraction must be in [0, 1]")
        if self.perturbation_scale <= 0:
            raise ValueError("perturbation_scale must be positive")


def sample_params(
    template: StrategyParams,
    rng: np.random.Generator,
    max_attempts: int = 50,
) -> StrategyParams:
    """Draw a uniformly random parameter set from the declared space.

    Retries on cross-parameter constraint failures -- e.g. momentum requires
    ``fast_lookback < slow_lookback``, which a uniform draw violates about half
    the time. Falls back to the template if the constraints cannot be satisfied,
    so the caller always gets a valid object.
    """
    for _ in range(max_attempts):
        draw = {
            name: _uniform(spec, rng) for name, spec in type(template).SPACE.items()
        }
        try:
            candidate = template.replace(**draw)
            candidate.validate()
            return candidate
        except ValueError:
            continue
    return template


def perturb_params(
    base: StrategyParams,
    rng: np.random.Generator,
    scale: float = 0.15,
    max_attempts: int = 50,
) -> StrategyParams:
    """Gaussian move around ``base``, scaled to each parameter's own range.

    Scaling by range rather than by value keeps the step size meaningful for
    parameters with very different units -- a lookback of 100 bars and a z-score
    threshold of 2.0 should not receive proportionally identical nudges.
    """
    space = type(base).SPACE
    for _ in range(max_attempts):
        draw = {}
        for name, spec in space.items():
            current = float(getattr(base, name))
            step = rng.normal(0.0, scale * (spec.high - spec.low))
            draw[name] = spec.clip(current + step)
        try:
            candidate = base.replace(**draw)
            candidate.validate()
            return candidate
        except ValueError:
            continue
    return base


def propose_candidates(
    momentum_template: StrategyParams,
    reversion_template: StrategyParams,
    config: SearchConfig | None = None,
    *,
    include_incumbent: bool = True,
) -> list[tuple[StrategyParams, StrategyParams]]:
    """Build a round of candidate parameter pairs.

    Args:
        momentum_template: Incumbent momentum parameters, used as the local
            search centre.
        reversion_template: Incumbent reversion parameters, likewise.
        config: Search controls.
        include_incumbent: Whether to include the unmodified incumbent as a
            candidate. Keep this True -- it is what lets a search round conclude
            "nothing beat what we already had", which is the correct outcome
            most of the time and the one a search will never reach if the
            incumbent is not on the ballot.

    Returns:
        Candidate ``(momentum, reversion)`` pairs, incumbent first when included.
    """
    cfg = config or SearchConfig()
    rng = np.random.default_rng(cfg.seed)

    candidates: list[tuple[StrategyParams, StrategyParams]] = []
    if include_incumbent:
        candidates.append((momentum_template, reversion_template))

    remaining = max(0, cfg.n_candidates - len(candidates))
    n_explore = int(round(remaining * cfg.explore_fraction))

    for i in range(remaining):
        if i < n_explore:
            mom = sample_params(momentum_template, rng)
            rev = sample_params(reversion_template, rng)
        else:
            mom = perturb_params(momentum_template, rng, cfg.perturbation_scale)
            rev = perturb_params(reversion_template, rng, cfg.perturbation_scale)
        candidates.append((mom, rev))

    return candidates


def _uniform(spec: ParamSpec, rng: np.random.Generator) -> float:
    value = rng.uniform(spec.low, spec.high)
    return spec.clip(value)


def count_trials(candidates: Sequence[tuple[StrategyParams, StrategyParams]]) -> int:
    """Number of distinct configurations in a round.

    Exposed separately because this number must be reported honestly to the
    deflated Sharpe calculation. Undercounting it is the easiest way to make a
    worthless configuration clear the gate, and it is an easy mistake to make by
    accident when candidates are generated in more than one place.
    """
    return len(candidates)
