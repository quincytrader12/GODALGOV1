"""Regime-aware allocation between momentum and mean reversion.

The central design claim of this system: momentum and mean reversion are not two
independent alphas to be averaged, they are opposite bets on the sign of serial
correlation. A naive 50/50 blend is close to self-cancelling -- one book pays the
other's spread, and what survives is mostly fees.

So allocation is conditional. The regime classifier estimates which
autocorrelation sign currently holds, and capital follows that estimate:

* TRENDING       -> momentum gets the weight
* MEAN_REVERTING -> reversion gets the weight
* INDETERMINATE  -> both are cut back hard

Two properties are load-bearing:

**Soft weights, not a hard switch.** The regime estimate is noisy. A binary
switch would flip the entire book on a marginal change in the test statistic,
paying full turnover for a coin flip. Weights move continuously with confidence.

**A floor on the off-regime strategy.** The out-of-favour strategy keeps a small
allocation rather than going to zero. The classifier is a lagging estimate --
regimes turn before the statistics confirm it -- and a small standing position in
the other book softens the handover.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from godalgo.core.types import Regime

__all__ = ["AllocationConfig", "blend_signals", "regime_weights"]


@dataclass(frozen=True, slots=True)
class AllocationConfig:
    """Controls how regime confidence maps to strategy weights."""

    off_regime_floor: float = 0.15
    """Weight retained by the out-of-favour strategy at maximum confidence."""

    indeterminate_scale: float = 0.35
    """Total risk retained when no regime can be established.

    Not zero. An indeterminate reading means the tests disagree, not that the
    market has stopped moving -- and forcing a full exit on every ambiguous bar
    is itself a large, uncompensated turnover cost.
    """

    confidence_floor: float = 0.20
    """Minimum risk scale even at zero confidence within a declared regime."""

    def __post_init__(self) -> None:
        for name in ("off_regime_floor", "indeterminate_scale", "confidence_floor"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if self.off_regime_floor > 0.5:
            raise ValueError("off_regime_floor above 0.5 inverts the intended allocation")


def regime_weights(
    regime: Regime,
    confidence: float,
    config: AllocationConfig | None = None,
) -> tuple[float, float]:
    """Map a regime call to ``(momentum_weight, reversion_weight)``.

    Weights sum to at most 1.0 -- the shortfall is deliberate de-risking when
    confidence is low, not capital waiting to be deployed elsewhere.

    Args:
        regime: Classifier output.
        confidence: Classifier confidence in [0, 1].
        config: Allocation controls; defaults used if omitted.

    Returns:
        ``(momentum_weight, reversion_weight)``, each in [0, 1].
    """
    cfg = config or AllocationConfig()
    confidence = float(np.clip(confidence, 0.0, 1.0))

    if regime is Regime.INDETERMINATE:
        # Split what little risk we take evenly -- with no directional view on
        # autocorrelation there is no basis for preferring either book.
        half = cfg.indeterminate_scale / 2.0
        return half, half

    # Risk scale ramps from the floor to full as confidence rises.
    scale = cfg.confidence_floor + (1.0 - cfg.confidence_floor) * confidence

    # Within that budget the off-regime strategy keeps its floor and the
    # favoured strategy takes the remainder.
    off = scale * cfg.off_regime_floor
    favoured = max(0.0, scale - off)

    if regime is Regime.TRENDING:
        return favoured, off
    return off, favoured


def blend_signals(
    momentum: pd.Series,
    reversion: pd.Series,
    regimes: pd.Series,
    confidences: pd.Series,
    config: AllocationConfig | None = None,
) -> pd.DataFrame:
    """Combine both strategies' convictions into one target series.

    Args:
        momentum: Momentum conviction per bar, in [-1, 1].
        reversion: Reversion conviction per bar, in [-1, 1].
        regimes: ``Regime`` per bar.
        confidences: Classifier confidence per bar, in [0, 1].
        config: Allocation controls.

    Returns:
        Frame indexed like the inputs with columns ``momentum``, ``reversion``
        (post-weight contributions), ``combined`` (their sum, the target
        conviction), and ``w_momentum`` / ``w_reversion`` for attribution.

    Raises:
        ValueError: If the input indexes are not aligned. Silent misalignment
            here would blend a signal with a regime call from a different bar,
            which is a lookahead bug that no test downstream would catch.
    """
    cfg = config or AllocationConfig()

    index = momentum.index
    for name, series in (
        ("reversion", reversion),
        ("regimes", regimes),
        ("confidences", confidences),
    ):
        if not series.index.equals(index):
            raise ValueError(f"{name} index is not aligned with momentum index")

    w_mom = np.empty(len(index), dtype=float)
    w_rev = np.empty(len(index), dtype=float)
    regime_values = regimes.to_numpy()
    confidence_values = confidences.to_numpy(dtype=float)

    for i in range(len(index)):
        regime = regime_values[i]
        if not isinstance(regime, Regime):
            regime = Regime.INDETERMINATE
        conf = confidence_values[i]
        w_mom[i], w_rev[i] = regime_weights(
            regime, 0.0 if not np.isfinite(conf) else conf, cfg
        )

    mom_contrib = momentum.fillna(0.0).to_numpy(dtype=float) * w_mom
    rev_contrib = reversion.fillna(0.0).to_numpy(dtype=float) * w_rev

    return pd.DataFrame(
        {
            "momentum": mom_contrib,
            "reversion": rev_contrib,
            "combined": np.clip(mom_contrib + rev_contrib, -1.0, 1.0),
            "w_momentum": w_mom,
            "w_reversion": w_rev,
        },
        index=index,
    )
