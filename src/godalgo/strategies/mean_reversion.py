"""Ornstein-Uhlenbeck mean reversion.

Grounding: Avellaneda & Lee (2010), "Statistical Arbitrage in the US Equities
Market", model a residual spread as an OU process and trade its normalised
deviation -- the "s-score". We apply the same machinery to a single asset's
deviation from its own trailing mean, which is the degenerate case of their
setup with no factor model attached.

The three things that separate this from a naive Bollinger-band system, and that
account for most of the difference between a reversion strategy that survives
and one that does not:

1. **Half-life gating.** A spread that reverts with a 300-bar half-life is
   statistically mean-reverting and economically untradeable -- financing and
   fees consume the edge before it converges. Signals are suppressed unless the
   fitted half-life sits inside a tradeable band. The lower bound matters too:
   a 1-bar half-life is usually microstructure noise, not an edge we can capture
   at our latency.

2. **Asymmetric entry and exit thresholds.** Entering at |z| = 2 and exiting at
   |z| = 0.5 rather than at the same level creates hysteresis. Symmetric
   thresholds produce a stream of round trips as z jitters across one boundary,
   and in fee terms that is how a reversion strategy bleeds out.

3. **A stop on the z-score itself.** Mean reversion's failure mode is a regime
   break -- the "mean" moves and the position scales into a loss that keeps
   growing, because the further it goes the more attractive the signal looks.
   A hard stop at ``stop_zscore`` is the admission that the model is wrong
   rather than early.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pandas as pd

from godalgo.core.types import Regime
from godalgo.features.indicators import rolling_zscore
from godalgo.features.regime import ou_half_life
from godalgo.strategies.base import ParamSpec, Strategy, StrategyParams

__all__ = ["MeanReversionParams", "MeanReversionStrategy"]


@dataclass(frozen=True)
class MeanReversionParams(StrategyParams):
    """Tunables for OU mean reversion."""

    lookback: int = 50
    """Window for the rolling mean and standard deviation of log price."""

    entry_zscore: float = 2.0
    """Absolute z-score at which a position is opened, against the deviation."""

    exit_zscore: float = 0.5
    """Absolute z-score at which an open position is closed. Must be < entry."""

    stop_zscore: float = 4.0
    """Absolute z-score at which the position is abandoned. Must be > entry."""

    half_life_window: int = 120
    """Window over which the OU half-life is refitted."""

    min_half_life: float = 2.0
    """Below this, reversion is too fast to capture -- treated as noise."""

    max_half_life: float = 100.0
    """Above this, reversion is too slow to be worth the carry."""

    refit_every: int = 20
    """Bars between half-life refits.

    Refitting every bar is expensive and adds no information -- the estimate is
    drawn from a rolling window that has barely changed one bar later.
    """

    SPACE: ClassVar[dict[str, ParamSpec]] = {
        "lookback": ParamSpec(10, 200, integer=True),
        "entry_zscore": ParamSpec(0.5, 4.0),
        "exit_zscore": ParamSpec(0.0, 2.0),
        "stop_zscore": ParamSpec(2.0, 8.0),
        "half_life_window": ParamSpec(50, 400, integer=True),
        "min_half_life": ParamSpec(1.0, 10.0),
        "max_half_life": ParamSpec(20.0, 400.0),
        "refit_every": ParamSpec(1, 100, integer=True),
    }

    def validate(self) -> None:
        super().validate()
        if self.exit_zscore >= self.entry_zscore:
            raise ValueError(
                f"exit_zscore ({self.exit_zscore}) must be < "
                f"entry_zscore ({self.entry_zscore})"
            )
        if self.stop_zscore <= self.entry_zscore:
            raise ValueError(
                f"stop_zscore ({self.stop_zscore}) must be > "
                f"entry_zscore ({self.entry_zscore})"
            )
        if self.min_half_life >= self.max_half_life:
            raise ValueError(
                f"min_half_life ({self.min_half_life}) must be < "
                f"max_half_life ({self.max_half_life})"
            )


class MeanReversionStrategy(Strategy):
    """Half-life-gated z-score reversion with hysteresis and a stop."""

    name: ClassVar[str] = "mean_reversion"
    favoured_regime: ClassVar[Regime] = Regime.MEAN_REVERTING

    def __init__(self, params: MeanReversionParams | None = None) -> None:
        super().__init__(params or MeanReversionParams())
        self.params: MeanReversionParams

    @property
    def warmup(self) -> int:
        return int(max(self.params.lookback, self.params.half_life_window))

    def generate(self, bars: pd.DataFrame) -> pd.Series:
        """Walk the z-score through an explicit position state machine.

        A stateful loop rather than vectorised threshold comparisons, because
        entry, exit, and stop depend on whether a position is *already* open.
        That path dependence cannot be expressed as a pointwise function of the
        z-score without losing the hysteresis that the design depends on.
        """
        if "close" not in bars:
            raise ValueError("bars must contain a 'close' column")

        close = bars["close"].astype(float)
        if close.empty:
            return pd.Series(dtype=float, index=bars.index)

        log_close = np.log(close)
        z = rolling_zscore(log_close, self.params.lookback)
        tradeable = self._tradeable_half_life(log_close)

        z_values = z.to_numpy()
        ok = tradeable.to_numpy()
        out = np.zeros(len(close), dtype=float)

        position = 0.0
        for i in range(self.warmup, len(close)):
            zi = z_values[i]

            if not np.isfinite(zi):
                out[i] = position
                continue

            abs_z = abs(zi)

            if position != 0.0:
                # Stop first: an invalidated model outranks a profitable-looking
                # signal, and at |z| > stop the signal looks its most attractive.
                if abs_z >= self.params.stop_zscore or abs_z <= self.params.exit_zscore:
                    position = 0.0
                else:
                    # Scale conviction with the remaining distance to the exit,
                    # so the position bleeds off as the trade works rather than
                    # flipping to flat in one step.
                    span = self.params.stop_zscore - self.params.exit_zscore
                    scaled = (abs_z - self.params.exit_zscore) / span
                    position = float(np.sign(position) * np.clip(scaled, 0.0, 1.0))
            elif ok[i] and self.params.entry_zscore <= abs_z < self.params.stop_zscore:
                # Fade the deviation: short a high z, long a low one.
                span = self.params.stop_zscore - self.params.exit_zscore
                scaled = (abs_z - self.params.exit_zscore) / span
                position = float(-np.sign(zi) * np.clip(scaled, 0.0, 1.0))

            out[i] = position

        return pd.Series(out, index=bars.index, name=self.name)

    def _tradeable_half_life(self, log_close: pd.Series) -> pd.Series:
        """Boolean mask: is the fitted half-life inside the tradeable band?

        Refits every ``refit_every`` bars on a trailing window and forward-fills
        between refits. Forward-filling is causal -- each value is carried from a
        fit that used only past data.
        """
        window = int(self.params.half_life_window)
        step = int(self.params.refit_every)
        n = len(log_close)
        mask = np.zeros(n, dtype=bool)

        prices = np.exp(log_close.to_numpy())
        current = False
        for i in range(n):
            if i >= window and (i - window) % step == 0:
                hl = ou_half_life(prices[i - window : i + 1])
                current = (
                    hl is not None
                    and self.params.min_half_life <= hl <= self.params.max_half_life
                )
            mask[i] = current

        return pd.Series(mask, index=log_close.index)
