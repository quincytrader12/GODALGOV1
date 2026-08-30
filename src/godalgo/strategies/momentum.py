"""Time-series momentum.

Grounding: Moskowitz, Ooi & Pedersen (2012), "Time Series Momentum", found that
an asset's own past 12-month excess return predicts its next-month return across
58 instruments and 25 years -- distinct from Jegadeesh & Titman (1993)
cross-sectional momentum, which ranks assets against each other. We use the
time-series form because it works on a single instrument and does not need a
broad cross-section, which matters when trading a handful of crypto pairs.

Two departures from the equity-literature defaults, both forced by the asset
class:

1. **Volatility scaling of the signal**, following Baltas & Kosowski (2013) and
   the vol-managed portfolio result of Moreira & Muir (2017). Raw momentum sizes
   its largest bets exactly when volatility is highest, which is when drawdowns
   are worst. Dividing by trailing vol substantially improves the Sharpe of
   trend strategies. Crypto vol ranges over an order of magnitude, so this is
   not optional here.

2. **A blend of lookbacks rather than one.** Single-lookback trend systems are
   notoriously sensitive to the exact window -- a 40-day and 60-day breakout can
   differ wildly over a given sample for no economic reason. Averaging several
   horizons trades a little peak backtest performance for much less parameter
   fragility, which is the right side of that trade when a self-improvement loop
   is searching the parameter space and will happily latch onto a lucky window.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pandas as pd

from godalgo.core.types import Regime
from godalgo.features.indicators import ewma_volatility, log_returns
from godalgo.strategies.base import ParamSpec, Strategy, StrategyParams

__all__ = ["MomentumParams", "MomentumStrategy"]


@dataclass(frozen=True)
class MomentumParams(StrategyParams):
    """Tunables for time-series momentum."""

    fast_lookback: int = 20
    """Shortest trend horizon, in bars."""

    slow_lookback: int = 100
    """Longest trend horizon. Must exceed ``fast_lookback``."""

    n_horizons: int = 3
    """How many geometrically spaced lookbacks to blend between fast and slow."""

    vol_halflife: float = 30.0
    """EWMA half-life for the volatility used to scale the signal."""

    signal_cap: float = 2.0
    """Clip on the vol-scaled score before the squashing function.

    Bounds the influence of any single extreme move, so one outlier bar cannot
    dominate the blended signal.
    """

    entry_threshold: float = 0.10
    """Dead zone. Blended conviction below this in absolute value is set flat.

    Suppresses churn when the trend reading is ambiguous; every crossing of zero
    would otherwise pay the spread twice.
    """

    SPACE: ClassVar[dict[str, ParamSpec]] = {
        "fast_lookback": ParamSpec(5, 60, integer=True),
        "slow_lookback": ParamSpec(40, 300, integer=True),
        "n_horizons": ParamSpec(1, 5, integer=True),
        "vol_halflife": ParamSpec(5.0, 120.0),
        "signal_cap": ParamSpec(1.0, 4.0),
        "entry_threshold": ParamSpec(0.0, 0.5),
    }

    def validate(self) -> None:
        super().validate()
        if self.fast_lookback >= self.slow_lookback:
            raise ValueError(
                f"fast_lookback ({self.fast_lookback}) must be < "
                f"slow_lookback ({self.slow_lookback})"
            )

    def horizons(self) -> list[int]:
        """Geometrically spaced lookbacks from fast to slow.

        Geometric rather than linear because trend signals scale with the
        logarithm of the horizon -- 20 vs 40 bars is a far bigger change in
        behaviour than 200 vs 220.
        """
        if self.n_horizons <= 1:
            return [int(self.slow_lookback)]
        raw = np.geomspace(self.fast_lookback, self.slow_lookback, int(self.n_horizons))
        return sorted({max(2, int(round(h))) for h in raw})


class MomentumStrategy(Strategy):
    """Volatility-scaled, multi-horizon time-series momentum."""

    name: ClassVar[str] = "momentum"
    favoured_regime: ClassVar[Regime] = Regime.TRENDING

    def __init__(self, params: MomentumParams | None = None) -> None:
        super().__init__(params or MomentumParams())
        self.params: MomentumParams

    @property
    def warmup(self) -> int:
        # The slow lookback plus room for the EWMA vol estimate to stabilise.
        return int(self.params.slow_lookback + 3 * self.params.vol_halflife)

    def generate(self, bars: pd.DataFrame) -> pd.Series:
        """Blend vol-scaled trend scores across horizons into one conviction."""
        if "close" not in bars:
            raise ValueError("bars must contain a 'close' column")

        close = bars["close"].astype(float)
        if close.empty:
            return pd.Series(dtype=float, index=bars.index)

        returns = log_returns(close)
        # Per-bar vol scaled to each horizon: a k-bar return has ~sqrt(k) times
        # the dispersion of a 1-bar return, so the denominator must scale too.
        bar_vol = ewma_volatility(returns, halflife=self.params.vol_halflife)

        scores = []
        for horizon in self.params.horizons():
            trend = np.log(close) - np.log(close.shift(horizon))
            horizon_vol = bar_vol * np.sqrt(horizon)
            score = trend / horizon_vol.replace(0.0, np.nan)
            scores.append(score.clip(-self.params.signal_cap, self.params.signal_cap))

        blended = pd.concat(scores, axis=1).mean(axis=1)

        # tanh squash: monotone, bounded in [-1, 1], and saturating -- so a
        # 6-sigma trend does not demand three times the position of a 2-sigma
        # one. Strong trends are not proportionally more likely to continue.
        conviction = np.tanh(blended / self.params.signal_cap)

        conviction = conviction.where(
            conviction.abs() >= self.params.entry_threshold, 0.0
        )

        # Warm-up bars are flat by construction, never NaN.
        conviction.iloc[: self.warmup] = 0.0
        return conviction.fillna(0.0).rename(self.name)
