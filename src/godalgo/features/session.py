"""Session and overnight-drift effects.

## Why the equity result does not port directly

The overnight drift anomaly is one of the most robust findings in equity market
microstructure. *Cooper, Cliff & Gulen (2008)* and *Lou, Polk & Skouras (2019)*
document that essentially the entire US equity risk premium accrues **close to
open**, while the intraday open-to-close return is approximately zero or
negative. *Hendershott, Livdan & Rosch (2020)* find the same night/day split
holds for market beta itself.

Crypto trades 24/7. There is no close, no open, and no gap. The literal
overnight return does not exist, and any implementation claiming to trade "the
overnight anomaly" on BTC/USDT is trading a quantity that is not defined.

## What does port: the mechanism, not the calendar

Lou, Polk & Skouras attribute the effect to **clientele**: different participant
types transact at different times of day, and their systematic order-flow
imbalances create predictable, partially-reversing price pressure. That
mechanism has nothing to do with an exchange being closed -- it needs only that
*who is trading* varies predictably with the clock.

That condition holds in crypto, through four structures:

1. **Regional sessions.** Asia, Europe, and US hours carry different participant
   mixes, and hence different flow.
2. **The CME gap.** Bitcoin CME futures do close -- Friday 22:00 UTC to Sunday
   23:00 UTC -- so a genuine close-to-open gap exists in a major venue, and it
   propagates to spot.
3. **Funding settlement.** Perpetual funding settles on a fixed 8-hour cycle
   (00:00 / 08:00 / 16:00 UTC), producing scheduled, predictable flow.
4. **Risk-asset spillover.** Crypto's correlation with equities concentrates
   beta in US cash-session hours.

So this module estimates a **conditional drift by hour-of-week**, learned from
the data rather than assumed from the equity calendar.

## The statistical problem, and the fix

Estimating 168 hour-of-week means is 168 simultaneous tests. Some will look
significant on noise alone, and a strategy that tilts toward whichever bucket
looked best is the same overfitting failure the promotion gate exists to stop --
just relocated into a feature.

The fix is **empirical-Bayes shrinkage** (James-Stein). Each bucket mean is
pulled toward the grand mean by an amount determined by its own noise: buckets
with few observations or high variance shrink almost entirely away, while a
bucket with a genuinely large and consistent effect survives. No bucket is
trusted on the strength of having looked good once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import numpy as np
import pandas as pd

__all__ = ["Bucketing", "SessionConfig", "SessionProfile", "fit_session_profile"]


class Bucketing(str, Enum):
    """How the clock is partitioned."""

    HOUR_OF_DAY = "hour_of_day"
    """24 buckets. Captures regional session and funding effects."""

    HOUR_OF_WEEK = "hour_of_week"
    """168 buckets. Captures weekend and CME-gap effects too, at the cost of
    seven times less data per bucket -- which shrinkage then handles."""

    SESSION = "session"
    """3 coarse buckets (Asia / Europe / US). Most data per bucket, least
    resolution. The right default when history is short."""


# UTC hour ranges. Deliberately coarse -- session boundaries are fuzzy, and
# precision here would be false.
_SESSIONS = {
    0: ("asia", range(8)),
    1: ("europe", range(8, 14)),
    2: ("us", range(14, 24)),
}


@dataclass(frozen=True, slots=True)
class SessionConfig:
    """Controls estimation and how strongly the result is allowed to matter."""

    bucketing: Bucketing = Bucketing.HOUR_OF_DAY
    min_observations: int = 30
    """Below this, a bucket is shrunk to zero regardless of what it shows."""

    tilt_weight: float = 0.20
    """How much of the final signal the session tilt may contribute.

    Deliberately a minority share. The session effect is a real but small
    conditional drift; letting it dominate a momentum or reversion signal would
    be trading the calendar rather than the market.
    """

    max_tilt: float = 1.0
    """Cap on the normalised tilt magnitude."""

    def __post_init__(self) -> None:
        if not 0.0 <= self.tilt_weight <= 1.0:
            raise ValueError("tilt_weight must be in [0, 1]")
        if self.min_observations < 2:
            raise ValueError("min_observations must be at least 2")


def bucket_of(moment: datetime, bucketing: Bucketing) -> int:
    """Bucket index for a timestamp. Assumes UTC."""
    if bucketing is Bucketing.HOUR_OF_DAY:
        return moment.hour
    if bucketing is Bucketing.HOUR_OF_WEEK:
        return moment.weekday() * 24 + moment.hour
    for index, (_, hours) in _SESSIONS.items():
        if moment.hour in hours:
            return index
    return 0


def _n_buckets(bucketing: Bucketing) -> int:
    return {
        Bucketing.HOUR_OF_DAY: 24,
        Bucketing.HOUR_OF_WEEK: 168,
        Bucketing.SESSION: 3,
    }[bucketing]


@dataclass(frozen=True, slots=True)
class SessionProfile:
    """Shrunk conditional drift per clock bucket."""

    bucketing: Bucketing
    raw_means: np.ndarray
    shrunk_means: np.ndarray
    """Post-shrinkage per-bar drift, in log-return units."""

    counts: np.ndarray
    shrinkage: np.ndarray
    """Per-bucket weight on the sample mean. Near 0 means fully shrunk away."""

    scale: float
    """Normalisation constant -- the dispersion of shrunk means."""

    config: SessionConfig = field(default_factory=SessionConfig)

    def drift(self, moment: datetime) -> float:
        """Shrunk expected per-bar log return for this bucket."""
        return float(self.shrunk_means[bucket_of(moment, self.bucketing)])

    def expected_drift_bps(self, moment: datetime, horizon_bars: float = 1.0) -> float:
        """Expected drift in bps over ``horizon_bars``.

        Feeds the router's cost gate, so it is reported as a real economic
        magnitude rather than a normalised score -- a tilt that cannot pay the
        spread should not be able to argue for a trade.
        """
        return float(self.drift(moment) * horizon_bars * 1e4)

    def tilt(self, moment: datetime) -> float:
        """Normalised directional tilt in ``[-max_tilt, max_tilt]``.

        Scaled by the dispersion of the shrunk means, so a profile whose buckets
        were all shrunk to near zero produces a near-zero tilt rather than
        amplifying noise up to full scale.
        """
        if self.scale <= 0:
            return 0.0
        raw = self.drift(moment) / self.scale
        return float(np.clip(raw, -self.config.max_tilt, self.config.max_tilt))

    def tilt_series(self, index: pd.DatetimeIndex) -> pd.Series:
        """Tilt for every timestamp in an index."""
        return pd.Series([self.tilt(t) for t in index], index=index, name="session_tilt")

    def summary(self, top: int = 5) -> str:
        """The strongest surviving buckets, for inspection."""
        order = np.argsort(-np.abs(self.shrunk_means))[:top]
        lines = [f"{self.bucketing.value}: scale={self.scale:.6f}"]
        for b in order:
            label = self._label(int(b))
            lines.append(
                f"  {label:<14s} raw={self.raw_means[b]*1e4:+7.2f}bps  "
                f"shrunk={self.shrunk_means[b]*1e4:+7.2f}bps  "
                f"n={int(self.counts[b]):<5d} lambda={self.shrinkage[b]:.2f}"
            )
        return "\n".join(lines)

    def _label(self, bucket: int) -> str:
        if self.bucketing is Bucketing.HOUR_OF_DAY:
            return f"{bucket:02d}:00 UTC"
        if self.bucketing is Bucketing.HOUR_OF_WEEK:
            day = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][bucket // 24]
            return f"{day} {bucket % 24:02d}:00"
        return _SESSIONS.get(bucket, ("?", None))[0]


def fit_session_profile(
    returns: pd.Series,
    config: SessionConfig | None = None,
) -> SessionProfile:
    """Estimate conditional drift per bucket, with empirical-Bayes shrinkage.

    The shrinkage factor for bucket ``b`` is

        lambda_b = tau^2 / (tau^2 + sigma_b^2 / n_b)

    where ``tau^2`` is the estimated variance of true effects *between* buckets
    and ``sigma_b^2 / n_b`` is the sampling variance of that bucket's mean. When
    a bucket's estimate is noisy relative to the spread of real effects, lambda
    goes to zero and the bucket contributes nothing.

    ``tau^2`` is estimated by subtracting the average within-bucket sampling
    variance from the observed variance of the bucket means. If that difference
    is negative -- meaning the observed spread is entirely explicable as noise --
    ``tau^2`` is floored at zero and **every** bucket shrinks fully to the grand
    mean. That is the correct answer to "there is no session effect here", and
    it is the outcome this estimator is built to be capable of reaching.

    Args:
        returns: Per-bar log returns indexed by UTC timestamp.
        config: Estimation controls.

    Returns:
        A ``SessionProfile``.

    Raises:
        TypeError: If the index is not a ``DatetimeIndex``, which would make
            bucketing silently meaningless.
    """
    cfg = config or SessionConfig()

    if not isinstance(returns.index, pd.DatetimeIndex):
        raise TypeError("returns must be indexed by a DatetimeIndex")

    clean = returns.dropna()
    n_buckets = _n_buckets(cfg.bucketing)

    raw = np.zeros(n_buckets)
    counts = np.zeros(n_buckets)
    variances = np.zeros(n_buckets)

    if clean.empty:
        return SessionProfile(
            bucketing=cfg.bucketing, raw_means=raw, shrunk_means=raw.copy(),
            counts=counts, shrinkage=np.zeros(n_buckets), scale=0.0, config=cfg,
        )

    buckets = np.array([bucket_of(t, cfg.bucketing) for t in clean.index])
    values = clean.to_numpy(dtype=float)

    for b in range(n_buckets):
        sample = values[buckets == b]
        counts[b] = sample.size
        if sample.size >= 2:
            raw[b] = float(sample.mean())
            variances[b] = float(sample.var(ddof=1))

    grand_mean = float(values.mean())

    # Buckets too thin to say anything are excluded from the tau^2 estimate --
    # including them would inflate the apparent spread of true effects with
    # pure sampling noise.
    usable = counts >= cfg.min_observations
    if usable.sum() < 2:
        shrunk = np.full(n_buckets, grand_mean)
        return SessionProfile(
            bucketing=cfg.bucketing, raw_means=raw, shrunk_means=shrunk,
            counts=counts, shrinkage=np.zeros(n_buckets), scale=0.0, config=cfg,
        )

    sampling_var = np.where(counts > 0, variances / np.maximum(counts, 1), np.inf)
    observed_spread = float(np.var(raw[usable], ddof=1))
    mean_sampling_var = float(np.mean(sampling_var[usable]))

    # tau^2 floored at zero: if the observed spread of bucket means is no larger
    # than sampling noise alone would produce, there is no effect to estimate.
    tau_squared = max(0.0, observed_spread - mean_sampling_var)

    shrinkage = np.zeros(n_buckets)
    shrunk = np.full(n_buckets, grand_mean)
    for b in range(n_buckets):
        if not usable[b] or tau_squared <= 0:
            continue
        lam = tau_squared / (tau_squared + sampling_var[b])
        shrinkage[b] = lam
        shrunk[b] = lam * raw[b] + (1.0 - lam) * grand_mean

    # Centre the profile: we want the *relative* tilt between clock buckets, not
    # the asset's unconditional drift, which momentum already trades.
    shrunk = shrunk - float(shrunk.mean())
    scale = float(np.std(shrunk)) if np.any(shrinkage > 0) else 0.0

    return SessionProfile(
        bucketing=cfg.bucketing, raw_means=raw, shrunk_means=shrunk,
        counts=counts, shrinkage=shrinkage, scale=scale, config=cfg,
    )
