"""Shared value types.

Deliberately plain dataclasses: these cross the boundary between deterministic
strategy code and the agent runtime, so they stay trivially serialisable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class Side(str, Enum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class Regime(str, Enum):
    """Which of the two strategy families the market currently favours."""

    TRENDING = "trending"          # momentum edge; Hurst > 0.5, VR > 1
    MEAN_REVERTING = "mean_reverting"  # reversion edge; Hurst < 0.5, VR < 1
    INDETERMINATE = "indeterminate"    # neither test is significant -> de-risk


@dataclass(frozen=True, slots=True)
class Signal:
    """A strategy's desired exposure, before portfolio construction.

    ``strength`` is a unitless conviction in [-1, 1]. It is NOT a position size --
    sizing is the portfolio layer's job, so that risk limits apply uniformly no
    matter which strategy produced the signal.
    """

    timestamp: datetime
    symbol: str
    strength: float
    strategy: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not -1.0 <= self.strength <= 1.0:
            raise ValueError(f"signal strength {self.strength} outside [-1, 1]")

    @property
    def side(self) -> Side:
        if self.strength > 0:
            return Side.LONG
        if self.strength < 0:
            return Side.SHORT
        return Side.FLAT


@dataclass(frozen=True, slots=True)
class RegimeState:
    """Output of the regime classifier for one symbol at one point in time."""

    timestamp: datetime
    symbol: str
    regime: Regime
    hurst: float
    variance_ratio: float
    vr_zscore: float
    half_life: float | None
    adf_pvalue: float | None
    confidence: float
    """In [0, 1]. Scales how much risk the allocator is willing to take."""


@dataclass(frozen=True, slots=True)
class Target:
    """Post-sizing, post-risk desired position as a fraction of equity."""

    timestamp: datetime
    symbol: str
    weight: float
    contributions: dict[str, float] = field(default_factory=dict)
    """Per-strategy weight before blending, kept for attribution."""


@dataclass(slots=True)
class Fill:
    timestamp: datetime
    symbol: str
    weight_delta: float
    price: float
    cost: float
    """Total cost of the trade in equity fraction (fees + modelled slippage)."""
