"""Strategy interface.

Strategies emit *conviction*, never position size. Sizing and risk live in the
portfolio and risk layers so that a limit applies identically no matter which
strategy asked for exposure -- a strategy cannot size its way around a cap it
does not control.

Parameters are declared, bounded, and introspectable. That is what lets the
self-improvement loop propose new parameter sets without ever generating code:
the search space is a fixed, validated box rather than an open text field.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, fields
from typing import Any, ClassVar

import pandas as pd

from godalgo.core.types import Regime

__all__ = ["ParamSpec", "Strategy", "StrategyParams"]


@dataclass(frozen=True, slots=True)
class ParamSpec:
    """Bounds for one tunable parameter."""

    low: float
    high: float
    integer: bool = False

    def clip(self, value: float) -> float | int:
        """Clamp into bounds, returning a true ``int`` for integer parameters.

        Returning ``float(round(v))`` here instead would produce values like
        ``36.0`` that pass every validation check and then fail deep inside
        pandas, which rejects a float ``min_periods``. That failure surfaces as
        a skipped candidate rather than an error, so the search would quietly
        collapse to a single configuration while still reporting plausible
        results.
        """
        v = min(max(float(value), self.low), self.high)
        return int(round(v)) if self.integer else v

    def contains(self, value: float) -> bool:
        if not math.isfinite(value):
            return False
        return self.low <= value <= self.high


# NOTE: deliberately not slots=True. Dataclass slots synthesise a *new* class
# object, which invalidates the __class__ cell that zero-argument super() relies
# on -- so every subclass calling super().validate() would fail at runtime.
@dataclass(frozen=True)
class StrategyParams:
    """Base for strategy parameter sets.

    Subclasses declare a ``SPACE`` mapping each field to a ``ParamSpec``. Any
    parameter absent from ``SPACE`` is treated as fixed and is never searched.
    """

    SPACE: ClassVar[dict[str, ParamSpec]] = {}

    def validate(self) -> None:
        """Raise if any searchable parameter is outside its declared bounds."""
        for name, spec in self.SPACE.items():
            value = getattr(self, name)
            if not spec.contains(value):
                raise ValueError(
                    f"{type(self).__name__}.{name}={value!r} outside [{spec.low}, {spec.high}]"
                )

    def replace(self, **changes: float) -> StrategyParams:
        """Return a copy with ``changes`` applied and clipped into bounds."""
        known = {f.name for f in fields(self)}
        unknown = set(changes) - known
        if unknown:
            raise ValueError(f"unknown parameters: {sorted(unknown)}")
        clipped = {
            name: self.SPACE[name].clip(value) if name in self.SPACE else value
            for name, value in changes.items()
        }
        return type(self)(**{**asdict(self), **clipped})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Strategy(ABC):
    """Base strategy.

    Attributes:
        name: Stable identifier used in attribution and the promotion ledger.
        favoured_regime: The regime in which this strategy is expected to have
            an edge. The allocator uses it to decide who gets capital.
    """

    name: ClassVar[str]
    favoured_regime: ClassVar[Regime]

    def __init__(self, params: StrategyParams) -> None:
        params.validate()
        self.params = params

    @abstractmethod
    def generate(self, bars: pd.DataFrame) -> pd.Series:
        """Compute conviction in [-1, 1] for every bar.

        Args:
            bars: OHLCV frame indexed by timestamp, oldest first, with at least
                a ``close`` column.

        Returns:
            Series aligned to ``bars.index``. Positive is long, negative short,
            zero flat. Warm-up bars must be 0.0, not NaN, so that downstream
            arithmetic does not silently propagate nulls into position sizing.

        Implementations must be strictly causal -- the value at bar ``t`` may
        only depend on bars up to and including ``t``.
        """

    @property
    @abstractmethod
    def warmup(self) -> int:
        """Bars required before the strategy emits a non-zero signal.

        The backtester discards this prefix so that the first live-equivalent
        bar is one the strategy could genuinely have traded.
        """

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.params.to_dict()})"
