"""Deterministic risk limits.

This module is the answer to the obvious hazard in the design: an LLM-driven
runtime with order authority. Every limit here is plain arithmetic with no model
in the path, and every order is expected to pass through ``RiskManager.apply``
before reaching an exchange.

The rule the rest of the system is built around: **risk limits are not
parameters the self-improvement loop may tune.** They are not in any strategy's
``SPACE``, the optimiser never sees them, and ``RiskManager`` is constructed from
configuration rather than from anything the agent produces. A system that can
relax its own stop-loss does not have a stop-loss.

Limits compose multiplicatively -- each is a separate ceiling and the binding one
wins, so a new limit can only ever reduce exposure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

import numpy as np

__all__ = ["KillSwitchTripped", "RiskDecision", "RiskLimits", "RiskManager"]


class KillSwitchTripped(RuntimeError):
    """Raised when trading is halted and an order is nonetheless attempted."""


@dataclass(frozen=True, slots=True)
class RiskLimits:
    """Hard bounds on exposure. Set from config; never tuned by the optimiser."""

    max_gross_weight: float = 1.0
    """Maximum absolute position as a fraction of equity."""

    max_position_change: float = 0.35
    """Maximum weight change in a single bar.

    Bounds the damage from a single bad signal, and from a data glitch that
    makes one bar look like an enormous opportunity.
    """

    daily_loss_limit: float = 0.04
    """Fractional equity loss within one UTC day that halts trading."""

    max_drawdown_limit: float = 0.20
    """Peak-to-trough drawdown that trips the kill switch permanently."""

    max_consecutive_losses: int = 8
    """Consecutive losing bars that force a de-risk.

    A blunt instrument on purpose. It catches the case where the model is simply
    wrong about the current market in a way no single-trade stop detects.
    """

    consecutive_loss_derisk: float = 0.4
    """Exposure multiplier applied once the consecutive-loss count is hit."""

    def __post_init__(self) -> None:
        if not 0 < self.max_gross_weight <= 10:
            raise ValueError("max_gross_weight must be in (0, 10]")
        if not 0 < self.max_position_change <= 2 * self.max_gross_weight:
            raise ValueError("max_position_change must be positive and reachable")
        for name in ("daily_loss_limit", "max_drawdown_limit"):
            value = getattr(self, name)
            if not 0 < value < 1:
                raise ValueError(f"{name} must be in (0, 1), got {value}")
        if self.max_consecutive_losses < 1:
            raise ValueError("max_consecutive_losses must be >= 1")
        if not 0 <= self.consecutive_loss_derisk <= 1:
            raise ValueError("consecutive_loss_derisk must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class RiskDecision:
    """The outcome of applying limits to one desired weight."""

    weight: float
    """The weight actually permitted."""

    requested: float
    """What the portfolio layer asked for, kept for diagnostics."""

    binding: tuple[str, ...] = ()
    """Names of the limits that actually reduced the request."""

    halted: bool = False
    """True when the kill switch is active and the only legal weight is zero."""

    @property
    def was_reduced(self) -> bool:
        return bool(self.binding)


@dataclass(slots=True)
class RiskManager:
    """Stateful enforcement of ``RiskLimits`` across a trading session.

    Tracks equity high-water mark, per-day starting equity, and the consecutive
    loss count. ``apply`` is the single gate every desired weight passes through.
    """

    limits: RiskLimits = field(default_factory=RiskLimits)
    equity: float = 1.0
    peak_equity: float = 1.0
    day_start_equity: float = 1.0
    current_day: date | None = None
    consecutive_losses: int = 0
    halted: bool = False
    halt_reason: str | None = None
    current_weight: float = 0.0

    def update_equity(self, equity: float, timestamp: datetime) -> None:
        """Record a new equity mark and evaluate the halting conditions.

        Must be called once per bar, before ``apply``, so limits act on current
        state rather than on the previous bar's.
        """
        if not np.isfinite(equity) or equity <= 0:
            self._halt("non-finite or non-positive equity")
            return

        day = timestamp.date()
        if self.current_day is None or day != self.current_day:
            # New UTC day: the daily loss budget resets. Drawdown does not --
            # it is measured against the all-time peak by design, so a slow
            # bleed cannot hide behind daily resets.
            self.current_day = day
            self.day_start_equity = equity

        previous = self.equity
        self.equity = equity
        self.peak_equity = max(self.peak_equity, equity)

        if equity < previous:
            self.consecutive_losses += 1
        elif equity > previous:
            self.consecutive_losses = 0

        drawdown = 1.0 - equity / self.peak_equity
        if drawdown >= self.limits.max_drawdown_limit:
            self._halt(f"max drawdown {drawdown:.2%} >= {self.limits.max_drawdown_limit:.2%}")
            return

        daily_loss = 1.0 - equity / self.day_start_equity
        if daily_loss >= self.limits.daily_loss_limit:
            self._halt(f"daily loss {daily_loss:.2%} >= {self.limits.daily_loss_limit:.2%}")

    def apply(self, desired_weight: float) -> RiskDecision:
        """Clamp a desired weight through every limit in turn.

        Returns:
            A ``RiskDecision`` recording the permitted weight and which limits
            bound it. When halted, the permitted weight is always 0.0 -- the
            kill switch flattens, it does not merely stop adding.
        """
        requested = float(desired_weight) if np.isfinite(desired_weight) else 0.0

        if self.halted:
            self.current_weight = 0.0
            return RiskDecision(
                weight=0.0, requested=requested, binding=("kill_switch",), halted=True
            )

        binding: list[str] = []
        weight = requested

        if self.consecutive_losses >= self.limits.max_consecutive_losses:
            weight *= self.limits.consecutive_loss_derisk
            binding.append("consecutive_losses")

        capped = float(np.clip(weight, -self.limits.max_gross_weight, self.limits.max_gross_weight))
        if capped != weight:
            binding.append("max_gross_weight")
        weight = capped

        change = weight - self.current_weight
        if abs(change) > self.limits.max_position_change:
            weight = self.current_weight + np.sign(change) * self.limits.max_position_change
            binding.append("max_position_change")

        self.current_weight = float(weight)
        return RiskDecision(
            weight=float(weight), requested=requested, binding=tuple(binding), halted=False
        )

    def reset_day(self) -> None:
        """Clear a daily-loss halt. Drawdown halts are deliberately not cleared."""
        if self.halt_reason and self.halt_reason.startswith("daily loss"):
            self.halted = False
            self.halt_reason = None
        self.day_start_equity = self.equity
        self.consecutive_losses = 0

    def _halt(self, reason: str) -> None:
        if not self.halted:
            self.halted = True
            self.halt_reason = reason
