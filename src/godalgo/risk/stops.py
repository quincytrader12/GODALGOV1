"""Per-position stop management: initial stops, trailing, and break-even.

The risk layer already bounds the *account* -- drawdown, daily loss, gross
exposure. None of that bounds a single trade. A position can run against you
indefinitely inside limits that only trip once the damage is account-sized,
which is far too late to be called risk management.

This module bounds the trade. Three mechanisms, each answering a different
question:

* **Initial stop** -- "how much am I willing to lose being wrong about this?"
  Placed at entry, sized in volatility units rather than currency, because a
  fixed-percentage stop is arbitrarily tight on a volatile instrument and
  arbitrarily loose on a quiet one. ATR is the natural unit.

* **Break-even move** -- "when does this trade stop being able to hurt me?"
  Once price has travelled far enough in your favour, the stop moves to entry
  (plus costs). This is what converts an open risk into a free option, and it
  is the single highest-value rule in the module.

* **Trailing stop** -- "how much of an open gain am I willing to give back?"
  Follows price at a volatility-scaled distance, ratcheting only in the
  favourable direction.

The ratchet is the invariant that matters: **a stop never moves against the
position.** Widening a stop because price is approaching it is the mechanism by
which a small controlled loss becomes an account-ending one, and it is the most
common way a disciplined system is talked out of its own rules. The code makes
it structurally impossible rather than merely discouraged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum

import numpy as np

__all__ = ["ExitReason", "StopConfig", "StopManager", "StopState"]

logger = logging.getLogger(__name__)


class ExitReason(str, Enum):
    """Why a stop fired. Recorded so the journal can distinguish exit types."""

    INITIAL_STOP = "initial_stop"
    TRAILING_STOP = "trailing_stop"
    BREAK_EVEN = "break_even"
    TAKE_PROFIT = "take_profit"
    TIME_STOP = "time_stop"


@dataclass(frozen=True, slots=True)
class StopConfig:
    """Stop distances, all in ATR multiples rather than percentages.

    Volatility units travel: the same configuration behaves sensibly on a quiet
    pair and a violent one, where a fixed percentage would be far too tight on
    one and far too loose on the other.
    """

    initial_atr: float = 2.0
    """Initial stop distance in ATR multiples."""

    trail_atr: float = 2.5
    """Trailing distance. Wider than the initial stop on purpose -- a trail
    tight enough to protect every tick is tight enough to be hit by noise on
    the way to a winner."""

    breakeven_trigger_atr: float = 1.5
    """Favourable travel, in ATR, before the stop moves to break-even."""

    breakeven_offset_atr: float = 0.15
    """How far beyond entry break-even sits.

    Not zero. A stop exactly at entry exits at a small loss once fees and
    slippage are counted, which is not what "break even" means.
    """

    trail_activate_atr: float = 2.0
    """Favourable travel before trailing begins.

    Trailing from entry converts winners into scratches: the stop follows price
    up through the noise band and is taken out by the first pullback.
    """

    take_profit_atr: float | None = None
    """Optional hard target. ``None`` lets winners run, which is usually right
    for a trend book and usually wrong for a reversion one."""

    max_bars: int | None = None
    """Optional time stop. A thesis that has not worked within its expected
    holding period is usually wrong rather than early."""

    def __post_init__(self) -> None:
        for name in ("initial_atr", "trail_atr", "breakeven_trigger_atr", "trail_activate_atr"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.breakeven_offset_atr < 0:
            raise ValueError("breakeven_offset_atr must be non-negative")
        if self.take_profit_atr is not None and self.take_profit_atr <= self.initial_atr:
            raise ValueError(
                "take_profit_atr must exceed initial_atr, or every trade closes "
                "for less than it risks"
            )


@dataclass
class StopState:
    """Live stop state for one position."""

    symbol: str
    side: str
    """``long`` or ``short``."""

    entry_price: float
    atr: float
    stop_price: float
    opened_at: datetime
    bars_held: int = 0
    best_price: float = 0.0
    """Most favourable price seen. The ratchet's high-water mark."""

    at_breakeven: bool = False
    trailing: bool = False
    take_profit_price: float | None = None

    @property
    def direction(self) -> float:
        return 1.0 if self.side == "long" else -1.0

    def favourable_travel(self, price: float) -> float:
        """Distance moved in the position's favour, in ATR units."""
        if self.atr <= 0:
            return 0.0
        return self.direction * (price - self.entry_price) / self.atr

    def risk_atr(self) -> float:
        """Current distance to the stop, in ATR. Negative once past break-even."""
        if self.atr <= 0:
            return 0.0
        return self.direction * (self.entry_price - self.stop_price) / self.atr

    def to_dict(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "entry_price": round(self.entry_price, 8),
            "stop_price": round(self.stop_price, 8),
            "atr": round(self.atr, 8),
            "bars_held": self.bars_held,
            "at_breakeven": self.at_breakeven,
            "trailing": self.trailing,
            "take_profit": self.take_profit_price,
        }


@dataclass
class StopManager:
    """Tracks and advances stops for every open position."""

    config: StopConfig = field(default_factory=StopConfig)
    positions: dict[str, StopState] = field(default_factory=dict)

    def open(
        self, symbol: str, side: str, entry_price: float, atr: float,
        *, moment: datetime | None = None,
    ) -> StopState:
        """Place the initial stop for a new position.

        Raises:
            ValueError: If ATR is non-positive. A stop cannot be placed in
                volatility units without a volatility estimate, and defaulting
                to some percentage would silently substitute a different risk
                model for the one configured.
        """
        if atr <= 0 or not np.isfinite(atr):
            raise ValueError(f"cannot place a stop without a valid ATR (got {atr})")
        if entry_price <= 0:
            raise ValueError(f"entry price must be positive (got {entry_price})")

        direction = 1.0 if side == "long" else -1.0
        stop = entry_price - direction * self.config.initial_atr * atr

        take_profit = None
        if self.config.take_profit_atr is not None:
            take_profit = entry_price + direction * self.config.take_profit_atr * atr

        state = StopState(
            symbol=symbol, side=side, entry_price=entry_price, atr=atr,
            stop_price=stop, opened_at=moment or datetime.now(UTC),
            best_price=entry_price, take_profit_price=take_profit,
        )
        self.positions[symbol] = state
        logger.info(
            "stop placed %s %s entry %.8f stop %.8f (%.1f ATR)",
            symbol, side, entry_price, stop, self.config.initial_atr,
        )
        return state

    def update(self, symbol: str, price: float, atr: float | None = None) -> ExitReason | None:
        """Advance the stop and report whether the position should exit.

        Returns:
            The reason to exit, or ``None`` to stay in.
        """
        state = self.positions.get(symbol)
        if state is None or price <= 0:
            return None

        state.bars_held += 1
        if atr is not None and atr > 0 and np.isfinite(atr):
            state.atr = atr

        direction = state.direction

        # High-water mark in the favourable direction.
        if direction * (price - state.best_price) > 0:
            state.best_price = price

        travel = state.favourable_travel(price)

        # Break-even: move the stop to entry plus a cost buffer once the trade
        # has proved itself. From here the position cannot lose money.
        if not state.at_breakeven and travel >= self.config.breakeven_trigger_atr:
            breakeven = state.entry_price + direction * self.config.breakeven_offset_atr * state.atr
            if self._is_improvement(state, breakeven):
                state.stop_price = breakeven
                state.at_breakeven = True
                logger.info("%s stop moved to break-even at %.8f", symbol, breakeven)

        # Trailing, once the trade is clear of the noise band around entry.
        if travel >= self.config.trail_activate_atr:
            state.trailing = True
        if state.trailing:
            trail = state.best_price - direction * self.config.trail_atr * state.atr
            if self._is_improvement(state, trail):
                state.stop_price = trail

        # --- exits ---
        if direction * (price - state.stop_price) <= 0:
            if state.trailing:
                return ExitReason.TRAILING_STOP
            return ExitReason.BREAK_EVEN if state.at_breakeven else ExitReason.INITIAL_STOP

        if (
            state.take_profit_price is not None
            and direction * (price - state.take_profit_price) >= 0
        ):
            return ExitReason.TAKE_PROFIT

        if self.config.max_bars is not None and state.bars_held >= self.config.max_bars:
            return ExitReason.TIME_STOP

        return None

    @staticmethod
    def _is_improvement(state: StopState, candidate: float) -> bool:
        """Whether a proposed stop is strictly better for the position.

        The ratchet. A stop only ever moves toward price, never away from it.
        Loosening a stop as price approaches is how a bounded loss becomes an
        unbounded one, so it is made impossible here rather than left to
        discipline elsewhere.
        """
        return state.direction * (candidate - state.stop_price) > 0

    def close(self, symbol: str) -> StopState | None:
        """Drop tracking for a closed position."""
        return self.positions.pop(symbol, None)

    def stop_for(self, symbol: str) -> float | None:
        state = self.positions.get(symbol)
        return state.stop_price if state else None

    def risk_fraction(self, symbol: str, equity: float) -> float:
        """Fraction of equity currently at risk on one position.

        Measured to the stop, not to zero. What a position can actually lose is
        the distance to its stop, and sizing against notional instead
        systematically under-deploys capital on tight stops and over-deploys on
        wide ones.
        """
        state = self.positions.get(symbol)
        if state is None or equity <= 0:
            return 0.0
        return abs(state.entry_price - state.stop_price) / state.entry_price

    def snapshot(self) -> list[dict[str, object]]:
        return [s.to_dict() for s in self.positions.values()]
