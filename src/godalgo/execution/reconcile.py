"""Position reconciliation.

The bot's belief about its position and the venue's record of it will diverge.
Not might -- will. Missed fills, partial fills, liquidations, manual
intervention, and dropped WebSocket frames all cause it, and none of them
announce themselves.

A bot trading on a drifted position is trading blind: it sizes against a
position it does not have, and the error compounds with every subsequent order.
So the venue is always the source of truth, reconciliation runs on a schedule
rather than on suspicion, and a drift beyond tolerance halts trading instead of
being silently corrected.

Silent auto-correction is the tempting design and the wrong one. Drift means an
assumption broke; papering over it lets the same break recur unobserved.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime

from godalgo.execution.broker import Broker
from godalgo.execution.types import Position

__all__ = ["Reconciler", "ReconciliationResult"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ReconciliationResult:
    """Comparison of believed position against venue truth."""

    symbol: str
    believed_quantity: float
    actual_quantity: float
    equity: float
    timestamp: datetime
    tolerance: float

    @property
    def drift(self) -> float:
        return self.actual_quantity - self.believed_quantity

    @property
    def within_tolerance(self) -> bool:
        return abs(self.drift) <= self.tolerance

    def describe(self) -> str:
        return (
            f"{self.symbol}: believed {self.believed_quantity:+.8f}, "
            f"venue {self.actual_quantity:+.8f}, drift {self.drift:+.8f} "
            f"({'ok' if self.within_tolerance else 'OUT OF TOLERANCE'})"
        )


@dataclass
class Reconciler:
    """Checks believed positions against the venue.

    Args:
        broker: Source of truth.
        absolute_tolerance: Base-currency drift treated as noise. Exists because
            exchanges round quantities and a tolerance of exactly zero would
            halt on rounding dust.
        relative_tolerance: Additional tolerance as a fraction of the believed
            position, for venues whose rounding scales with size.
    """

    broker: Broker
    absolute_tolerance: float = 1e-8
    relative_tolerance: float = 0.001

    async def check(self, symbol: str, believed_quantity: float) -> ReconciliationResult:
        """Compare a believed position with the venue's."""
        position: Position = await self.broker.position(symbol)
        equity = await self.broker.equity()

        tolerance = self.absolute_tolerance + self.relative_tolerance * abs(believed_quantity)
        result = ReconciliationResult(
            symbol=symbol,
            believed_quantity=believed_quantity,
            actual_quantity=position.quantity,
            equity=equity,
            timestamp=datetime.now(UTC),
            tolerance=tolerance,
        )

        if not result.within_tolerance:
            logger.error("position drift detected -- %s", result.describe())
        else:
            logger.debug("reconciled %s", result.describe())
        return result

    async def check_all(self, believed: dict[str, float]) -> list[ReconciliationResult]:
        """Reconcile several symbols at once."""
        return [await self.check(symbol, qty) for symbol, qty in believed.items()]
