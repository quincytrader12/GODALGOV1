"""Order and execution value types.

The vocabulary shared by the router, the brokers, and the reconciler. Kept
free of behaviour so that a paper fill and a live fill are described by exactly
the same object -- if the two diverged in shape, paper results would stop being
evidence about live behaviour, which is the only reason to run paper at all.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum

__all__ = [
    "Order",
    "OrderResult",
    "OrderSide",
    "OrderStatus",
    "OrderType",
    "Position",
    "TimeInForce",
    "TradingMode",
]


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"


class TimeInForce(str, Enum):
    GTC = "GTC"
    IOC = "IOC"
    """Immediate-or-cancel: fill what you can now, cancel the rest."""

    FOK = "FOK"
    POST_ONLY = "PO"
    """Rejected rather than filled if it would cross the spread.

    The default for anything running at high frequency. It guarantees the maker
    fee tier, and the difference between maker and taker is usually larger than
    the edge a sub-minute signal is trying to capture.
    """


class OrderStatus(str, Enum):
    PENDING = "pending"
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    UNKNOWN = "unknown"
    """Sent, outcome not established.

    Distinct from REJECTED on purpose. A network timeout after transmission
    leaves an order that may or may not be live on the exchange, and treating
    that as "did not happen" is how a position silently doubles.
    """

    @property
    def is_terminal(self) -> bool:
        return self in {
            OrderStatus.FILLED,
            OrderStatus.CANCELED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        }

    @property
    def is_live(self) -> bool:
        return self in {OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED, OrderStatus.PENDING}


class TradingMode(str, Enum):
    """How far an order is allowed to travel.

    Ordered by escalating consequence, and the default is the harmless one.
    Moving to LIVE is an explicit, deliberate act -- never a default, never
    inferred from the presence of API keys.
    """

    DRY_RUN = "dry_run"
    """Compute and log orders; send nothing. The default."""

    PAPER = "paper"
    """Simulate fills against real market data. No exchange contact."""

    LIVE = "live"
    """Real orders, real money."""


def new_client_order_id(prefix: str = "gav1") -> str:
    """Generate a client order id for idempotency.

    Every order carries one so a retry after an ambiguous failure can be matched
    against what the exchange already has, rather than blindly resubmitted. This
    is the difference between a retry and an accidental double position.
    """
    return f"{prefix}-{uuid.uuid4().hex[:20]}"


@dataclass(frozen=True, slots=True)
class Order:
    """An order request. Immutable; outcomes live in ``OrderResult``."""

    symbol: str
    side: OrderSide
    amount: float
    """Base-currency quantity. Always positive; direction is ``side``."""

    order_type: OrderType = OrderType.LIMIT
    price: float | None = None
    time_in_force: TimeInForce = TimeInForce.POST_ONLY
    client_order_id: str = field(default_factory=new_client_order_id)
    reduce_only: bool = False
    """Whether this order may only shrink an existing position.

    Set on every exit. Without it, an exit racing a signal flip can overshoot
    through flat and open the opposite position.
    """

    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        if self.amount <= 0:
            raise ValueError(f"order amount must be positive, got {self.amount}")
        if self.order_type is OrderType.LIMIT and self.price is None:
            raise ValueError("limit orders require a price")
        if self.price is not None and self.price <= 0:
            raise ValueError(f"order price must be positive, got {self.price}")
        if self.order_type is OrderType.MARKET and self.time_in_force is TimeInForce.POST_ONLY:
            raise ValueError("post-only is meaningless for a market order")

    @property
    def signed_amount(self) -> float:
        return self.amount if self.side is OrderSide.BUY else -self.amount


@dataclass(frozen=True, slots=True)
class OrderResult:
    """What actually happened to an order."""

    order: Order
    status: OrderStatus
    exchange_order_id: str | None = None
    filled_amount: float = 0.0
    average_price: float | None = None
    fee: float = 0.0
    submitted_at: datetime | None = None
    updated_at: datetime | None = None
    error: str | None = None

    @property
    def remaining(self) -> float:
        return max(0.0, self.order.amount - self.filled_amount)

    @property
    def signed_filled(self) -> float:
        sign = 1.0 if self.order.side is OrderSide.BUY else -1.0
        return sign * self.filled_amount

    @property
    def notional(self) -> float:
        return self.filled_amount * (self.average_price or 0.0)


@dataclass(frozen=True, slots=True)
class Position:
    """A position as reported by the venue.

    Sourced from the exchange, never inferred from our own order history.
    Local state drifts -- through missed fills, partial fills, liquidations, and
    manual intervention -- and a bot trading on drifted state is trading blind.
    """

    symbol: str
    quantity: float
    """Signed. Positive long, negative short, in base currency."""

    entry_price: float | None = None
    mark_price: float | None = None
    unrealised_pnl: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def is_flat(self) -> bool:
        return abs(self.quantity) < 1e-12

    @property
    def notional(self) -> float:
        return abs(self.quantity) * (self.mark_price or self.entry_price or 0.0)
