"""Order routing: turning a target weight into orders worth sending.

The router is where a high-frequency system either works or quietly bleeds out,
and the reason is arithmetic rather than latency.

A round trip costs, at minimum, the spread plus two fees. At a 2bp spread and
6bp taker fees that is **14bp** before slippage. A signal predicting a 5bp move
is profitable in the backtest and loses money on contact with a venue. Speed
does not fix this -- executing a losing trade faster only loses faster.

So every order passes an explicit economic gate: the expected edge of the trade
must exceed its modelled round-trip cost by a margin. Trades that do not clear
it are not sent, and the target weight is left unrealised. Declining to trade is
a valid, and often correct, output.

The router also enforces the mechanics that separate a working bot from one that
gets rate-limited or double-fills: minimum notional, exchange precision,
in-flight deduplication, and a submission throttle.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from godalgo.execution.broker import BookSnapshot, FeeSchedule
from godalgo.execution.types import (
    Order,
    OrderSide,
    OrderType,
    TimeInForce,
)

__all__ = ["OrderRouter", "RoutingConfig", "RoutingDecision"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RoutingConfig:
    """Economic and mechanical constraints on order generation."""

    min_edge_multiple: float = 1.5
    """Required ratio of expected edge to round-trip cost.

    1.0 would mean trading at break-even *in expectation*, which after
    estimation error is a loss. 1.5 demands the signal be half again as large as
    the cost it must overcome. This is the single most important parameter in
    high-frequency operation.
    """

    min_notional: float = 10.0
    """Venue minimum order size in quote currency."""

    min_weight_change: float = 0.02
    """Smallest position change worth an order, as a fraction of equity."""

    max_spread_bps: float = 20.0
    """Refuse to trade when the book is wider than this.

    Wide spreads mean thin liquidity, and thin liquidity is when slippage
    estimates are least reliable. Standing aside is cheaper than discovering
    the real cost by paying it.
    """

    use_post_only: bool = True
    """Post passively rather than crossing.

    The default. The maker/taker difference is usually larger than the edge a
    fast signal is chasing. The cost is fill uncertainty, which the engine
    handles by re-quoting rather than by crossing.
    """

    passive_offset_bps: float = 0.0
    """How far inside the touch to post, in bps. 0 posts at the touch."""

    max_orders_per_minute: int = 60
    """Submission throttle. Exceeding venue limits earns a ban, not an error."""

    amount_precision: int = 8
    price_precision: int = 2

    def __post_init__(self) -> None:
        if self.min_edge_multiple < 1.0:
            raise ValueError(
                "min_edge_multiple below 1.0 sends orders with negative expected value"
            )
        if self.min_notional <= 0 or self.min_weight_change <= 0:
            raise ValueError("min_notional and min_weight_change must be positive")


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    """Whether to trade, and why not when the answer is no."""

    order: Order | None
    reason: str
    expected_edge_bps: float = 0.0
    round_trip_cost_bps: float = 0.0
    target_weight: float = 0.0
    current_weight: float = 0.0

    @property
    def should_trade(self) -> bool:
        return self.order is not None


@dataclass
class OrderRouter:
    """Builds orders from target weights, subject to economics and throttling."""

    config: RoutingConfig = field(default_factory=RoutingConfig)
    fees: FeeSchedule = field(default_factory=FeeSchedule)
    _submission_times: list[float] = field(default_factory=list, init=False)
    _in_flight: dict[str, str] = field(default_factory=dict, init=False)

    def round_trip_cost_bps(self, book: BookSnapshot, *, is_maker: bool) -> float:
        """Modelled cost of entering and exiting, in basis points.

        A maker round trip pays two maker fees and gives up the spread only on
        adverse selection; a taker round trip pays two taker fees *and* crosses
        the spread twice. Charging the full spread to the maker case would
        overstate maker costs and push the gate toward crossing -- exactly the
        wrong direction.
        """
        fee_bps = 2.0 * self.fees.rate_for(is_maker) * 1e4
        spread_cost = 0.0 if is_maker else 2.0 * book.spread_bps
        return fee_bps + spread_cost

    def decide(
        self,
        symbol: str,
        target_weight: float,
        current_weight: float,
        equity: float,
        book: BookSnapshot,
        expected_edge_bps: float,
    ) -> RoutingDecision:
        """Decide whether to trade toward ``target_weight``, and how.

        Args:
            symbol: Market to trade.
            target_weight: Desired position as a fraction of equity, signed.
            current_weight: Present position as a fraction of equity, signed.
            equity: Account equity in quote currency.
            book: Current top of book.
            expected_edge_bps: The strategy's expected move over the intended
                holding period, in bps. This is what the cost gate tests
                against, so an inflated estimate here defeats the gate entirely.

        Returns:
            A ``RoutingDecision``. ``order`` is None when no trade should be
            sent, with ``reason`` naming the binding constraint.
        """
        delta = target_weight - current_weight

        def refuse(reason: str, *, cost: float = 0.0) -> RoutingDecision:
            return RoutingDecision(
                order=None, reason=reason,
                expected_edge_bps=expected_edge_bps, round_trip_cost_bps=cost,
                target_weight=target_weight, current_weight=current_weight,
            )

        if abs(delta) < self.config.min_weight_change:
            return refuse(f"weight change {abs(delta):.4f} below minimum")

        if book.spread_bps > self.config.max_spread_bps:
            return refuse(f"spread {book.spread_bps:.1f}bps exceeds maximum")

        if symbol in self._in_flight:
            # One order per symbol at a time. Stacking orders against the same
            # target is how a single signal becomes a double position.
            return refuse(f"order {self._in_flight[symbol]} already in flight")

        if not self._throttle_allows():
            return refuse("submission throttle reached")

        is_maker = self.config.use_post_only
        cost_bps = self.round_trip_cost_bps(book, is_maker=is_maker)
        required = cost_bps * self.config.min_edge_multiple

        # Exits are exempt. A position being closed for risk reasons must not be
        # held open because closing it is not individually profitable -- that
        # reasoning is how a stop-loss fails to stop anything.
        is_reducing = abs(target_weight) < abs(current_weight) and (
            target_weight == 0.0 or (target_weight > 0) == (current_weight > 0)
        )

        if not is_reducing and abs(expected_edge_bps) < required:
            return refuse(
                f"edge {abs(expected_edge_bps):.1f}bps below required "
                f"{required:.1f}bps (cost {cost_bps:.1f}bps x {self.config.min_edge_multiple})",
                cost=cost_bps,
            )

        side = OrderSide.BUY if delta > 0 else OrderSide.SELL
        price = self._limit_price(book, side)
        amount = self._amount_for(abs(delta), equity, price)

        if amount * price < self.config.min_notional:
            return refuse(
                f"notional {amount * price:.2f} below venue minimum "
                f"{self.config.min_notional:.2f}",
                cost=cost_bps,
            )

        order = Order(
            symbol=symbol,
            side=side,
            amount=amount,
            order_type=OrderType.LIMIT if self.config.use_post_only else OrderType.MARKET,
            price=price if self.config.use_post_only else None,
            time_in_force=TimeInForce.POST_ONLY if self.config.use_post_only else TimeInForce.IOC,
            reduce_only=is_reducing,
        )

        return RoutingDecision(
            order=order,
            reason="ok",
            expected_edge_bps=expected_edge_bps,
            round_trip_cost_bps=cost_bps,
            target_weight=target_weight,
            current_weight=current_weight,
        )

    def _limit_price(self, book: BookSnapshot, side: OrderSide) -> float:
        """Passive price, offset inside the touch if configured."""
        base = book.passive_for(side)
        if self.config.passive_offset_bps:
            offset = base * self.config.passive_offset_bps / 1e4
            base = base - offset if side is OrderSide.BUY else base + offset
        return round(base, self.config.price_precision)

    def _amount_for(self, weight_delta: float, equity: float, price: float) -> float:
        return round(weight_delta * equity / price, self.config.amount_precision)

    def _throttle_allows(self) -> bool:
        now = time.monotonic()
        self._submission_times = [t for t in self._submission_times if now - t < 60.0]
        return len(self._submission_times) < self.config.max_orders_per_minute

    def mark_submitted(self, order: Order) -> None:
        """Record a submission for throttling and in-flight tracking."""
        self._submission_times.append(time.monotonic())
        self._in_flight[order.symbol] = order.client_order_id

    def mark_settled(self, symbol: str) -> None:
        """Clear the in-flight marker once an order reaches a terminal state.

        Must be called on every terminal outcome including rejection, or the
        symbol stays permanently blocked and the bot silently stops trading it.
        """
        self._in_flight.pop(symbol, None)

    @property
    def in_flight_symbols(self) -> set[str]:
        return set(self._in_flight)
