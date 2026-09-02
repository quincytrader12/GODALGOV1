"""Alpaca as a ``Broker``.

The engine never learns which broker it is holding, so everything specific to
this venue has to be absorbed here. Three things do not map cleanly, and each
is handled explicitly rather than papered over, because a broker that quietly
reinterprets an order is worse than one that refuses it.

**There is no post-only.** This project defaults to post-only precisely because
the maker/taker difference usually exceeds the edge a fast signal is chasing.
Alpaca has no such flag, so the guarantee is not available: a post-only order
becomes a plain limit order. On equities that costs nothing -- they are
commission-free, so there is no maker tier to lose. On crypto it is real, and
the cost model must assume the taker side rather than hoping. Silently
downgrading it while the router still prices maker fees would make every
crypto backtest optimistic in a way nothing on screen would reveal.

**There is no reduce-only.** The flag exists so an exit racing a signal flip
cannot overshoot through flat and open the opposite position. Without venue
support the same guarantee is enforced here, by clamping the quantity to the
position the venue currently reports. Sourced from the venue, never from what
we think we hold -- the whole reason ``reduce_only`` exists is that those two
disagree.

**Crypto positions come back without the slash.** ``BTC/USD`` is held as
``BTCUSD``. Matching on the raw string silently reports every crypto position
as flat, which reads as "the bot has no position" while it does.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from godalgo.execution.broker import Broker
from godalgo.execution.types import (
    Order,
    OrderResult,
    OrderStatus,
    OrderType,
    Position,
    TimeInForce,
)
from godalgo.venues.alpaca import AlpacaClient, AlpacaError, is_crypto

__all__ = ["AlpacaBroker"]

logger = logging.getLogger(__name__)

# Alpaca's vocabulary is wider than ours, and several of its states mean the
# same thing to a trading loop. Anything unlisted is deliberately UNKNOWN
# rather than assumed dead: an order whose fate we cannot establish must not be
# treated as "did not happen", which is how a position silently doubles.
_STATUS = {
    "new": OrderStatus.OPEN,
    "accepted": OrderStatus.OPEN,
    "pending_new": OrderStatus.PENDING,
    "accepted_for_bidding": OrderStatus.OPEN,
    "partially_filled": OrderStatus.PARTIALLY_FILLED,
    "filled": OrderStatus.FILLED,
    "done_for_day": OrderStatus.CANCELED,
    "canceled": OrderStatus.CANCELED,
    "expired": OrderStatus.EXPIRED,
    "rejected": OrderStatus.REJECTED,
    "suspended": OrderStatus.CANCELED,
    "stopped": OrderStatus.CANCELED,
}


def _normalise(symbol: str) -> str:
    """A comparable form for a symbol the venue may spell either way."""
    return symbol.replace("/", "").replace("-", "").upper()


def _time_in_force(order: Order) -> str:
    """Our time-in-force, in Alpaca's vocabulary.

    Crypto accepts only ``gtc`` and ``ioc``, so anything else has to land on
    one of those. Post-only becomes a resting order rather than an aggressive
    one, which is the closest honest reading of the intent even though the
    maker guarantee is gone.
    """
    crypto = is_crypto(order.symbol)
    if order.time_in_force is TimeInForce.IOC:
        return "ioc"
    if order.time_in_force is TimeInForce.FOK:
        return "ioc" if crypto else "fok"
    if order.time_in_force is TimeInForce.GTC:
        return "gtc"
    # POST_ONLY. Rests on the book; the fee tier is not guaranteed.
    return "gtc" if crypto else "day"


@dataclass
class AlpacaBroker(Broker):
    """Orders and positions at Alpaca.

    Args:
        client: A configured client. Paper or live is its decision, not this
            class's -- there is exactly one place that choice is made.
    """

    client: AlpacaClient
    _warned_post_only: bool = field(default=False, init=False, repr=False)

    async def submit(self, order: Order) -> OrderResult:
        """Send an order. A rejection is a result; only ambiguity raises."""
        amount = order.amount
        if order.reduce_only:
            amount = await self._clamped_to_position(order)
            if amount <= 0:
                # Nothing to reduce. Not an error, and not an order: sending it
                # would open a position in the opposite direction, which is the
                # exact outcome reduce_only exists to prevent.
                return OrderResult(
                    order=order, status=OrderStatus.CANCELED,
                    error="reduce-only with no position to reduce",
                )

        if order.time_in_force is TimeInForce.POST_ONLY and not self._warned_post_only:
            self._warned_post_only = True
            logger.warning(
                "Alpaca has no post-only; resting limit order instead. On "
                "crypto the maker fee tier is not guaranteed."
            )

        payload: dict[str, Any] = {
            "symbol": order.symbol,
            "qty": f"{amount:.9f}".rstrip("0").rstrip("."),
            "side": order.side.value,
            "type": order.order_type.value,
            "time_in_force": _time_in_force(order),
            "client_order_id": order.client_order_id,
        }
        if order.order_type is OrderType.LIMIT and order.price is not None:
            payload["limit_price"] = f"{order.price:.6f}"

        try:
            raw = await self.client.submit_order(payload)
        except AlpacaError as exc:
            if exc.status == 0:
                # Transport failure. The order may or may not have reached the
                # venue, and saying "rejected" here would be a guess with a
                # position on the other side of it.
                return OrderResult(
                    order=order, status=OrderStatus.UNKNOWN, error=exc.message,
                )
            return OrderResult(
                order=order, status=OrderStatus.REJECTED, error=exc.message,
            )
        return self._result(order, raw)

    async def cancel(self, order_id: str, symbol: str) -> bool:
        try:
            await self.client.cancel_order(order_id)
        except AlpacaError as exc:
            # 404 means it is already gone, which is the state we asked for.
            if exc.status == 404:
                return True
            logger.warning("could not cancel %s: %s", order_id, exc.message)
            return False
        return True

    async def open_orders(self, symbol: str | None = None) -> list[OrderResult]:
        try:
            raw = await self.client.orders("open")
        except AlpacaError:
            logger.exception("could not list open orders")
            return []

        wanted = _normalise(symbol) if symbol else None
        out: list[OrderResult] = []
        for row in raw:
            if wanted and _normalise(str(row.get("symbol", ""))) != wanted:
                continue
            out.append(self._result(_order_from(row), row))
        return out

    async def position(self, symbol: str) -> Position:
        """The position the venue reports, or flat.

        Every position is listed and matched on a normalised symbol rather than
        asking for one by name: crypto is held as ``BTCUSD`` while we trade
        ``BTC/USD``, and a direct lookup on the traded spelling returns 404 --
        indistinguishable from flat, and wrong in the most expensive direction.
        """
        try:
            rows = await self.client.positions()
        except AlpacaError:
            logger.exception("could not read positions")
            return Position(symbol=symbol, quantity=0.0)

        wanted = _normalise(symbol)
        for row in rows:
            if _normalise(str(row.get("symbol", ""))) != wanted:
                continue
            return Position(
                symbol=symbol,
                quantity=float(row.get("qty") or 0.0),
                entry_price=_float_or_none(row.get("avg_entry_price")),
                mark_price=_float_or_none(row.get("current_price")),
                unrealised_pnl=float(row.get("unrealized_pl") or 0.0),
            )
        return Position(symbol=symbol, quantity=0.0)

    async def equity(self) -> float:
        account = await self.client.account()
        return float(account.get("equity") or account.get("portfolio_value") or 0.0)

    # -- internals ---------------------------------------------------------

    async def _clamped_to_position(self, order: Order) -> float:
        """How much of this reduce-only order the position can actually absorb."""
        held = await self.position(order.symbol)
        if held.is_flat:
            return 0.0
        # Only an order opposing the position reduces it.
        opposing = (held.quantity > 0) != (order.signed_amount > 0)
        if not opposing:
            return 0.0
        return min(order.amount, abs(held.quantity))

    def _result(self, order: Order, raw: dict[str, Any]) -> OrderResult:
        filled = float(raw.get("filled_qty") or 0.0)
        return OrderResult(
            order=order,
            status=_STATUS.get(str(raw.get("status", "")), OrderStatus.UNKNOWN),
            exchange_order_id=raw.get("id"),
            filled_amount=filled,
            average_price=_float_or_none(raw.get("filled_avg_price")),
            # Equities are commission-free and Alpaca does not return a fee on
            # the order. Crypto fees are charged but are not on this object
            # either; the router's cost model carries them instead of a zero
            # here being mistaken for "this trade was free".
            fee=0.0,
            submitted_at=_time(raw.get("submitted_at")),
            updated_at=_time(raw.get("updated_at")) or datetime.now(UTC),
        )


def _order_from(raw: dict[str, Any]) -> Order:
    """Rebuild the request from what the venue reports.

    Open orders can outlive the process that sent them -- a restart, a crash,
    a second copy -- so they must be describable without the original object.
    """
    from godalgo.execution.types import OrderSide

    price = _float_or_none(raw.get("limit_price"))
    order_type = OrderType.LIMIT if price else OrderType.MARKET
    return Order(
        symbol=str(raw.get("symbol", "")),
        side=OrderSide.BUY if raw.get("side") == "buy" else OrderSide.SELL,
        amount=float(raw.get("qty") or 0.0) or 1e-9,
        order_type=order_type,
        price=price,
        time_in_force=TimeInForce.IOC if order_type is OrderType.MARKET
        else TimeInForce.GTC,
        client_order_id=str(raw.get("client_order_id") or "unknown"),
    )


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _time(value: Any) -> datetime | None:
    from godalgo.venues.alpaca import _parse_time

    return _parse_time(value)
