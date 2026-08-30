"""Broker interface and paper implementation.

One interface, three modes. The engine never learns which broker it holds, so
the code path exercised in dry-run and paper is the same one that runs live --
the only difference is where orders end up. A separate "live mode" code path
would mean the thing you tested is not the thing you shipped.

The paper broker models the mechanics that actually decide whether a
high-frequency strategy makes money:

* **Post-only orders rest.** They do not fill on submission. They fill when the
  market trades through their price. A paper broker that fills limit orders
  instantly at the limit price reports maker fees on taker behaviour, and will
  make any fast strategy look profitable.
* **Queue priority is not free.** Resting at the touch does not guarantee a fill
  when price merely touches your level -- you are behind existing size. The
  model requires price to trade *through* the level.
* **Fees are charged by liquidity type.** Maker and taker are different numbers,
  and at high frequency their difference is often larger than the signal.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime

from godalgo.execution.types import (
    Order,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    TimeInForce,
)

__all__ = ["BookSnapshot", "Broker", "DryRunBroker", "FeeSchedule", "PaperBroker"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FeeSchedule:
    """Maker/taker fees as fractions of notional.

    Defaults approximate a retail crypto tier. The maker/taker gap is the number
    that decides whether a high-frequency strategy is viable: crossing the
    spread on every trade costs roughly twice the maker rate per round trip,
    before any slippage.
    """

    maker: float = 0.0002
    taker: float = 0.0006

    def rate_for(self, is_maker: bool) -> float:
        return self.maker if is_maker else self.taker


@dataclass(frozen=True, slots=True)
class BookSnapshot:
    """Top of book at a point in time."""

    symbol: str
    bid: float
    ask: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    bid_size: float = 0.0
    ask_size: float = 0.0

    def __post_init__(self) -> None:
        if self.bid <= 0 or self.ask <= 0:
            raise ValueError(f"non-positive book prices: bid={self.bid} ask={self.ask}")
        if self.ask < self.bid:
            raise ValueError(f"crossed book: bid={self.bid} > ask={self.ask}")

    @property
    def mid(self) -> float:
        return 0.5 * (self.bid + self.ask)

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def spread_bps(self) -> float:
        """Spread in basis points of mid.

        The single most useful number for deciding whether a fast strategy can
        trade at all: a round trip costs at least this much, before fees.
        """
        return 1e4 * self.spread / self.mid

    def touch_for(self, side: OrderSide) -> float:
        """The price a taker would pay/receive."""
        return self.ask if side is OrderSide.BUY else self.bid

    def passive_for(self, side: OrderSide) -> float:
        """The price a maker would post at without crossing."""
        return self.bid if side is OrderSide.BUY else self.ask


class Broker(ABC):
    """What the engine is allowed to ask of a venue."""

    @abstractmethod
    async def submit(self, order: Order) -> OrderResult:
        """Send an order.

        Must never raise on a rejected order -- a rejection is a result, not an
        exception. Raise only when the outcome could not be established, and
        return ``OrderStatus.UNKNOWN`` where the order may or may not be live.
        """

    @abstractmethod
    async def cancel(self, order_id: str, symbol: str) -> bool:
        """Cancel a resting order. Returns whether it is now gone."""

    @abstractmethod
    async def open_orders(self, symbol: str | None = None) -> list[OrderResult]:
        """Orders currently live at the venue."""

    @abstractmethod
    async def position(self, symbol: str) -> Position:
        """Current position **as the venue reports it**, never as we infer it."""

    @abstractmethod
    async def equity(self) -> float:
        """Account equity in quote currency."""

    async def cancel_all(self, symbol: str | None = None) -> int:
        """Cancel every resting order. Returns how many were cancelled.

        Part of the interface because the kill switch depends on it: flattening
        while stale orders rest is how a halted bot re-enters the market.
        """
        cancelled = 0
        for result in await self.open_orders(symbol):
            if result.exchange_order_id and await self.cancel(
                result.exchange_order_id, result.order.symbol
            ):
                cancelled += 1
        return cancelled


class DryRunBroker(Broker):
    """Logs orders and accepts none of them.

    The default. Everything upstream runs exactly as it would live -- signals,
    risk checks, order construction, sizing -- and nothing reaches a venue.
    """

    def __init__(self, starting_equity: float = 10_000.0) -> None:
        self._equity = starting_equity
        self.submitted: list[Order] = []

    async def submit(self, order: Order) -> OrderResult:
        self.submitted.append(order)
        logger.info(
            "DRY RUN would submit %s %s %.8f %s @ %s",
            order.side.value, order.symbol, order.amount,
            order.order_type.value, order.price,
        )
        return OrderResult(
            order=order,
            status=OrderStatus.REJECTED,
            error="dry run: order not sent",
            submitted_at=datetime.now(UTC),
        )

    async def cancel(self, order_id: str, symbol: str) -> bool:
        return True

    async def open_orders(self, symbol: str | None = None) -> list[OrderResult]:
        return []

    async def position(self, symbol: str) -> Position:
        return Position(symbol=symbol, quantity=0.0)

    async def equity(self) -> float:
        return self._equity


class PaperBroker(Broker):
    """Simulated fills against real book data.

    Call ``on_book`` with each market update; resting orders are matched against
    it. Fill logic is deliberately conservative -- when in doubt it does not
    fill, because an optimistic paper broker is worse than none at all.
    """

    def __init__(
        self,
        starting_equity: float = 10_000.0,
        fees: FeeSchedule | None = None,
        *,
        taker_slippage_bps: float = 1.0,
    ) -> None:
        self._equity = starting_equity
        self.fees = fees or FeeSchedule()
        self.taker_slippage_bps = taker_slippage_bps
        self._positions: dict[str, float] = {}
        self._entry: dict[str, float] = {}
        self._resting: dict[str, OrderResult] = {}
        self._books: dict[str, BookSnapshot] = {}
        self._counter = 0
        self.fills: list[OrderResult] = []
        self.realised_pnl = 0.0
        self.fees_paid = 0.0

    # -- market data ------------------------------------------------------

    def on_book(self, book: BookSnapshot) -> list[OrderResult]:
        """Advance the simulation. Returns orders that filled on this update."""
        self._books[book.symbol] = book
        filled: list[OrderResult] = []

        for oid, resting in list(self._resting.items()):
            if resting.order.symbol != book.symbol:
                continue
            if self._crosses(resting.order, book):
                result = self._fill(resting.order, resting.order.price, is_maker=True, oid=oid)
                filled.append(result)
                del self._resting[oid]

        return filled

    @staticmethod
    def _crosses(order: Order, book: BookSnapshot) -> bool:
        """Has the market traded *through* a resting order's price?

        Strict inequality is the conservative choice, and it is the honest one.
        Price merely touching your level does not fill you -- you are behind
        whatever size was already queued there. Using ``<=`` would inflate fill
        rates for exactly the passive strategies that depend on them most.
        """
        if order.price is None:
            return False
        if order.side is OrderSide.BUY:
            return book.ask < order.price
        return book.bid > order.price

    # -- broker interface -------------------------------------------------

    async def submit(self, order: Order) -> OrderResult:
        book = self._books.get(order.symbol)
        if book is None:
            return OrderResult(
                order=order,
                status=OrderStatus.REJECTED,
                error="no market data for symbol",
                submitted_at=datetime.now(UTC),
            )

        self._counter += 1
        oid = f"paper-{self._counter}"

        if order.order_type is OrderType.MARKET or order.time_in_force in {
            TimeInForce.IOC, TimeInForce.FOK
        }:
            price = book.touch_for(order.side)
            slip = price * self.taker_slippage_bps / 1e4
            price += slip if order.side is OrderSide.BUY else -slip
            return self._fill(order, price, is_maker=False, oid=oid)

        if order.time_in_force is TimeInForce.POST_ONLY and self._would_cross(order, book):
            # The venue rejects rather than fills. Modelling it as a fill would
            # silently convert maker strategies into taker ones.
            return OrderResult(
                order=order,
                status=OrderStatus.REJECTED,
                error="post-only would cross the spread",
                submitted_at=datetime.now(UTC),
            )

        result = OrderResult(
            order=order,
            status=OrderStatus.OPEN,
            exchange_order_id=oid,
            submitted_at=datetime.now(UTC),
        )
        self._resting[oid] = result
        return result

    @staticmethod
    def _would_cross(order: Order, book: BookSnapshot) -> bool:
        if order.price is None:
            return True
        if order.side is OrderSide.BUY:
            return order.price >= book.ask
        return order.price <= book.bid

    def _fill(self, order: Order, price: float, *, is_maker: bool, oid: str) -> OrderResult:
        fee = abs(order.amount * price) * self.fees.rate_for(is_maker)
        signed = order.signed_amount

        symbol = order.symbol
        previous = self._positions.get(symbol, 0.0)
        new_quantity = previous + signed

        # Realise P&L on the portion of the trade that reduces the position.
        if previous != 0.0 and (previous > 0) != (signed > 0):
            closed = min(abs(signed), abs(previous))
            entry = self._entry.get(symbol, price)
            direction = 1.0 if previous > 0 else -1.0
            self.realised_pnl += direction * closed * (price - entry)

        if abs(new_quantity) < 1e-12:
            new_quantity = 0.0
            self._entry.pop(symbol, None)
        elif previous == 0.0 or (previous > 0) == (signed > 0):
            # Opening or adding: blend into a weighted average entry.
            entry = self._entry.get(symbol, price)
            total = abs(previous) + abs(signed)
            self._entry[symbol] = (abs(previous) * entry + abs(signed) * price) / total
        self._positions[symbol] = new_quantity

        self.fees_paid += fee
        self._equity -= fee

        result = OrderResult(
            order=order,
            status=OrderStatus.FILLED,
            exchange_order_id=oid,
            filled_amount=order.amount,
            average_price=price,
            fee=fee,
            submitted_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        self.fills.append(result)
        return result

    async def cancel(self, order_id: str, symbol: str) -> bool:
        return self._resting.pop(order_id, None) is not None

    async def open_orders(self, symbol: str | None = None) -> list[OrderResult]:
        return [
            r for r in self._resting.values()
            if symbol is None or r.order.symbol == symbol
        ]

    async def position(self, symbol: str) -> Position:
        quantity = self._positions.get(symbol, 0.0)
        book = self._books.get(symbol)
        mark = book.mid if book else None
        entry = self._entry.get(symbol)
        unrealised = 0.0
        if mark is not None and entry is not None and quantity != 0.0:
            unrealised = quantity * (mark - entry)
        return Position(
            symbol=symbol,
            quantity=quantity,
            entry_price=entry,
            mark_price=mark,
            unrealised_pnl=unrealised,
        )

    async def equity(self) -> float:
        total = self._equity + self.realised_pnl
        for symbol, quantity in self._positions.items():
            book = self._books.get(symbol)
            entry = self._entry.get(symbol)
            if book and entry and quantity:
                total += quantity * (book.mid - entry)
        return total
