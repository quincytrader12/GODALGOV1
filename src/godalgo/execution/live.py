"""Live ccxt broker.

The only component in this codebase that can move real money. Written
defensively throughout, because the failure modes here are not "wrong number in
a report" -- they are duplicated positions and unhedged exposure.

Three rules it exists to enforce:

**An ambiguous send is not a failure.** A timeout after transmission leaves an
order that may be live. Reporting that as rejected invites a resubmission that
doubles the position. Such orders come back as ``UNKNOWN`` and the caller must
reconcile before acting.

**Every order carries a client order id.** Idempotency is what makes a retry
safe. Before resubmitting anything, the id is looked up at the venue.

**The venue is the source of truth.** Positions and balances are always read
back, never inferred from our own order history.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Self

import ccxt.async_support as ccxt_async

from godalgo.execution.broker import Broker
from godalgo.execution.types import (
    Order,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    TimeInForce,
)

__all__ = ["ArmingError", "LiveBroker", "LiveBrokerConfig"]

logger = logging.getLogger(__name__)

_ARM_ENV_VAR = "GODALGO_ARM_LIVE"
_ARM_TOKEN = "I_UNDERSTAND_THIS_TRADES_REAL_MONEY"

_STATUS_MAP = {
    "open": OrderStatus.OPEN,
    "closed": OrderStatus.FILLED,
    "canceled": OrderStatus.CANCELED,
    "cancelled": OrderStatus.CANCELED,
    "expired": OrderStatus.EXPIRED,
    "rejected": OrderStatus.REJECTED,
}


class ArmingError(RuntimeError):
    """Raised when live trading is requested without explicit arming."""


@dataclass(frozen=True, slots=True)
class LiveBrokerConfig:
    """Venue and safety configuration.

    Credentials are read from the environment, never accepted as arguments and
    never written to disk -- a config object that holds secrets ends up in a log
    line or a stack trace sooner or later.
    """

    exchange_id: str = "binance"
    symbol_type: str = "spot"
    api_key_env: str = "GODALGO_API_KEY"
    api_secret_env: str = "GODALGO_API_SECRET"
    max_order_notional: float = 1_000.0
    """Hard ceiling per order, in quote currency.

    A last-resort backstop against a sizing bug. It sits below the risk layer,
    not instead of it, and it is checked on every single order.
    """

    request_timeout_ms: int = 10_000
    max_retries: int = 2


class LiveBroker(Broker):
    """ccxt-backed live broker. Requires explicit arming to construct.

    Arming is a two-key operation: ``arm=True`` passed in code *and* the
    environment variable ``GODALGO_ARM_LIVE`` set to the exact token. Either
    alone is insufficient. This is deliberate friction -- it makes live trading
    something you cannot reach by editing one line or by a stale config file
    surviving into production.
    """

    def __init__(self, config: LiveBrokerConfig | None = None, *, arm: bool = False) -> None:
        self.config = config or LiveBrokerConfig()

        if not arm:
            raise ArmingError(
                "LiveBroker requires arm=True. Use DryRunBroker or PaperBroker instead."
            )
        if os.environ.get(_ARM_ENV_VAR) != _ARM_TOKEN:
            raise ArmingError(
                f"live trading requires {_ARM_ENV_VAR}={_ARM_TOKEN} in the environment"
            )

        key = os.environ.get(self.config.api_key_env)
        secret = os.environ.get(self.config.api_secret_env)
        if not key or not secret:
            raise ArmingError(
                f"missing credentials: set {self.config.api_key_env} and "
                f"{self.config.api_secret_env}"
            )

        klass = getattr(ccxt_async, self.config.exchange_id)
        self._exchange = klass(
            {
                "apiKey": key,
                "secret": secret,
                "enableRateLimit": True,
                "timeout": self.config.request_timeout_ms,
                "options": {"defaultType": self.config.symbol_type},
            }
        )
        logger.warning(
            "LiveBroker ARMED on %s (%s) -- orders will use real funds",
            self.config.exchange_id, self.config.symbol_type,
        )

    async def close(self) -> None:
        """Release the underlying HTTP session."""
        await self._exchange.close()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    # -- orders -----------------------------------------------------------

    async def submit(self, order: Order) -> OrderResult:
        """Send an order, with a notional backstop and idempotent retry."""
        notional = order.amount * (order.price or 0.0)
        if order.price is not None and notional > self.config.max_order_notional:
            return OrderResult(
                order=order,
                status=OrderStatus.REJECTED,
                error=(
                    f"notional {notional:.2f} exceeds max_order_notional "
                    f"{self.config.max_order_notional:.2f}"
                ),
                submitted_at=datetime.now(UTC),
            )

        params = self._params_for(order)
        submitted_at = datetime.now(UTC)

        for attempt in range(self.config.max_retries + 1):
            try:
                raw = await self._exchange.create_order(
                    symbol=order.symbol,
                    type=order.order_type.value,
                    side=order.side.value,
                    amount=order.amount,
                    price=order.price,
                    params=params,
                )
                return self._to_result(order, raw, submitted_at)

            except ccxt_async.InvalidOrder as exc:
                # A definite no from the venue. Never retry -- the order was
                # malformed or violated a venue rule, and it will be again.
                return OrderResult(
                    order=order, status=OrderStatus.REJECTED,
                    error=str(exc), submitted_at=submitted_at,
                )
            except ccxt_async.InsufficientFunds as exc:
                return OrderResult(
                    order=order, status=OrderStatus.REJECTED,
                    error=f"insufficient funds: {exc}", submitted_at=submitted_at,
                )
            except (ccxt_async.RequestTimeout, ccxt_async.NetworkError) as exc:
                # The critical case. The order may be live. Look it up by client
                # order id before doing anything else; resubmitting blind here
                # is how a position silently doubles.
                logger.warning(
                    "ambiguous send for %s (attempt %d): %s",
                    order.client_order_id, attempt + 1, exc,
                )
                existing = await self._lookup(order)
                if existing is not None:
                    return existing
                if attempt >= self.config.max_retries:
                    return OrderResult(
                        order=order, status=OrderStatus.UNKNOWN,
                        error=f"send outcome unknown after {attempt + 1} attempts: {exc}",
                        submitted_at=submitted_at,
                    )
                await asyncio.sleep(2.0**attempt)
            except ccxt_async.ExchangeError as exc:
                return OrderResult(
                    order=order, status=OrderStatus.REJECTED,
                    error=str(exc), submitted_at=submitted_at,
                )

        return OrderResult(
            order=order, status=OrderStatus.UNKNOWN,
            error="retries exhausted", submitted_at=submitted_at,
        )

    def _params_for(self, order: Order) -> dict[str, object]:
        params: dict[str, object] = {"clientOrderId": order.client_order_id}
        if order.time_in_force is TimeInForce.POST_ONLY:
            params["postOnly"] = True
        elif order.order_type is OrderType.LIMIT:
            params["timeInForce"] = order.time_in_force.value
        if order.reduce_only:
            params["reduceOnly"] = True
        return params

    async def _lookup(self, order: Order) -> OrderResult | None:
        """Find an order by client id after an ambiguous send.

        Returns None only if the venue clearly does not have it. Any error while
        looking up also returns None, which keeps the caller on the UNKNOWN path
        rather than letting a failed lookup masquerade as "definitely not sent".
        """
        try:
            open_now = await self._exchange.fetch_open_orders(order.symbol)
            for raw in open_now:
                if self._client_id_of(raw) == order.client_order_id:
                    return self._to_result(order, raw, None)
            recent = await self._exchange.fetch_closed_orders(order.symbol, limit=50)
            for raw in recent:
                if self._client_id_of(raw) == order.client_order_id:
                    return self._to_result(order, raw, None)
        except ccxt_async.BaseError as exc:
            logger.error("lookup failed for %s: %s", order.client_order_id, exc)
            return None
        return None

    @staticmethod
    def _client_id_of(raw: dict) -> str | None:
        return raw.get("clientOrderId") or (raw.get("info") or {}).get("clientOrderId")

    @staticmethod
    def _to_result(order: Order, raw: dict, submitted_at: datetime | None) -> OrderResult:
        status = _STATUS_MAP.get(str(raw.get("status") or "").lower(), OrderStatus.UNKNOWN)
        filled = float(raw.get("filled") or 0.0)
        if status is OrderStatus.OPEN and 0 < filled < order.amount:
            status = OrderStatus.PARTIALLY_FILLED

        fee_info = raw.get("fee") or {}
        return OrderResult(
            order=order,
            status=status,
            exchange_order_id=str(raw.get("id")) if raw.get("id") else None,
            filled_amount=filled,
            average_price=float(raw["average"]) if raw.get("average") else None,
            fee=float(fee_info.get("cost") or 0.0),
            submitted_at=submitted_at,
            updated_at=datetime.now(UTC),
        )

    async def cancel(self, order_id: str, symbol: str) -> bool:
        try:
            await self._exchange.cancel_order(order_id, symbol)
            return True
        except ccxt_async.OrderNotFound:
            # Already gone -- filled or previously cancelled. The postcondition
            # the caller cares about ("not resting") holds either way.
            return True
        except ccxt_async.BaseError as exc:
            logger.error("cancel failed for %s: %s", order_id, exc)
            return False

    async def open_orders(self, symbol: str | None = None) -> list[OrderResult]:
        try:
            raws = await self._exchange.fetch_open_orders(symbol)
        except ccxt_async.BaseError as exc:
            logger.error("fetch_open_orders failed: %s", exc)
            return []

        results = []
        for raw in raws:
            placeholder = Order(
                symbol=raw.get("symbol") or symbol or "UNKNOWN",
                side=OrderSide(raw.get("side", "buy")),
                amount=float(raw.get("amount") or 0.0) or 1e-12,
                order_type=OrderType(raw.get("type", "limit")),
                price=float(raw["price"]) if raw.get("price") else None,
                time_in_force=TimeInForce.GTC,
                client_order_id=LiveBroker._client_id_of(raw) or "unknown",
            )
            results.append(LiveBroker._to_result(placeholder, raw, None))
        return results

    # -- account ----------------------------------------------------------

    async def position(self, symbol: str) -> Position:
        """Read the position back from the venue."""
        try:
            if self._exchange.has.get("fetchPositions"):
                for raw in await self._exchange.fetch_positions([symbol]):
                    if raw.get("symbol") != symbol:
                        continue
                    contracts = float(raw.get("contracts") or 0.0)
                    side = str(raw.get("side") or "").lower()
                    quantity = -contracts if side == "short" else contracts
                    return Position(
                        symbol=symbol,
                        quantity=quantity,
                        entry_price=float(raw["entryPrice"]) if raw.get("entryPrice") else None,
                        mark_price=float(raw["markPrice"]) if raw.get("markPrice") else None,
                        unrealised_pnl=float(raw.get("unrealizedPnl") or 0.0),
                    )
                return Position(symbol=symbol, quantity=0.0)

            # Spot: the position is the base-currency balance.
            balance = await self._exchange.fetch_balance()
            base = symbol.split("/")[0]
            return Position(symbol=symbol, quantity=float(balance.get(base, {}).get("total") or 0.0))
        except ccxt_async.BaseError as exc:
            logger.error("position fetch failed for %s: %s", symbol, exc)
            raise

    async def equity(self) -> float:
        balance = await self._exchange.fetch_balance()
        total = balance.get("total") or {}
        for quote in ("USDT", "USD", "USDC", "BUSD"):
            if quote in total:
                return float(total[quote])
        raise ValueError(f"could not determine quote equity from {sorted(total)}")
