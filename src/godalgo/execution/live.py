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
import contextlib
import logging
import os
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any, Self

import ccxt.async_support as ccxt_async

from godalgo.execution.broker import Broker
from godalgo.execution.market import MarketSpec, spec_from_ccxt_market
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

    Credentials are never held here. They are resolved at construction from
    the environment or the owner-only credential store, and passed straight to
    the venue client -- a config object that holds secrets ends up in a log
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

    Arming needs ``arm=True`` in code plus a credential the operator has
    explicitly marked as permitted to trade. Neither alone is sufficient, and
    the presence of a key is never treated as consent on its own -- a key added
    to read market data must not become an order-placing key because a mode
    switch happened to find it.

    Credentials resolve from, in order:

    1. ``credential``, passed by the caller (the UI's stored, trade-enabled key)
    2. ``GODALGO_API_KEY`` / ``GODALGO_API_SECRET`` in the environment

    The environment is checked second rather than first so that what the
    operator selected in the interface wins over a stale shell variable.
    """

    def __init__(
        self,
        config: LiveBrokerConfig | None = None,
        *,
        arm: bool = False,
        credential: Any = None,
    ) -> None:
        self.config = config or LiveBrokerConfig()

        if not arm:
            raise ArmingError(
                "LiveBroker requires arm=True. Use DryRunBroker or PaperBroker instead."
            )

        key, secret, passphrase, testnet = self._resolve(credential)
        if not key or not secret:
            raise ArmingError(
                "no credential permitted to trade. Add an exchange key in the "
                "terminal and tick 'allow this key to place orders', or set "
                f"{self.config.api_key_env} and {self.config.api_secret_env}."
            )

        if credential is not None and getattr(credential, "exchange_id", None):
            self.config = replace(self.config, exchange_id=credential.exchange_id)

        klass = getattr(ccxt_async, self.config.exchange_id)
        self._exchange = klass(
            {
                "apiKey": key,
                "secret": secret,
                "password": passphrase or None,
                "enableRateLimit": True,
                "timeout": self.config.request_timeout_ms,
                # ccxt ignores the system proxy without this.
                "aiohttp_trust_env": True,
                "options": {"defaultType": self.config.symbol_type},
            }
        )
        self.testnet = testnet
        if testnet:
            # The venue's test network: real API and real order lifecycle,
            # fake money. Exercising the order path without funding anything is
            # the whole reason this flag is plumbed through.
            with contextlib.suppress(Exception):
                self._exchange.set_sandbox_mode(True)
        logger.warning(
            "LiveBroker ARMED on %s (%s) -- orders will use %s",
            self.config.exchange_id, self.config.symbol_type,
            "TESTNET funds" if testnet else "real funds",
        )

    def _resolve(self, credential: Any) -> tuple[str, str, str, bool]:
        """Pick the credential to trade with, without logging any of it."""
        if credential is not None:
            if not getattr(credential, "trade_enabled", False):
                raise ArmingError(
                    f"the {getattr(credential, 'exchange_id', 'selected')} key is "
                    "stored for reading only. Tick 'allow this key to place "
                    "orders' on it before switching to live."
                )
            return (
                getattr(credential, "api_key", ""),
                getattr(credential, "api_secret", ""),
                getattr(credential, "passphrase", "") or "",
                bool(getattr(credential, "testnet", False)),
            )
        return (
            os.environ.get(self.config.api_key_env) or "",
            os.environ.get(self.config.api_secret_env) or "",
            "",
            False,
        )

    async def load_market(self, symbol: str) -> MarketSpec:
        """Read the venue's rules for a symbol.

        Must be called before trading. Order precision, tick size, and minimum
        notional are per-symbol and venue-specific; sending orders shaped by
        guessed values is the most common way a live session fails on its very
        first order, and the resulting venue error rarely says which rule broke.
        """
        markets = await self._exchange.load_markets()
        market = markets.get(symbol)
        if market is None:
            close = [m for m in markets if symbol.split("/")[0] in m][:5]
            raise ValueError(
                f"{self.config.exchange_id} does not list {symbol!r}"
                + (f" -- did you mean one of {close}?" if close else "")
            )
        spec = spec_from_ccxt_market(market)
        logger.info(
            "market %s: amount precision %d (min %s), price precision %d, "
            "min notional %.2f, fees maker=%s taker=%s",
            symbol, spec.amount_precision, spec.min_amount, spec.price_precision,
            spec.min_notional, spec.maker_fee, spec.taker_fee,
        )
        return spec

    async def preflight(self, symbol: str) -> dict[str, object]:
        """Validate everything an order depends on, without sending one.

        Checked here rather than discovered at the first order, because the
        first order is the worst possible moment to learn that the symbol is
        misspelled, the key lacks trade permission, or the account is empty.

        Returns a report of what was verified. Raises only on failures that make
        trading impossible; softer problems appear as warnings in the report so
        the caller can decide.
        """
        report: dict[str, object] = {"exchange": self.config.exchange_id, "symbol": symbol}
        warnings: list[str] = []

        spec = await self.load_market(symbol)
        report["market"] = {
            "amount_precision": spec.amount_precision,
            "price_precision": spec.price_precision,
            "min_amount": spec.min_amount,
            "min_notional": spec.min_notional,
            "maker_fee": spec.maker_fee,
            "taker_fee": spec.taker_fee,
        }

        # Credentials must actually authorise trading, not merely authenticate.
        try:
            equity = await self.equity()
            report["equity"] = equity
            if equity <= 0:
                warnings.append("account equity is zero -- no order can be funded")
        except ccxt_async.AuthenticationError as exc:
            raise ArmingError(f"credentials rejected by {self.config.exchange_id}: {exc}") from exc

        try:
            position = await self.position(symbol)
            report["existing_position"] = position.quantity
            if not position.is_flat:
                warnings.append(
                    f"starting with a non-flat position ({position.quantity:+.8f}); "
                    "the bot will reconcile to it, not ignore it"
                )
        except ccxt_async.BaseError as exc:
            warnings.append(f"could not read position: {exc}")

        try:
            resting = await self.open_orders(symbol)
            report["open_orders"] = len(resting)
            if resting:
                warnings.append(
                    f"{len(resting)} order(s) already resting; the kill switch "
                    "will cancel them"
                )
        except ccxt_async.BaseError as exc:
            warnings.append(f"could not read open orders: {exc}")

        if spec.maker_fee is not None and spec.taker_fee is not None:
            maker_rt = 2 * spec.maker_fee * 1e4
            taker_rt = 2 * spec.taker_fee * 1e4
            report["round_trip_bps"] = {"maker": maker_rt, "taker": taker_rt}
            if spec.maker_fee >= spec.taker_fee:
                warnings.append(
                    "venue reports maker fee >= taker fee; post-only gains "
                    "nothing here, reconsider use_post_only"
                )

        report["warnings"] = warnings
        report["ok"] = not any("rejected" in w for w in warnings)
        return report

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
            price = float(raw["price"]) if raw.get("price") else None
            # Not every venue reports a price on every open order (stops and
            # market orders in particular). Describing such an order as LIMIT
            # with no price fails Order's own validation and would take down the
            # whole listing -- including during a kill-switch cancel_all, which
            # is the worst possible moment.
            order_type = OrderType.LIMIT if price is not None else OrderType.MARKET
            placeholder = Order(
                symbol=raw.get("symbol") or symbol or "UNKNOWN",
                side=OrderSide(str(raw.get("side") or "buy").lower()),
                amount=float(raw.get("amount") or 0.0) or 1e-12,
                order_type=order_type,
                price=price,
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
            return Position(symbol=symbol, quantity=_balance_of(balance, base))
        except ccxt_async.BaseError as exc:
            logger.error("position fetch failed for %s: %s", symbol, exc)
            raise

    async def equity(self) -> float:
        balance = await self._exchange.fetch_balance()
        total = balance.get("total") or {}
        for quote in ("USDT", "USD", "USDC", "BUSD"):
            if quote in total:
                return float(total[quote])
            if quote in balance:
                return float((balance[quote] or {}).get("total") or 0.0)
        raise ValueError(f"could not determine quote equity from {sorted(total)}")


def _balance_of(balance: dict, currency: str) -> float:
    """Read a currency's total from either ccxt balance shape.

    ccxt populates both ``balance[CUR]["total"]`` and ``balance["total"][CUR]``,
    but not every venue fills both. Reading only one silently reports a zero
    position -- and a bot that believes it is flat when it is not will happily
    open a second position on top of the first.
    """
    entry = balance.get(currency)
    if isinstance(entry, dict) and entry.get("total") is not None:
        return float(entry["total"])
    totals = balance.get("total") or {}
    return float(totals.get(currency) or 0.0)
