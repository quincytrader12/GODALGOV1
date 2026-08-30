"""LiveBroker conformance against a mock ccxt exchange.

A real venue is unreachable from this sandbox, so these test the half that is
ours: how we shape orders, how we interpret venue responses, and above all how
we behave when the venue's answer is ambiguous.

The ambiguous-send path is the one that matters. A timeout after transmission
leaves an order that may be live; treating that as "did not happen" and retrying
is how a position silently doubles.
"""

import asyncio

import ccxt.async_support as ccxt_async
import pytest

from godalgo.execution.live import ArmingError, LiveBroker, LiveBrokerConfig
from godalgo.execution.market import MarketSpec, spec_from_ccxt_market
from godalgo.execution.types import Order, OrderSide, OrderStatus, OrderType, TimeInForce

ARM_TOKEN = "I_UNDERSTAND_THIS_TRADES_REAL_MONEY"


class MockCcxt:
    """Stand-in for a ccxt async exchange."""

    def __init__(self, *, create_results=None, open_orders=None, closed_orders=None,
                 markets=None, balance=None, positions=None):
        self._create = list(create_results or [])
        self._open = open_orders or []
        self._closed = closed_orders or []
        self._markets = markets or {}
        self._balance = balance or {"total": {"USDT": 10_000.0}}
        self._positions = positions or []
        self.has = {"fetchPositions": bool(positions)}
        self.create_calls = []
        self.cancelled = []
        self.closed_session = False

    async def create_order(self, **kwargs):
        self.create_calls.append(kwargs)
        item = self._create.pop(0) if self._create else {"id": "1", "status": "open"}
        if isinstance(item, Exception):
            raise item
        return item

    async def fetch_open_orders(self, symbol=None):
        return self._open

    async def fetch_closed_orders(self, symbol=None, limit=None):
        return self._closed

    async def cancel_order(self, order_id, symbol):
        self.cancelled.append(order_id)

    async def load_markets(self):
        return self._markets

    async def fetch_balance(self):
        return self._balance

    async def fetch_positions(self, symbols=None):
        return self._positions

    async def close(self):
        self.closed_session = True


@pytest.fixture
def armed(monkeypatch):
    monkeypatch.setenv("GODALGO_ARM_LIVE", ARM_TOKEN)
    monkeypatch.setenv("GODALGO_API_KEY", "test-key")
    monkeypatch.setenv("GODALGO_API_SECRET", "test-secret")


def make_broker(mock, config=None):
    broker = LiveBroker.__new__(LiveBroker)
    # A cap above the test order sizes, so only the test that targets the cap
    # exercises it.
    broker.config = config or LiveBrokerConfig(max_order_notional=1_000_000.0)
    broker._exchange = mock
    return broker


def an_order(amount=0.1, price=50_000.0):
    return Order("BTC/USDT", OrderSide.BUY, amount, OrderType.LIMIT, price=price)


# --- arming ----------------------------------------------------------------

def test_arming_requires_all_three_keys(monkeypatch):
    monkeypatch.delenv("GODALGO_ARM_LIVE", raising=False)
    with pytest.raises(ArmingError):
        LiveBroker(arm=True)


def test_credentials_are_never_taken_as_arguments():
    """Secrets passed as arguments end up in a log line or a traceback."""
    import inspect

    params = inspect.signature(LiveBrokerConfig).parameters
    assert not any("key" in p or "secret" in p for p in params if not p.endswith("_env"))


# --- order submission ------------------------------------------------------

def test_filled_order_is_parsed():
    mock = MockCcxt(create_results=[
        {"id": "abc", "status": "closed", "filled": 0.1, "average": 50_000.0,
         "fee": {"cost": 3.0}},
    ])
    result = asyncio.run(make_broker(mock).submit(an_order()))
    assert result.status is OrderStatus.FILLED
    assert result.filled_amount == 0.1
    assert result.average_price == 50_000.0
    assert result.fee == 3.0


def test_partial_fill_is_distinguished_from_open():
    mock = MockCcxt(create_results=[{"id": "a", "status": "open", "filled": 0.04}])
    result = asyncio.run(make_broker(mock).submit(an_order(amount=0.1)))
    assert result.status is OrderStatus.PARTIALLY_FILLED
    assert result.remaining == pytest.approx(0.06)


def test_client_order_id_and_post_only_are_sent():
    """Idempotency and fee tier both depend on these reaching the venue."""
    mock = MockCcxt()
    order = Order("BTC/USDT", OrderSide.BUY, 0.1, OrderType.LIMIT, price=50_000.0,
                  time_in_force=TimeInForce.POST_ONLY, reduce_only=True)
    asyncio.run(make_broker(mock).submit(order))
    params = mock.create_calls[0]["params"]
    assert params["clientOrderId"] == order.client_order_id
    assert params["postOnly"] is True
    assert params["reduceOnly"] is True


def test_invalid_order_is_rejected_without_retry():
    """A malformed order will be malformed again; retrying only wastes rate limit."""
    mock = MockCcxt(create_results=[ccxt_async.InvalidOrder("bad precision")] * 3)
    result = asyncio.run(make_broker(mock).submit(an_order()))
    assert result.status is OrderStatus.REJECTED
    assert len(mock.create_calls) == 1


def test_insufficient_funds_is_rejected():
    mock = MockCcxt(create_results=[ccxt_async.InsufficientFunds("no balance")])
    result = asyncio.run(make_broker(mock).submit(an_order()))
    assert result.status is OrderStatus.REJECTED
    assert "insufficient funds" in result.error


def test_notional_cap_blocks_before_sending():
    """A last-resort backstop against a sizing bug, checked on every order."""
    mock = MockCcxt()
    broker = make_broker(mock, LiveBrokerConfig(max_order_notional=100.0))
    result = asyncio.run(broker.submit(an_order(amount=1.0, price=50_000.0)))
    assert result.status is OrderStatus.REJECTED
    assert "max_order_notional" in result.error
    assert mock.create_calls == []


# --- the ambiguous send ----------------------------------------------------

def test_timeout_recovers_the_order_by_client_id():
    """The critical path: the order was live all along, so do not resend."""
    order = an_order()
    mock = MockCcxt(
        create_results=[ccxt_async.RequestTimeout("timed out")],
        open_orders=[{"id": "found", "status": "open", "filled": 0.0,
                      "clientOrderId": order.client_order_id}],
    )
    result = asyncio.run(make_broker(mock).submit(order))
    assert result.status is OrderStatus.OPEN
    assert result.exchange_order_id == "found"
    assert len(mock.create_calls) == 1, "must not resend an order that exists"


def test_timeout_finds_an_already_filled_order():
    order = an_order()
    mock = MockCcxt(
        create_results=[ccxt_async.RequestTimeout("timed out")],
        closed_orders=[{"id": "done", "status": "closed", "filled": 0.1,
                        "average": 50_000.0, "clientOrderId": order.client_order_id}],
    )
    result = asyncio.run(make_broker(mock).submit(order))
    assert result.status is OrderStatus.FILLED
    assert len(mock.create_calls) == 1


def test_unrecoverable_timeout_reports_unknown_not_rejected():
    """UNKNOWN forces the caller to reconcile. REJECTED would invite a resend."""
    mock = MockCcxt(create_results=[ccxt_async.RequestTimeout("t")] * 5)
    broker = make_broker(mock, LiveBrokerConfig(max_retries=1, max_order_notional=1e9))
    result = asyncio.run(broker.submit(an_order()))
    assert result.status is OrderStatus.UNKNOWN
    assert not result.status.is_terminal


def test_failed_lookup_does_not_masquerade_as_not_sent():
    class FailingLookup(MockCcxt):
        async def fetch_open_orders(self, symbol=None):
            raise ccxt_async.ExchangeError("lookup down")

    mock = FailingLookup(create_results=[ccxt_async.RequestTimeout("t")] * 5)
    broker = make_broker(mock, LiveBrokerConfig(max_retries=0, max_order_notional=1e9))
    result = asyncio.run(broker.submit(an_order()))
    assert result.status is OrderStatus.UNKNOWN


# --- cancel ----------------------------------------------------------------

def test_cancelling_a_missing_order_succeeds():
    """The caller's postcondition is 'not resting', which already holds."""
    class NotFound(MockCcxt):
        async def cancel_order(self, order_id, symbol):
            raise ccxt_async.OrderNotFound("gone")

    assert asyncio.run(make_broker(NotFound()).cancel("x", "BTC/USDT")) is True


def test_cancel_failure_is_reported():
    class Broken(MockCcxt):
        async def cancel_order(self, order_id, symbol):
            raise ccxt_async.ExchangeError("nope")

    assert asyncio.run(make_broker(Broken()).cancel("x", "BTC/USDT")) is False


# --- account ---------------------------------------------------------------

def test_spot_position_reads_the_base_balance():
    mock = MockCcxt(balance={"total": {"USDT": 5_000.0, "BTC": 0.25}})
    position = asyncio.run(make_broker(mock).position("BTC/USDT"))
    assert position.quantity == pytest.approx(0.25)


def test_short_futures_position_is_signed_negative():
    mock = MockCcxt(positions=[
        {"symbol": "BTC/USDT", "contracts": 2.0, "side": "short",
         "entryPrice": 50_000.0, "markPrice": 49_000.0, "unrealizedPnl": 2_000.0},
    ])
    position = asyncio.run(make_broker(mock).position("BTC/USDT"))
    assert position.quantity == pytest.approx(-2.0)


def test_equity_falls_back_across_quote_currencies():
    mock = MockCcxt(balance={"total": {"USDC": 1_234.0}})
    assert asyncio.run(make_broker(mock).equity()) == pytest.approx(1_234.0)


def test_unrecognised_quote_currency_raises_rather_than_guessing():
    mock = MockCcxt(balance={"total": {"XYZ": 1.0}})
    with pytest.raises(ValueError, match="could not determine quote equity"):
        asyncio.run(make_broker(mock).equity())


# --- market metadata and preflight ----------------------------------------

def test_unknown_symbol_names_near_misses():
    mock = MockCcxt(markets={"BTC/USD": {}, "BTC/EUR": {}})
    with pytest.raises(ValueError, match="did you mean"):
        asyncio.run(make_broker(mock).load_market("BTC/USDT"))


def test_market_spec_is_loaded_from_the_venue():
    mock = MockCcxt(markets={"BTC/USDT": {
        "symbol": "BTC/USDT", "precision": {"amount": 1e-05, "price": 0.01},
        "limits": {"amount": {"min": 1e-05}, "cost": {"min": 5.0}},
        "maker": 0.0002, "taker": 0.0006,
    }})
    spec = asyncio.run(make_broker(mock).load_market("BTC/USDT"))
    assert not spec.is_default
    assert spec.amount_precision == 5
    assert spec.min_notional == 5.0


def test_preflight_reports_blockers_before_any_order():
    mock = MockCcxt(
        markets={"BTC/USDT": {"symbol": "BTC/USDT", "precision": {"amount": 3, "price": 2},
                              "limits": {"cost": {"min": 10.0}}, "maker": 0.0002, "taker": 0.0006}},
        balance={"total": {"USDT": 0.0}},
        open_orders=[{"id": "stale", "status": "open", "filled": 0.0}],
    )
    report = asyncio.run(make_broker(mock).preflight("BTC/USDT"))
    warnings = " ".join(report["warnings"])
    assert "equity is zero" in warnings
    assert "resting" in warnings
    assert report["round_trip_bps"]["maker"] == pytest.approx(4.0)


def test_preflight_flags_a_venue_where_post_only_gains_nothing():
    mock = MockCcxt(markets={"BTC/USDT": {
        "symbol": "BTC/USDT", "precision": {"amount": 3, "price": 2},
        "limits": {"cost": {"min": 10.0}}, "maker": 0.001, "taker": 0.001,
    }})
    report = asyncio.run(make_broker(mock).preflight("BTC/USDT"))
    assert any("post-only gains nothing" in w for w in report["warnings"])


def test_bad_credentials_raise_rather_than_warn():
    class Unauthorised(MockCcxt):
        async def fetch_balance(self):
            raise ccxt_async.AuthenticationError("bad key")

    mock = Unauthorised(markets={"BTC/USDT": {"symbol": "BTC/USDT", "precision": {},
                                              "limits": {}}})
    with pytest.raises(ArmingError, match="credentials rejected"):
        asyncio.run(make_broker(mock).preflight("BTC/USDT"))


# --- precision conventions -------------------------------------------------

def test_step_and_decimal_precision_forms_both_parse():
    step = spec_from_ccxt_market({"symbol": "A/B", "precision": {"amount": 0.001, "price": 0.01},
                                  "limits": {}})
    decimals = spec_from_ccxt_market({"symbol": "A/B", "precision": {"amount": 3, "price": 2},
                                      "limits": {}})
    assert step.amount_precision == decimals.amount_precision == 3
    assert step.amount_step == 0.001
    assert decimals.amount_step is None


def test_amount_rounds_down_never_up():
    """Rounding up can breach a balance or a risk cap by a hair, which is enough."""
    spec = MarketSpec("A/B", amount_precision=3)
    assert spec.round_amount(0.9999) == 0.999


def test_price_quantisation_only_makes_orders_more_passive():
    spec = MarketSpec("A/B", price_precision=2, price_tick=0.01)
    assert spec.round_price(100.567, side_is_buy=True) <= 100.567
    assert spec.round_price(100.561, side_is_buy=False) >= 100.561
