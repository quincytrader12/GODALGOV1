"""Execution layer: order construction, economics, fills, and safety."""

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from godalgo.execution.broker import BookSnapshot, DryRunBroker, FeeSchedule, PaperBroker
from godalgo.execution.router import OrderRouter, RoutingConfig
from godalgo.execution.types import (
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
)


def book(bid=50_000.0, ask=50_010.0):
    return BookSnapshot("BTC/USDT", bid=bid, ask=ask)


# --- types -----------------------------------------------------------------

def test_order_rejects_non_positive_amount():
    with pytest.raises(ValueError, match="amount must be positive"):
        Order("BTC/USDT", OrderSide.BUY, 0.0, price=1.0)


def test_limit_order_requires_a_price():
    with pytest.raises(ValueError, match="require a price"):
        Order("BTC/USDT", OrderSide.BUY, 1.0, OrderType.LIMIT)


def test_post_only_market_order_is_incoherent():
    with pytest.raises(ValueError, match="post-only is meaningless"):
        Order("BTC/USDT", OrderSide.BUY, 1.0, OrderType.MARKET,
              time_in_force=TimeInForce.POST_ONLY)


def test_client_order_ids_are_unique():
    """Idempotency depends on this; a collision means a retry hits the wrong order."""
    ids = {Order("BTC/USDT", OrderSide.BUY, 1.0, price=1.0).client_order_id for _ in range(500)}
    assert len(ids) == 500


def test_unknown_status_is_not_terminal():
    """An ambiguous send must not be treated as 'did not happen'."""
    assert not OrderStatus.UNKNOWN.is_terminal
    assert not OrderStatus.UNKNOWN.is_live


def test_crossed_book_is_rejected():
    with pytest.raises(ValueError, match="crossed book"):
        BookSnapshot("BTC/USDT", bid=100.0, ask=99.0)


# --- router economics ------------------------------------------------------

def test_taker_round_trip_costs_more_than_maker():
    """The gap is what decides whether fast trading is viable at all."""
    r = OrderRouter()
    b = book()
    assert r.round_trip_cost_bps(b, is_maker=False) > r.round_trip_cost_bps(b, is_maker=True)


def test_trade_refused_when_edge_below_cost():
    r = OrderRouter()
    d = r.decide("BTC/USDT", 0.5, 0.0, 100_000.0, book(), expected_edge_bps=1.0)
    assert not d.should_trade
    assert "below required" in d.reason


def test_trade_allowed_when_edge_clears_cost():
    r = OrderRouter()
    d = r.decide("BTC/USDT", 0.5, 0.0, 100_000.0, book(), expected_edge_bps=50.0)
    assert d.should_trade
    assert d.order.side is OrderSide.BUY


def test_exits_are_exempt_from_the_edge_gate():
    """A risk exit must never be blocked for being unprofitable on its own."""
    r = OrderRouter()
    d = r.decide("BTC/USDT", target_weight=0.0, current_weight=0.5,
                 equity=100_000.0, book=book(), expected_edge_bps=0.0)
    assert d.should_trade
    assert d.order.reduce_only


def test_wide_spread_stands_the_router_down():
    r = OrderRouter()
    d = r.decide("BTC/USDT", 0.5, 0.0, 100_000.0,
                 book(bid=50_000.0, ask=50_500.0), expected_edge_bps=1e4)
    assert not d.should_trade
    assert "spread" in d.reason


def test_one_order_per_symbol_in_flight():
    """Stacking orders against one target is how a signal becomes a double position."""
    r = OrderRouter()
    first = r.decide("BTC/USDT", 0.5, 0.0, 100_000.0, book(), expected_edge_bps=50.0)
    r.mark_submitted(first.order)
    second = r.decide("BTC/USDT", 0.6, 0.0, 100_000.0, book(), expected_edge_bps=50.0)
    assert not second.should_trade
    assert "in flight" in second.reason

    r.mark_settled("BTC/USDT")
    assert r.decide("BTC/USDT", 0.6, 0.0, 100_000.0, book(), expected_edge_bps=50.0).should_trade


def test_throttle_blocks_runaway_submission():
    r = OrderRouter(RoutingConfig(max_orders_per_minute=3))
    for _ in range(3):
        d = r.decide("BTC/USDT", 0.5, 0.0, 100_000.0, book(), expected_edge_bps=50.0)
        r.mark_submitted(d.order)
        r.mark_settled("BTC/USDT")
    blocked = r.decide("BTC/USDT", 0.5, 0.0, 100_000.0, book(), expected_edge_bps=50.0)
    assert not blocked.should_trade
    assert "throttle" in blocked.reason


def test_sub_minimum_notional_is_refused():
    r = OrderRouter(RoutingConfig(min_notional=1_000.0))
    d = r.decide("BTC/USDT", 0.05, 0.0, 100.0, book(), expected_edge_bps=100.0)
    assert not d.should_trade
    assert "minimum" in d.reason


def test_negative_edge_multiple_is_rejected():
    with pytest.raises(ValueError, match="negative expected value"):
        RoutingConfig(min_edge_multiple=0.5)


# --- paper broker fill semantics -------------------------------------------

def test_post_only_rests_and_fills_only_on_a_through_trade():
    async def run():
        broker = PaperBroker(starting_equity=100_000.0)
        broker.on_book(book())
        order = Order("BTC/USDT", OrderSide.BUY, 0.1, price=50_000.0)
        result = await broker.submit(order)
        assert result.status is OrderStatus.OPEN
        assert (await broker.position("BTC/USDT")).is_flat

        # Price merely touching the level must not fill -- we are behind the queue.
        assert broker.on_book(book(bid=49_990.0, ask=50_000.0)) == []
        assert (await broker.position("BTC/USDT")).is_flat

        # Trading through does fill.
        fills = broker.on_book(book(bid=49_980.0, ask=49_990.0))
        assert len(fills) == 1
        assert (await broker.position("BTC/USDT")).quantity == pytest.approx(0.1)

    asyncio.run(run())


def test_post_only_that_would_cross_is_rejected_not_filled():
    async def run():
        broker = PaperBroker()
        broker.on_book(book())
        crossing = Order("BTC/USDT", OrderSide.BUY, 0.1, price=50_020.0)
        result = await broker.submit(crossing)
        assert result.status is OrderStatus.REJECTED
        assert "cross" in result.error

    asyncio.run(run())


def test_market_order_pays_the_taker_fee():
    async def run():
        broker = PaperBroker(fees=FeeSchedule(maker=0.0, taker=0.001))
        broker.on_book(book())
        order = Order("BTC/USDT", OrderSide.BUY, 0.1, OrderType.MARKET,
                      time_in_force=TimeInForce.IOC)
        result = await broker.submit(order)
        assert result.status is OrderStatus.FILLED
        assert result.fee > 0

    asyncio.run(run())


def test_round_trip_realises_pnl_and_returns_to_flat():
    """Isolates P&L accounting: fees and slippage zeroed so the arithmetic is exact."""
    async def run():
        broker = PaperBroker(starting_equity=100_000.0, fees=FeeSchedule(0.0, 0.0),
                             taker_slippage_bps=0.0)
        broker.on_book(book(bid=50_000.0, ask=50_000.0))
        await broker.submit(Order("BTC/USDT", OrderSide.BUY, 1.0, OrderType.MARKET,
                                  time_in_force=TimeInForce.IOC))
        broker.on_book(book(bid=51_000.0, ask=51_000.0))
        await broker.submit(Order("BTC/USDT", OrderSide.SELL, 1.0, OrderType.MARKET,
                                  time_in_force=TimeInForce.IOC))
        assert (await broker.position("BTC/USDT")).is_flat
        assert broker.realised_pnl == pytest.approx(1_000.0)

    asyncio.run(run())


def test_taker_slippage_is_charged_on_both_legs():
    """Regression: a broker that skips slippage flatters every fast strategy."""
    async def run():
        broker = PaperBroker(fees=FeeSchedule(0.0, 0.0), taker_slippage_bps=1.0)
        broker.on_book(book(bid=50_000.0, ask=50_000.0))
        await broker.submit(Order("BTC/USDT", OrderSide.BUY, 1.0, OrderType.MARKET,
                                  time_in_force=TimeInForce.IOC))
        broker.on_book(book(bid=51_000.0, ask=51_000.0))
        await broker.submit(Order("BTC/USDT", OrderSide.SELL, 1.0, OrderType.MARKET,
                                  time_in_force=TimeInForce.IOC))
        # 1bp paid away on entry and on exit.
        assert broker.realised_pnl == pytest.approx(989.9, abs=0.05)

    asyncio.run(run())


def test_cancel_all_clears_resting_orders():
    async def run():
        broker = PaperBroker()
        broker.on_book(book())
        for price in (49_900.0, 49_800.0):
            await broker.submit(Order("BTC/USDT", OrderSide.BUY, 0.1, price=price))
        assert len(await broker.open_orders()) == 2
        assert await broker.cancel_all("BTC/USDT") == 2
        assert await broker.open_orders() == []

    asyncio.run(run())


# --- dry run ---------------------------------------------------------------

def test_dry_run_never_sends():
    async def run():
        broker = DryRunBroker()
        result = await broker.submit(Order("BTC/USDT", OrderSide.BUY, 1.0, price=1.0))
        assert result.status is OrderStatus.REJECTED
        assert "dry run" in result.error
        assert (await broker.position("BTC/USDT")).is_flat

    asyncio.run(run())


# --- arming ----------------------------------------------------------------

def test_live_broker_refuses_without_explicit_arming(monkeypatch):
    from godalgo.execution.live import ArmingError, LiveBroker

    monkeypatch.delenv("GODALGO_ARM_LIVE", raising=False)
    with pytest.raises(ArmingError, match="arm=True"):
        LiveBroker()
    with pytest.raises(ArmingError, match="GODALGO_ARM_LIVE"):
        LiveBroker(arm=True)


def test_live_broker_refuses_without_credentials(monkeypatch):
    from godalgo.execution.live import ArmingError, LiveBroker

    monkeypatch.setenv("GODALGO_ARM_LIVE", "I_UNDERSTAND_THIS_TRADES_REAL_MONEY")
    monkeypatch.delenv("GODALGO_API_KEY", raising=False)
    monkeypatch.delenv("GODALGO_API_SECRET", raising=False)
    with pytest.raises(ArmingError, match="missing credentials"):
        LiveBroker(arm=True)


# --- bar aggregation -------------------------------------------------------

def test_bars_close_on_wall_clock_boundaries_and_exclude_the_forming_bar():
    from godalgo.data.stream import BarAggregator

    agg = BarAggregator(interval_seconds=60)
    start = datetime(2024, 1, 1, 0, 0, 30, tzinfo=UTC)
    completed = [agg.on_tick(100.0 + i, 1.0, start + timedelta(seconds=i * 20))
                 for i in range(10)]
    bars = [b for b in completed if b is not None]
    assert bars
    assert all(b.start.second == 0 for b in bars)
    assert len(agg.frame()) == agg.n_complete


def test_staleness_is_measurable():
    from godalgo.data.stream import BarAggregator

    agg = BarAggregator(interval_seconds=60)
    t = datetime(2024, 1, 1, tzinfo=UTC)
    for i in range(200):
        agg.on_tick(100.0, 1.0, t + timedelta(seconds=i))
    age = agg.last_bar_age(t + timedelta(seconds=600))
    assert age is not None and age.total_seconds() > 300
