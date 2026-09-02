"""Alpaca: the client and the broker.

Against a stub of the HTTP layer, not a mock of the client, so the request
shapes and the response parsing are both exercised. What matters here is the
handful of places Alpaca does not match the interface it is being fitted to --
each one is a way to lose money quietly rather than loudly.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from godalgo.execution.types import (
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
)
from godalgo.venues.alpaca import (
    AlpacaAsset,
    AlpacaClient,
    AlpacaConfig,
    AlpacaError,
    MarketClock,
    _parse_time,
    _row,
    is_crypto,
)
from godalgo.venues.alpaca_broker import AlpacaBroker, _time_in_force


class _StubClient(AlpacaClient):
    """An AlpacaClient whose transport is a dictionary."""

    def __init__(self, responses: dict[str, Any] | None = None, **kw) -> None:
        super().__init__(config=AlpacaConfig(key_id="k", secret_key="s", **kw))
        self.responses = responses or {}
        self.calls: list[tuple[str, str, dict]] = []

    async def _request(self, method, base, path, **kwargs):
        self.calls.append((method, path, kwargs))
        value = self.responses.get(path)
        if isinstance(value, Exception):
            raise value
        return value


# --------------------------------------------------------------------------
# the paper/live boundary
# --------------------------------------------------------------------------

def test_paper_is_the_default():
    """One string separates paper from real money. It is never the default."""
    assert AlpacaConfig().paper is True
    assert "paper-api" in AlpacaConfig().trading_base
    assert "paper" not in AlpacaConfig(paper=False).trading_base


def test_an_unconfigured_client_knows_it():
    assert AlpacaConfig().configured is False
    assert AlpacaConfig(key_id="k", secret_key="s").configured is True


# --------------------------------------------------------------------------
# asset classes
# --------------------------------------------------------------------------

def test_crypto_is_told_from_equities_by_the_pair_separator():
    assert is_crypto("BTC/USD") is True
    assert is_crypto("AAPL") is False


def test_crypto_ignores_the_market_clock():
    """Most of why crypto is the sensible thing to run first."""
    from datetime import UTC, datetime

    shut = MarketClock(is_open=False, timestamp=datetime.now(UTC))
    assert shut.tradeable("BTC/USD") is True
    assert shut.tradeable("AAPL") is False


def test_the_universe_spans_both_classes():
    client = _StubClient()
    calls: list[str] = []

    async def _assets(asset_class="us_equity", *, tradable_only=True):
        calls.append(asset_class)
        return [AlpacaAsset(symbol="AAPL" if asset_class == "us_equity" else "BTC/USD",
                            asset_class=asset_class)]

    client.assets = _assets
    universe = asyncio.run(client.universe())
    assert sorted(calls) == ["crypto", "us_equity"]
    assert {a.symbol for a in universe} == {"AAPL", "BTC/USD"}


def test_one_asset_class_failing_does_not_lose_the_other():
    """A crypto outage must not empty the equities universe."""
    client = _StubClient()

    async def _assets(asset_class="us_equity", *, tradable_only=True):
        if asset_class == "crypto":
            raise AlpacaError(500, "down")
        return [AlpacaAsset(symbol="AAPL")]

    client.assets = _assets
    assert [a.symbol for a in asyncio.run(client.universe())] == ["AAPL"]


def test_untradable_assets_are_dropped():
    client = _StubClient({"/v2/assets": [
        {"symbol": "AAPL", "tradable": True, "class": "us_equity"},
        {"symbol": "DEAD", "tradable": False, "class": "us_equity"},
    ]})
    assets = asyncio.run(client.assets())
    assert [a.symbol for a in assets] == ["AAPL"]


# --------------------------------------------------------------------------
# market data
# --------------------------------------------------------------------------

def test_snapshots_split_by_class_and_merge():
    """A mixed watchlist costs two round trips, not one per symbol."""
    client = _StubClient()
    seen: list[list[str]] = []

    async def _stock(symbols):
        seen.append(symbols)
        return {"AAPL": {"price": 195.0}}

    async def _crypto(symbols):
        seen.append(symbols)
        return {"BTC/USD": {"price": 77_000.0}}

    client._stock_snapshots = _stock
    client._crypto_snapshots = _crypto
    rows = asyncio.run(client.snapshots(["AAPL", "BTC/USD", "MSFT"]))
    assert set(rows) == {"AAPL", "BTC/USD"}
    assert ["AAPL", "MSFT"] in seen and ["BTC/USD"] in seen


def test_a_snapshot_is_reduced_to_what_the_panel_renders():
    row = _row({
        "latestTrade": {"p": 195.5},
        "latestQuote": {"bp": 195.4, "ap": 195.6},
        "dailyBar": {"v": 1_000_000, "o": 190.0},
        "prevDailyBar": {"c": 190.0},
    })
    assert set(row) == {"price", "change_pct", "quote_volume", "spread_bps"}
    assert row["price"] == 195.5
    assert row["change_pct"] == pytest.approx(0.0289, abs=1e-3)
    assert row["spread_bps"] == pytest.approx(10.2, abs=0.5)


def test_change_is_measured_against_the_previous_close():
    """What '24h change' means for something that does not trade overnight."""
    row = _row({"latestTrade": {"p": 110.0}, "prevDailyBar": {"c": 100.0}})
    assert row["change_pct"] == pytest.approx(0.10)


def test_a_snapshot_with_nothing_in_it_does_not_raise():
    assert _row({})["price"] == 0.0


def test_nanosecond_timestamps_parse():
    """Alpaca sends nanoseconds; fromisoformat does not accept them."""
    assert _parse_time("2026-09-02T17:49:22.123456789Z") is not None
    assert _parse_time("2026-09-02T17:49:22Z") is not None
    assert _parse_time("nonsense") is None
    assert _parse_time(None) is None


# --------------------------------------------------------------------------
# order mapping — where Alpaca does not match the interface
# --------------------------------------------------------------------------

def _order(**kw) -> Order:
    base = {
        "symbol": "AAPL", "side": OrderSide.BUY, "amount": 10.0,
        "order_type": OrderType.LIMIT, "price": 100.0,
    }
    return Order(**{**base, **kw})


def test_post_only_becomes_a_resting_order_not_an_aggressive_one():
    """Alpaca has no post-only. The intent that survives is 'rest', not 'cross'."""
    assert _time_in_force(_order(time_in_force=TimeInForce.POST_ONLY)) == "day"
    assert _time_in_force(
        _order(symbol="BTC/USD", time_in_force=TimeInForce.POST_ONLY)
    ) == "gtc"


def test_crypto_only_gets_the_two_time_in_forces_it_supports():
    for tif in TimeInForce:
        mapped = _time_in_force(_order(symbol="BTC/USD", time_in_force=tif))
        assert mapped in ("gtc", "ioc"), f"{tif} mapped to {mapped}"


def test_a_rejection_is_a_result_not_an_exception():
    """The interface's rule. A raise here would kill the trading loop."""
    client = _StubClient({"/v2/orders": AlpacaError(422, "insufficient buying power")})
    result = asyncio.run(AlpacaBroker(client).submit(_order()))
    assert result.status is OrderStatus.REJECTED
    assert "buying power" in result.error


def test_a_transport_failure_is_unknown_rather_than_rejected():
    """The distinction that stops a position silently doubling: an order that
    may or may not be live is not an order that did not happen."""
    client = _StubClient({"/v2/orders": AlpacaError(0, "ReadTimeout")})
    result = asyncio.run(AlpacaBroker(client).submit(_order()))
    assert result.status is OrderStatus.UNKNOWN


def test_an_unrecognised_status_is_unknown_not_assumed_dead():
    client = _StubClient({"/v2/orders": {"id": "x", "status": "pending_review"}})
    result = asyncio.run(AlpacaBroker(client).submit(_order()))
    assert result.status is OrderStatus.UNKNOWN


def test_a_filled_order_carries_its_fill():
    client = _StubClient({"/v2/orders": {
        "id": "abc", "status": "filled", "filled_qty": "10",
        "filled_avg_price": "99.5", "submitted_at": "2026-09-02T17:00:00Z",
    }})
    result = asyncio.run(AlpacaBroker(client).submit(_order()))
    assert result.status is OrderStatus.FILLED
    assert result.filled_amount == 10.0
    assert result.average_price == 99.5
    assert result.exchange_order_id == "abc"


# --------------------------------------------------------------------------
# reduce-only, which Alpaca does not have
# --------------------------------------------------------------------------

def test_reduce_only_is_clamped_to_the_position():
    """Without this an exit racing a signal flip overshoots through flat and
    opens the opposite position -- the exact thing the flag exists to stop."""
    client = _StubClient({
        "/v2/positions": [{"symbol": "AAPL", "qty": "4", "current_price": "100"}],
        "/v2/orders": {"id": "x", "status": "filled", "filled_qty": "4"},
    })
    order = _order(side=OrderSide.SELL, amount=10.0, reduce_only=True)
    asyncio.run(AlpacaBroker(client).submit(order))

    posted = [c for c in client.calls if c[1] == "/v2/orders"][0]
    assert posted[2]["json"]["qty"] == "4"     # not 10


def test_reduce_only_with_no_position_sends_nothing():
    client = _StubClient({"/v2/positions": []})
    order = _order(side=OrderSide.SELL, reduce_only=True)
    result = asyncio.run(AlpacaBroker(client).submit(order))
    assert result.status is OrderStatus.CANCELED
    assert not [c for c in client.calls if c[0] == "POST"]


def test_reduce_only_does_not_fire_on_the_same_side_as_the_position():
    """Buying while long does not reduce anything."""
    client = _StubClient({
        "/v2/positions": [{"symbol": "AAPL", "qty": "4"}],
    })
    order = _order(side=OrderSide.BUY, reduce_only=True)
    result = asyncio.run(AlpacaBroker(client).submit(order))
    assert result.status is OrderStatus.CANCELED


# --------------------------------------------------------------------------
# positions
# --------------------------------------------------------------------------

def test_a_crypto_position_is_found_despite_the_missing_slash():
    """Alpaca holds BTC/USD as BTCUSD. Matching raw strings reports every
    crypto position as flat, which reads as 'no position' while there is one."""
    client = _StubClient({"/v2/positions": [
        {"symbol": "BTCUSD", "qty": "0.5", "avg_entry_price": "77000",
         "current_price": "77500", "unrealized_pl": "250"},
    ]})
    held = asyncio.run(AlpacaBroker(client).position("BTC/USD"))
    assert held.quantity == 0.5
    assert held.entry_price == 77_000.0
    assert held.is_flat is False


def test_an_absent_position_is_flat_not_an_error():
    client = _StubClient({"/v2/positions": []})
    assert asyncio.run(AlpacaBroker(client).position("AAPL")).is_flat


def test_a_short_position_keeps_its_sign():
    client = _StubClient({"/v2/positions": [{"symbol": "AAPL", "qty": "-3"}]})
    assert asyncio.run(AlpacaBroker(client).position("AAPL")).quantity == -3.0


def test_positions_failing_reports_flat_rather_than_raising():
    client = _StubClient({"/v2/positions": AlpacaError(500, "down")})
    assert asyncio.run(AlpacaBroker(client).position("AAPL")).is_flat


def test_equity_comes_from_the_account():
    client = _StubClient({"/v2/account": {"equity": "10123.45"}})
    assert asyncio.run(AlpacaBroker(client).equity()) == 10_123.45


# --------------------------------------------------------------------------
# cancelling
# --------------------------------------------------------------------------

def test_cancelling_something_already_gone_counts_as_success():
    """The state asked for is the state achieved."""
    client = _StubClient({"/v2/orders/x": AlpacaError(404, "not found")})
    assert asyncio.run(AlpacaBroker(client).cancel("x", "AAPL")) is True


def test_a_failed_cancel_says_so():
    client = _StubClient({"/v2/orders/x": AlpacaError(500, "down")})
    assert asyncio.run(AlpacaBroker(client).cancel("x", "AAPL")) is False


def test_open_orders_filter_by_symbol_across_spellings():
    client = _StubClient({"/v2/orders": [
        {"id": "1", "symbol": "BTCUSD", "side": "buy", "qty": "1",
         "status": "new", "limit_price": "70000"},
        {"id": "2", "symbol": "AAPL", "side": "sell", "qty": "5", "status": "new"},
    ]})
    orders = asyncio.run(AlpacaBroker(client).open_orders("BTC/USD"))
    assert [o.exchange_order_id for o in orders] == ["1"]


def test_the_error_class_separates_auth_from_everything_else():
    """401/403 sends you to the key; 422 sends you to the order."""
    assert AlpacaError(401, "x").is_auth is True
    assert AlpacaError(403, "x").is_auth is True
    assert AlpacaError(422, "x").is_auth is False
