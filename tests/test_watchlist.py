"""The auto-watchlist: what the bot is currently looking at.

Two things are load-bearing and both are tested here. The first is that a poll
costs *one* request rather than one per symbol -- the loop form is sequential,
so twelve symbols meant twelve round trips of latency stacked inside a single
tick, which is exactly what makes a watchlist feel slow and what trips venue
rate limits. The second is that a tick can never raise: this runs unattended
for weeks behind a UI, and a poller that dies takes the whole panel with it.
"""

from __future__ import annotations

import asyncio

import pytest

from godalgo.ui.credentials import CredentialStore
from godalgo.ui.feed import MarketFeed, _row
from godalgo.ui.journal import TradingJournal
from godalgo.ui.server import UIBridge


@pytest.fixture
def bridge(tmp_path):
    return UIBridge(
        journal=TradingJournal(path=tmp_path / "j.jsonl", summary_path=tmp_path / "s.jsonl"),
        credentials=CredentialStore(directory=tmp_path),
        market_feed_enabled=False,
    )


class _Venue:
    """A venue that counts how it was asked."""

    def __init__(self, *, batched=True, prices=None, fail_batch=False):
        self.has = {"fetchTickers": batched}
        self.prices = prices or {"BTC/USDT": 64000.0, "ETH/USDT": 3200.0}
        self.fail_batch = fail_batch
        self.batch_calls = 0
        self.single_calls = 0

    def _ticker(self, symbol):
        price = self.prices[symbol]
        return {
            "last": price, "bid": price * 0.9999, "ask": price * 1.0001,
            "percentage": 2.5, "quoteVolume": 1_000_000.0,
        }

    async def fetch_tickers(self, symbols=None):
        self.batch_calls += 1
        if self.fail_batch:
            raise NotImplementedError("this venue rejects an explicit symbol list")
        return {s: self._ticker(s) for s in (symbols or self.prices)}

    async def fetch_ticker(self, symbol):
        self.single_calls += 1
        return self._ticker(symbol)

    async def close(self):
        pass


def _run_tick(bridge, venue, symbols):
    feed = MarketFeed(bridge, symbols=symbols)
    feed._exchange = venue
    asyncio.run(feed._tick())
    return feed


# --- the request-count property, which is the whole anti-lag argument ------

def test_a_whole_watchlist_costs_one_request(bridge):
    """Not one per symbol. With a dozen symbols on a five second cadence this
    is the difference between 12 requests and 1, and the per-symbol form is
    sequential, so its latency stacks."""
    venue = _Venue(prices={f"S{i}/USDT": 100.0 + i for i in range(12)})
    _run_tick(bridge, venue, list(venue.prices))

    assert venue.batch_calls == 1
    assert venue.single_calls == 0
    assert len(bridge.watchlist) == 12


def test_a_venue_without_batching_still_works(bridge):
    venue = _Venue(batched=False)
    _run_tick(bridge, venue, ["BTC/USDT", "ETH/USDT"])

    assert venue.batch_calls == 0
    assert venue.single_calls == 2
    assert len(bridge.watchlist) == 2


def test_a_venue_that_advertises_batching_but_refuses_falls_back_once(bridge):
    """And remembers, so the failed call is not repaid every five seconds."""
    venue = _Venue(fail_batch=True)
    feed = MarketFeed(bridge, symbols=["BTC/USDT", "ETH/USDT"])
    feed._exchange = venue

    asyncio.run(feed._tick())
    asyncio.run(feed._tick())
    asyncio.run(feed._tick())

    assert venue.batch_calls == 1, "should not retry batching after it failed"
    assert venue.single_calls == 6
    assert len(bridge.watchlist) == 2


# --- never crash ------------------------------------------------------------

def test_a_venue_outage_does_not_raise_or_empty_the_list(bridge):
    """The panel must keep its last good values and go grey, not go blank."""
    venue = _Venue()
    _run_tick(bridge, venue, ["BTC/USDT", "ETH/USDT"])
    assert bridge.watchlist["BTC/USDT"].price == 64000.0

    class _Dead(_Venue):
        async def fetch_tickers(self, symbols=None):
            raise ConnectionError("venue down")

    feed = MarketFeed(bridge, symbols=["BTC/USDT", "ETH/USDT"])
    feed._exchange = _Dead()
    asyncio.run(feed._tick())  # must not raise

    assert bridge.watchlist["BTC/USDT"].price == 64000.0
    assert bridge.connected is False
    assert bridge.venue_status["market_data"]["ok"] is False


def test_a_symbol_the_venue_drops_goes_stale_rather_than_vanishing(bridge):
    """A row that disappears and returns makes the list flicker and loses the
    reader's place; a greyed row says plainly that one number is old."""
    venue = _Venue()
    _run_tick(bridge, venue, ["BTC/USDT", "ETH/USDT"])
    assert not bridge.watchlist["ETH/USDT"].stale

    partial = _Venue(prices={"BTC/USDT": 65000.0})
    _run_tick(bridge, partial, ["BTC/USDT"])

    assert "ETH/USDT" in bridge.watchlist
    assert bridge.watchlist["ETH/USDT"].stale is True
    assert bridge.watchlist["BTC/USDT"].stale is False
    assert bridge.watchlist["BTC/USDT"].price == 65000.0


def test_a_malformed_ticker_does_not_take_down_the_tick():
    """Venues omit fields. A missing bid must not raise on the spread maths."""
    row = _row({"last": 100.0})
    assert row["price"] == 100.0
    assert row["spread_bps"] == 0.0
    assert row["change_pct"] == 0.0

    assert _row({}) ["price"] == 0.0


# --- content ---------------------------------------------------------------

def test_percentage_is_converted_from_percent_to_fraction():
    """ccxt reports `percentage` as a percent. Treating 2.5 as a fraction
    would render a 2.5% move as 250%."""
    assert _row({"last": 1.0, "percentage": 2.5})["change_pct"] == pytest.approx(0.025)


def test_spread_is_basis_points_of_the_mid():
    row = _row({"last": 100.0, "bid": 99.95, "ask": 100.05})
    assert row["spread_bps"] == pytest.approx(10.0, abs=0.01)


def test_the_traded_symbol_is_flagged_and_sorted_first(bridge):
    bridge.symbol = "ETH/USDT"
    venue = _Venue(prices={"BTC/USDT": 64000.0, "ETH/USDT": 3200.0})
    _run_tick(bridge, venue, ["BTC/USDT", "ETH/USDT"])

    rows = bridge.watchlist_rows()
    assert rows[0].symbol == "ETH/USDT"
    assert rows[0].active is True
    assert rows[1].active is False


def test_held_symbols_sort_above_the_merely_active(bridge):
    """What the bot is *in* outranks what it is looking at."""
    bridge.symbol = "ETH/USDT"
    bridge.record_fill("SOL/USDT", 1.0, 150.0)
    venue = _Venue(prices={"BTC/USDT": 1.0, "ETH/USDT": 2.0, "SOL/USDT": 3.0})
    _run_tick(bridge, venue, list(venue.prices))

    rows = bridge.watchlist_rows()
    assert rows[0].symbol == "SOL/USDT"
    assert rows[0].held is True
    assert rows[1].symbol == "ETH/USDT"


def test_the_held_flag_clears_when_the_position_closes(bridge):
    """Recomputed each tick rather than tracked: a stale marker claims the bot
    is in something it is not."""
    venue = _Venue(prices={"SOL/USDT": 150.0})
    bridge.record_fill("SOL/USDT", 1.0, 150.0)
    _run_tick(bridge, venue, ["SOL/USDT"])
    assert bridge.watchlist["SOL/USDT"].held is True

    bridge.record_fill("SOL/USDT", -1.0, 160.0)
    _run_tick(bridge, venue, ["SOL/USDT"])
    assert bridge.watchlist["SOL/USDT"].held is False


def test_the_headline_price_follows_the_traded_symbol(bridge):
    """Not whatever the venue returned first, which is arbitrary dict order."""
    bridge.symbol = "ETH/USDT"
    venue = _Venue(prices={"BTC/USDT": 64000.0, "ETH/USDT": 3200.0})
    _run_tick(bridge, venue, ["BTC/USDT", "ETH/USDT"])
    assert bridge.last_price == 3200.0


def test_the_watchlist_reaches_the_snapshot_and_serialises(bridge):
    import json

    venue = _Venue()
    _run_tick(bridge, venue, ["BTC/USDT", "ETH/USDT"])
    body = bridge.snapshot().to_dict()

    assert len(body["watchlist"]) == 2
    assert {"symbol", "price", "change_pct", "spread_bps", "held", "active",
            "stale"} <= set(body["watchlist"][0])
    json.dumps(body)


def test_the_snapshot_carries_no_raw_venue_payload(bridge):
    """A ccxt ticker embeds an `info` blob of raw venue JSON. Shipping that per
    symbol once a second is a frame far larger than the panel needs."""
    venue = _Venue()
    venue.prices = {"BTC/USDT": 64000.0}
    original = venue._ticker

    def fat(symbol):
        ticker = original(symbol)
        ticker["info"] = {"junk": "x" * 5000}
        return ticker

    venue._ticker = fat
    _run_tick(bridge, venue, ["BTC/USDT"])

    import json
    blob = json.dumps(bridge.snapshot().to_dict())
    assert "junk" not in blob
    assert len(blob) < 20_000
