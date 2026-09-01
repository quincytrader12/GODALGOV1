"""Exchange ids, and the outage one capital letter caused.

Reported from a running build:

    module 'ccxt.async_support' has no attribute 'Binance'

ccxt ids are lowercase. "Binance" was typed into the exchange field, stored
that way, and then every lookup failed with a message naming an attribute
rather than the mistake. Because the market feed had just been changed to
follow the *stored* venue, that typo took market data down entirely and kept
failing every five seconds -- the terminal went from partly working to not
working at all, and nothing on screen said why.

Three defences, all tested here: normalise on the way in, repair what is
already stored, and never let an unusable id stop the feed.
"""

from __future__ import annotations

import pytest

from godalgo.ui.credentials import CredentialStore, ExchangeCredential
from godalgo.ui.venue import (
    known_exchange,
    normalise_exchange_id,
    suggest_exchange_ids,
)

# --- normalising ------------------------------------------------------------

@pytest.mark.parametrize(
    ("typed", "expected"),
    [
        ("Binance", "binance"),
        ("BINANCE", "binance"),
        ("  binance  ", "binance"),
        ("Binance.US", "binanceus"),
        ("binance us", "binanceus"),
        ("Coinbase Pro", "coinbase"),
        ("OKEx", "okx"),
        ("Kraken", "kraken"),
    ],
)
def test_what_people_type_becomes_what_ccxt_expects(typed, expected):
    assert normalise_exchange_id(typed) == expected


def test_every_normalised_alias_is_a_real_ccxt_exchange():
    """An alias table that maps onto a nonexistent id just moves the failure."""
    for typed in ("Binance", "Binance.US", "Coinbase Pro", "OKEx", "gdax"):
        assert known_exchange(normalise_exchange_id(typed)), typed


def test_a_genuine_typo_is_rejected_with_suggestions():
    """The remedy has to be in the message; "unknown exchange" is not one."""
    assert not known_exchange(normalise_exchange_id("binanace"))
    assert "binance" in suggest_exchange_ids("binanace")


def test_normalising_is_idempotent():
    once = normalise_exchange_id("Binance.US")
    assert normalise_exchange_id(once) == once


def test_empty_input_does_not_explode():
    assert normalise_exchange_id("") == ""
    assert normalise_exchange_id(None) == ""


# --- storage: normalise on the way in, and repair what is there ------------

def test_a_capitalised_id_is_normalised_before_it_is_stored(tmp_path):
    """A bad id written to disk outlives the session that made the mistake."""
    store = CredentialStore(directory=tmp_path)
    store.add(ExchangeCredential(
        exchange_id="Binance", api_key="k", api_secret="s", label="mine",
    ))
    assert store.get("mine").exchange_id == "binance"


def test_an_already_stored_bad_id_is_repaired_on_load(tmp_path):
    """The case an operator is actually in: the key is saved wrong already.

    Without this they have to work out that the fix is to delete and retype a
    key that looks perfectly correct on screen.
    """
    import json

    (tmp_path / "credentials.json").write_text(json.dumps({
        "version": 2,
        "credentials": {
            "mine": {
                "exchange_id": "Binance", "api_key": "k", "api_secret": "s",
                "label": "mine", "passphrase": "", "testnet": False,
                "trade_enabled": True, "added_at": "2026-01-01T00:00:00+00:00",
            }
        },
        "secrets": {},
    }))

    store = CredentialStore(directory=tmp_path)
    assert store.get("mine").exchange_id == "binance"
    assert store.tradeable().exchange_id == "binance"


def test_repair_preserves_everything_else(tmp_path):
    """Notably the trade permission: silently clearing consent would be its
    own bug, and re-granting it is not something to make someone redo."""
    store = CredentialStore(directory=tmp_path)
    store.add(ExchangeCredential(
        exchange_id="Binance.US", api_key="k", api_secret="s", label="mine",
        trade_enabled=True, testnet=True, passphrase="p",
    ))
    reloaded = CredentialStore(directory=tmp_path).get("mine")

    assert reloaded.exchange_id == "binanceus"
    assert reloaded.trade_enabled is True
    assert reloaded.testnet is True
    assert reloaded.passphrase == "p"
    assert reloaded.api_secret == "s"


# --- the feed must not be taken down by one ---------------------------------

def test_the_feed_falls_back_rather_than_failing_forever(tmp_path):
    """The actual outage. An unusable id is not a transient fault, so
    retrying it every five seconds forever just looks like a dead venue."""
    import asyncio

    from godalgo.ui.feed import MarketFeed
    from godalgo.ui.server import UIBridge

    bridge = UIBridge(
        credentials=CredentialStore(directory=tmp_path), market_feed_enabled=False,
    )
    feed = MarketFeed(bridge, exchange_id="Binance", symbols=["BTC/USDT"])

    async def build():
        return await feed._client()

    try:
        asyncio.run(build())
    finally:
        asyncio.run(feed._close())

    assert feed.exchange_id == "binance"


def test_an_unrecoverable_id_is_reported_before_the_fallback(tmp_path):
    """Falling back silently would leave someone watching the wrong market
    with no way to know."""
    import asyncio

    from godalgo.ui.feed import MarketFeed
    from godalgo.ui.server import UIBridge

    bridge = UIBridge(
        credentials=CredentialStore(directory=tmp_path), market_feed_enabled=False,
    )
    feed = MarketFeed(bridge, exchange_id="not_a_venue", symbols=["BTC/USDT"])

    try:
        asyncio.run(feed._client())
    finally:
        asyncio.run(feed._close())

    assert feed.exchange_id == "binance"
    messages = [e["message"] for e in bridge.events.entries()]
    assert any("unknown exchange" in m for m in messages)


def test_the_feed_venue_never_resolves_to_something_unusable(tmp_path):
    """_feed_exchange feeds the poller directly, so it must not pass a stored
    typo straight through."""
    from godalgo.ui.server import UIBridge, _feed_exchange

    bridge = UIBridge(
        credentials=CredentialStore(directory=tmp_path), market_feed_enabled=False,
    )
    bridge.exchange_id = "Binance"
    assert known_exchange(_feed_exchange(bridge))


# --- the API rejects rather than stores -------------------------------------

def test_the_refresh_endpoint_rejects_an_unknown_venue(tmp_path):
    from fastapi.testclient import TestClient

    from godalgo.ui.journal import TradingJournal
    from godalgo.ui.server import UIBridge, create_app

    bridge = UIBridge(
        credentials=CredentialStore(directory=tmp_path),
        journal=TradingJournal(path=tmp_path / "j.jsonl",
                               summary_path=tmp_path / "s.jsonl"),
        market_feed_enabled=False,
    )
    client = TestClient(create_app(bridge))

    response = client.post("/api/watchlist/refresh",
                           json={"exchange_id": "binanace"})
    assert response.status_code == 400
    assert "binance" in response.json()["detail"]


def test_a_capitalised_venue_is_accepted_by_the_endpoint(tmp_path, monkeypatch):
    """It is a valid venue typed imperfectly, not an error."""
    from fastapi.testclient import TestClient

    from godalgo.ui.journal import TradingJournal
    from godalgo.ui.server import UIBridge, create_app

    bridge = UIBridge(
        credentials=CredentialStore(directory=tmp_path),
        journal=TradingJournal(path=tmp_path / "j.jsonl",
                               summary_path=tmp_path / "s.jsonl"),
        market_feed_enabled=False,
    )
    client = TestClient(create_app(bridge))

    response = client.post("/api/watchlist/refresh", json={"exchange_id": "Binance"})
    assert response.status_code == 200
    assert response.json()["exchange_id"] == "binance"
