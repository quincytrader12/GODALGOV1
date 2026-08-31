"""Read-only venue checks, and the error classification that makes them useful.

The classification is the substance here. ccxt collapses several genuinely
different failures onto ``AuthenticationError``, and the remedy differs for
each: a wrong secret, a drifted clock and an IP that is not on the allow-list
all present the same way and need three different actions. Reporting
"authentication failed" for all three sends the operator to re-paste a key that
was always correct.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import ccxt
import pytest

from godalgo.ui.events import EventLog
from godalgo.ui.venue import VenueProbe, classify_error


def _credential(**kwargs):
    base = {
        "exchange_id": "binance", "api_key": "k", "api_secret": "s",
        "passphrase": "", "testnet": False, "trade_enabled": False,
    }
    base.update(kwargs)
    return SimpleNamespace(**base)


# --- classification --------------------------------------------------------

def test_ip_allow_list_rejection_is_not_reported_as_a_bad_key():
    """Binance -2015 is the single most common first-run failure.

    It means the key is fine and the IP is not on its allow-list. Calling it an
    authentication failure sends people to regenerate a working key.
    """
    exc = ccxt.AuthenticationError(
        'binance {"code":-2015,"msg":"Invalid API-key, IP, or permissions for action."}'
    )
    kind, explanation = classify_error(exc)
    assert kind == "ip_not_allowed"
    assert "allow-list" in explanation


def test_clock_skew_is_named_rather_than_called_an_auth_failure():
    """-1021 is a wrong clock, not a wrong key.

    Nothing about the credential will fix it, and it is common on a Windows
    box that has been asleep.
    """
    exc = ccxt.ExchangeError(
        'binance {"code":-1021,"msg":"Timestamp for this request is outside of the recvWindow."}'
    )
    kind, explanation = classify_error(exc)
    assert kind == "clock_skew"
    assert "clock" in explanation.lower()


def test_a_bad_secret_points_at_the_secret():
    exc = ccxt.AuthenticationError('binance {"code":-1022,"msg":"Signature for this request is not valid."}')
    kind, explanation = classify_error(exc)
    assert kind == "bad_signature"
    assert "secret" in explanation.lower()


def test_geo_block_is_distinguished_from_a_dead_connection():
    """HTTP 451 means the venue refused the region, not that the network is down.

    The remedy is a different venue, which no amount of network debugging
    reaches.
    """
    kind, explanation = classify_error(ccxt.NetworkError("binance GET ... 451 Unavailable"))
    assert kind == "geo_blocked"
    assert "region" in explanation.lower()


def test_a_plain_network_failure_stays_a_network_failure():
    kind, _ = classify_error(ccxt.NetworkError("Connection reset by peer"))
    assert kind == "unreachable"


def test_an_unrecognised_error_still_classifies():
    """No exception may escape unclassified; the UI has to render something."""
    kind, explanation = classify_error(RuntimeError("something odd"))
    assert kind == "unknown"
    assert explanation


# --- probes ----------------------------------------------------------------

class _FakeExchange:
    def __init__(self, *, markets=None, ticker=None, balance=None, raises=None):
        self._markets = markets or {}
        self._ticker = ticker or {}
        self._balance = balance or {}
        self._raises = raises or {}
        self.closed = False
        self.sandbox = False

    async def load_markets(self):
        if "load_markets" in self._raises:
            raise self._raises["load_markets"]
        return self._markets

    async def fetch_ticker(self, symbol):
        if "fetch_ticker" in self._raises:
            raise self._raises["fetch_ticker"]
        return self._ticker

    async def fetch_balance(self):
        if "fetch_balance" in self._raises:
            raise self._raises["fetch_balance"]
        return self._balance

    def set_sandbox_mode(self, on):
        self.sandbox = on

    async def close(self):
        self.closed = True


@pytest.fixture
def patched(monkeypatch):
    """Install a fake ccxt.async_support exchange class."""
    holder = {}

    def install(exchange):
        holder["exchange"] = exchange
        import ccxt.async_support as accxt
        monkeypatch.setattr(accxt, "binance", lambda *a, **k: exchange, raising=False)
        return exchange

    return install


def test_public_check_needs_no_credentials_and_reports_a_price(patched):
    """The check that settles 'is this talking to Binance' before any funding."""
    patched(_FakeExchange(
        markets={"BTC/USDT": {}, "ETH/USDT": {}},
        ticker={"last": 64000.0, "bid": 63999.0, "ask": 64001.0},
    ))
    events = EventLog()
    results = asyncio.run(VenueProbe(events).check_public("binance", "BTC/USDT"))

    assert [r.name for r in results] == ["reachable", "market_data"]
    assert all(r.ok for r in results)
    assert results[1].data["price"] == 64000.0
    assert any("64,000" in e["message"] for e in events.entries())


def test_an_empty_account_is_a_pass(patched):
    """A zero balance proves the credential just as well as a funded one.

    This is the whole point of the module: verifying the key must not require
    depositing first.
    """
    patched(_FakeExchange(
        markets={"BTC/USDT": {}},
        ticker={"last": 1.0},
        balance={"total": {}, "free": {}},
    ))
    events = EventLog()
    results = asyncio.run(VenueProbe(events).check_credentials(_credential()))

    credentials = results[-1]
    assert credentials.name == "credentials"
    assert credentials.ok is True
    assert "still works" in credentials.detail


def test_a_rejected_key_reports_the_remedy_not_just_the_failure(patched):
    patched(_FakeExchange(
        markets={"BTC/USDT": {}},
        ticker={"last": 1.0},
        raises={"fetch_balance": ccxt.AuthenticationError(
            'binance {"code":-2015,"msg":"Invalid API-key, IP, or permissions for action."}'
        )},
    ))
    events = EventLog()
    results = asyncio.run(VenueProbe(events).check_credentials(_credential()))

    credentials = results[-1]
    assert credentials.ok is False
    assert credentials.kind == "ip_not_allowed"
    assert any(e["level"] == "error" for e in events.entries())


def test_an_unreachable_venue_does_not_get_blamed_on_the_key(patched):
    """Testing a key against a venue we cannot reach would report an auth
    failure that is nothing of the sort."""
    patched(_FakeExchange(raises={"load_markets": ccxt.NetworkError("down")}))
    results = asyncio.run(VenueProbe(EventLog()).check_credentials(_credential()))

    assert [r.name for r in results] == ["reachable"]
    assert not any(r.name == "credentials" for r in results)


def test_testnet_credentials_switch_the_client_into_sandbox(patched):
    """Binance's test network is the one way to exercise the order path for free."""
    exchange = patched(_FakeExchange(
        markets={"BTC/USDT": {}}, ticker={"last": 1.0},
        balance={"total": {"USDT": 10_000.0}, "free": {"USDT": 10_000.0}},
    ))
    results = asyncio.run(
        VenueProbe(EventLog()).check_credentials(_credential(testnet=True))
    )
    assert exchange.sandbox is True
    assert results[-1].ok is True
    assert results[-1].data["testnet"] is True


def test_the_session_is_closed_even_when_a_probe_fails(patched):
    """An unclosed aiohttp session leaks a connector per probe."""
    exchange = patched(_FakeExchange(raises={"load_markets": ccxt.NetworkError("x")}))
    asyncio.run(VenueProbe(EventLog()).check_public("binance"))
    assert exchange.closed is True


def test_an_unknown_exchange_is_reported_not_raised():
    results = asyncio.run(VenueProbe(EventLog()).check_public("not_a_venue"))
    assert results[0].ok is False
    assert results[0].kind == "bad_exchange"


def test_no_probe_can_place_an_order():
    """Structural, not conventional: nothing in the module takes a side or size."""
    import inspect

    from godalgo.ui import venue

    source = inspect.getsource(venue)
    for forbidden in ("create_order", "cancel_order", "create_limit", "create_market"):
        assert forbidden not in source
