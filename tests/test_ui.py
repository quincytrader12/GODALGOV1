"""UI backend: position reconstruction, journal, credentials, and the server.

The security assertions here are the important ones. This process holds keys
that can move money and serves HTTP with no authentication, so "secrets never
leave the machine" and "loopback only" are tested rather than trusted.
"""

import json
import stat
import sys
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from godalgo.ui.credentials import CredentialStore, ExchangeCredential
from godalgo.ui.journal import TradingJournal
from godalgo.ui.server import UIBridge, _assert_loopback, create_app
from godalgo.ui.state import PositionTracker, TerminalHealth
from godalgo.ui.telegram import TelegramNotifier

T0 = datetime(2024, 1, 1, tzinfo=UTC)


# --- position reconstruction ----------------------------------------------

def test_round_trip_realises_pnl():
    t = PositionTracker()
    t.on_fill("BTC/USDT", 1.0, 100.0, moment=T0)
    closed = t.on_fill("BTC/USDT", -1.0, 110.0, moment=T0 + timedelta(hours=1))
    assert closed is not None
    assert closed.realised_pnl == pytest.approx(10.0)
    assert closed.state == "profit"


def test_short_profits_when_price_falls():
    t = PositionTracker()
    t.on_fill("BTC/USDT", -1.0, 100.0, moment=T0)
    closed = t.on_fill("BTC/USDT", 1.0, 90.0, moment=T0 + timedelta(hours=1))
    assert closed.realised_pnl == pytest.approx(10.0)
    assert closed.state == "profit"


def test_scaling_in_reaverages_the_entry():
    """A weight that rises is one position scaled into, not two positions."""
    t = PositionTracker()
    t.on_fill("BTC/USDT", 1.0, 100.0, moment=T0)
    t.on_fill("BTC/USDT", 1.0, 200.0, moment=T0)
    position = t.open_positions["BTC/USDT"]
    assert position.quantity == pytest.approx(2.0)
    assert position.entry_price == pytest.approx(150.0)
    assert not t.closed_positions


def test_partial_close_realises_only_the_closed_portion():
    """Marking the whole position at the exit would book unearned profit."""
    t = PositionTracker()
    t.on_fill("BTC/USDT", 2.0, 100.0, moment=T0)
    assert t.on_fill("BTC/USDT", -1.0, 110.0, moment=T0) is None
    position = t.open_positions["BTC/USDT"]
    assert position.realised_pnl == pytest.approx(10.0)
    assert position.quantity == pytest.approx(1.0)


def test_a_fill_through_flat_closes_and_reopens_the_other_way():
    t = PositionTracker()
    t.on_fill("BTC/USDT", 1.0, 100.0, moment=T0)
    closed = t.on_fill("BTC/USDT", -3.0, 90.0, moment=T0)
    assert closed is not None and closed.state == "loss"
    flipped = t.open_positions["BTC/USDT"]
    assert flipped.side == "short"
    assert flipped.quantity == pytest.approx(2.0)


def test_unrealised_marks_to_current_price_not_last_fill():
    t = PositionTracker()
    t.on_fill("BTC/USDT", 1.0, 100.0, moment=T0)
    t.mark("BTC/USDT", 130.0)
    assert t.open_positions["BTC/USDT"].unrealised_pnl == pytest.approx(30.0)


def test_open_positions_are_not_coloured_by_running_pnl():
    """An open position has not made or lost anything yet."""
    t = PositionTracker()
    t.on_fill("BTC/USDT", 1.0, 100.0, moment=T0)
    t.mark("BTC/USDT", 500.0)
    assert t.open_positions["BTC/USDT"].state == "open"


def test_fees_reduce_net_pnl():
    t = PositionTracker()
    t.on_fill("BTC/USDT", 1.0, 100.0, fee=3.0, moment=T0)
    closed = t.on_fill("BTC/USDT", -1.0, 110.0, fee=3.0, moment=T0)
    assert closed.realised_pnl == pytest.approx(10.0)
    assert closed.net_pnl == pytest.approx(4.0)


def test_closed_history_is_capped():
    t = PositionTracker(max_closed=5)
    for i in range(20):
        t.on_fill("BTC/USDT", 1.0, 100.0, moment=T0)
        t.on_fill("BTC/USDT", -1.0, 101.0 + i, moment=T0)
    assert len(t.closed_positions) == 5


def test_profit_factor_reports_infinity_rather_than_a_flattering_number():
    t = PositionTracker()
    t.on_fill("BTC/USDT", 1.0, 100.0, moment=T0)
    t.on_fill("BTC/USDT", -1.0, 110.0, moment=T0)
    assert t.profit_factor == float("inf")


# --- health ----------------------------------------------------------------

def _health(**kw):
    base = {"connected": True, "data_age_seconds": 1.0, "halted": False,
            "halt_reason": None, "reconnects": 0, "errors": 0,
            "decisions_run": 10, "drawdown": 0.0}
    return TerminalHealth(**{**base, **kw})


def test_healthy_terminal_scores_nominal():
    h = _health()
    assert h.score > 0.9 and h.label == "NOMINAL"


def test_halt_zeroes_the_score():
    h = _health(halted=True, halt_reason="drawdown")
    assert h.score == 0.0 and h.label == "HALTED"


def test_stale_data_dominates_the_score():
    """The failure that looks most like normality gets the heaviest penalty."""
    assert _health(data_age_seconds=600).score < _health(errors=3).score


def test_drawdown_degrades_health():
    assert _health(drawdown=0.2).score < _health(drawdown=0.0).score


# --- journal ---------------------------------------------------------------

def _journal(tmp_path):
    return TradingJournal(path=tmp_path / "j.jsonl", summary_path=tmp_path / "s.jsonl")


def _closed(tracker, pnl, symbol="BTC/USDT"):
    tracker.on_fill(symbol, 1.0, 100.0, moment=T0)
    return tracker.on_fill(symbol, -1.0, 100.0 + pnl, moment=T0 + timedelta(hours=1))


def test_journal_round_trips(tmp_path):
    j = _journal(tmp_path)
    t = PositionTracker()
    j.record(_closed(t, 12.0), equity=10_012.0)
    entries = j.entries()
    assert len(entries) == 1
    assert entries[0]["net_pnl"] == pytest.approx(12.0)


def test_daily_summary_aggregates(tmp_path):
    j = _journal(tmp_path)
    t = PositionTracker()
    for pnl in (10.0, -4.0, 6.0):
        j.record(_closed(t, pnl))
    s = j.summarise(T0.date(), 1000.0, 1012.0)
    assert s.trades == 3 and s.wins == 2 and s.losses == 1
    assert s.net_pnl == pytest.approx(12.0)
    assert s.best_trade == pytest.approx(10.0)
    assert s.worst_trade == pytest.approx(-4.0)


def test_rollover_only_fires_on_a_new_utc_day(tmp_path):
    j = _journal(tmp_path)
    # The first call establishes a baseline; rolling up a day the process did
    # not observe would report a partial day as a whole one.
    assert j.check_rollover(1000.0, now=T0) is None
    assert j.check_rollover(1000.0, now=T0 + timedelta(hours=6)) is None
    summary = j.check_rollover(1050.0, now=T0 + timedelta(days=1))
    assert summary is not None and summary.day == T0.date()
    assert j.summaries()


def test_malformed_journal_line_does_not_lose_the_file(tmp_path):
    j = _journal(tmp_path)
    t = PositionTracker()
    j.record(_closed(t, 5.0))
    with j.path.open("a") as fh:
        fh.write("{ this is not json\n")
    assert len(j.entries()) == 1


def test_telegram_summary_is_plain_text():
    from godalgo.ui.journal import DailySummary
    s = DailySummary(day=T0.date(), trades=2, wins=1, losses=1, gross_profit=10.0,
                     gross_loss=4.0, fees=0.5, net_pnl=6.0, best_trade=10.0,
                     worst_trade=-4.0, symbols=("BTC/USDT",), equity_start=1000.0,
                     equity_end=1006.0)
    text = s.as_telegram()
    assert "GODALGO daily" in text and "+6.00" in text


# --- credentials -----------------------------------------------------------

def test_credentials_are_stored_owner_only(tmp_path):
    """The store must be unreadable by other accounts, on every platform.

    Deliberately asserted through ``protection()`` rather than by inspecting
    mode bits directly. Windows has no POSIX permission bits: os.open with a
    mode sets at most the read-only flag and stat() reports 0o666 whatever was
    requested, so a mode-bit assertion fails there even when the file is
    correctly ACL-restricted -- and, worse, would pass on a Windows file that
    is genuinely readable by everyone if the check were simply relaxed.
    """
    store = CredentialStore(directory=tmp_path)
    store.add(ExchangeCredential("binance", "KEY1234567890", "SECRET_VALUE"))

    protection = store.protection()
    assert protection["exists"] is True
    assert protection["restricted"] is True, (
        f"credential file is readable by others: {protection['detail']}"
    )


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits only")
def test_posix_store_is_mode_600(tmp_path):
    """The POSIX half of the guarantee, checked concretely."""
    store = CredentialStore(directory=tmp_path)
    store.add(ExchangeCredential("binance", "KEY1234567890", "SECRET_VALUE"))
    mode = stat.S_IMODE(store.path.stat().st_mode)
    assert not mode & 0o077, f"mode {oct(mode)}"


def test_protection_reports_when_there_is_no_store(tmp_path):
    assert CredentialStore(directory=tmp_path).protection()["exists"] is False


def test_listing_never_contains_secrets(tmp_path):
    """This listing is served over HTTP and rendered in a browser."""
    store = CredentialStore(directory=tmp_path)
    store.add(ExchangeCredential("binance", "KEY1234567890", "SUPER_SECRET", label="x"))
    blob = json.dumps(store.listing())
    assert "SUPER_SECRET" not in blob
    assert "KEY1234567890" not in blob
    assert "KEY1" in blob and "7890" in blob   # enough to identify, not to use


def test_trading_is_off_by_default(tmp_path):
    """Pasting a key into a form must not by itself authorise orders."""
    store = CredentialStore(directory=tmp_path)
    store.add(ExchangeCredential("binance", "K" * 20, "S" * 20, label="x"))
    assert store.listing()[0]["trade_enabled"] is False


def test_credentials_survive_a_reload(tmp_path):
    store = CredentialStore(directory=tmp_path)
    store.add(ExchangeCredential("kraken", "K" * 20, "S" * 20, label="k"))
    assert CredentialStore(directory=tmp_path).get("k").api_secret == "S" * 20


def test_removal(tmp_path):
    store = CredentialStore(directory=tmp_path)
    store.add(ExchangeCredential("kraken", "K" * 20, "S" * 20, label="k"))
    assert store.remove("k") is True
    assert store.remove("k") is False


# --- telegram --------------------------------------------------------------

def test_telegram_status_leaks_nothing(monkeypatch):
    monkeypatch.setenv("GODALGO_TELEGRAM_TOKEN", "123456:AAAA_SECRET_TOKEN")
    monkeypatch.setenv("GODALGO_TELEGRAM_CHAT_ID", "987654321")
    status = TelegramNotifier().status
    assert "AAAA_SECRET_TOKEN" not in json.dumps(status)
    assert status["configured"] is True
    assert status["chat_id_tail"] == "4321"


def test_unconfigured_telegram_never_raises(monkeypatch):
    """A notifier that can take down the trading loop is a liability."""
    import asyncio

    monkeypatch.delenv("GODALGO_TELEGRAM_TOKEN", raising=False)
    monkeypatch.delenv("GODALGO_TELEGRAM_CHAT_ID", raising=False)
    assert asyncio.run(TelegramNotifier().send("hi")) is False


# --- server ----------------------------------------------------------------

def test_only_loopback_binds_are_permitted():
    for host in ("127.0.0.1", "localhost", "::1"):
        _assert_loopback(host)
    for host in ("0.0.0.0", "192.168.1.5", "10.0.0.2", "example.com"):
        with pytest.raises(ValueError, match="refusing to bind"):
            _assert_loopback(host)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.delenv("GODALGO_TELEGRAM_TOKEN", raising=False)
    bridge = UIBridge(
        journal=TradingJournal(path=tmp_path / "j.jsonl", summary_path=tmp_path / "s.jsonl"),
        credentials=CredentialStore(directory=tmp_path),
        # No real network call from the suite: the poller would make these
        # tests depend on an exchange being up.
        market_feed_enabled=False,
    )
    return TestClient(create_app(bridge)), bridge


def test_state_endpoint_shape(client):
    c, _ = client
    body = c.get("/api/state").json()
    assert {"timestamp", "neurons", "health", "pnl", "brain"} <= set(body)
    assert {"score", "label"} <= set(body["health"])


def test_profit_factor_is_zero_before_any_trade_closes(client):
    c, _ = client
    assert c.get("/api/state").json()["pnl"]["profit_factor"] == 0.0


def test_infinite_profit_factor_serialises_as_null(client):
    """All wins and no losses gives infinity, which is not valid JSON.

    Serialising it raw produces `Infinity`, which JSON.parse rejects -- the
    websocket frame fails and the whole terminal stops updating.
    """
    c, bridge = client
    bridge.record_fill("BTC/USDT", 1.0, 100.0, moment=T0)
    bridge.record_fill("BTC/USDT", -1.0, 110.0, moment=T0 + timedelta(hours=1))
    body = c.get("/api/state")
    assert body.json()["pnl"]["profit_factor"] is None
    json.loads(body.text)   # would raise if Infinity leaked through


def test_neuron_appears_after_a_fill(client):
    c, bridge = client
    bridge.record_fill("BTC/USDT", 1.0, 100.0, moment=T0)
    neurons = c.get("/api/state").json()["neurons"]
    assert len(neurons) == 1
    assert neurons[0]["symbol"] == "BTC/USDT" and neurons[0]["state"] == "open"


def test_position_detail_endpoint(client):
    c, bridge = client
    bridge.record_fill("BTC/USDT", 1.0, 100.0, moment=T0)
    nid = c.get("/api/state").json()["neurons"][0]["id"]
    assert c.get(f"/api/positions/{nid}").json()["symbol"] == "BTC/USDT"
    assert c.get("/api/positions/nope").status_code == 404


def test_connections_endpoint_never_returns_key_material(client):
    c, _ = client
    c.post("/api/connections", json={
        "exchange_id": "binance", "label": "main",
        "api_key": "KEY1234567890", "api_secret": "SUPER_SECRET",
    })
    body = c.get("/api/connections").text
    assert "SUPER_SECRET" not in body
    assert "KEY1234567890" not in body


def test_connection_requires_all_fields(client):
    c, _ = client
    r = c.post("/api/connections", json={"exchange_id": "binance"})
    assert r.status_code == 400
    assert "api_key" in r.json()["detail"]


def test_connection_delete(client):
    c, _ = client
    c.post("/api/connections", json={
        "exchange_id": "kraken", "label": "k", "api_key": "K" * 20, "api_secret": "S" * 20,
    })
    assert c.delete("/api/connections/k").status_code == 200
    assert c.delete("/api/connections/k").status_code == 404


def test_journal_endpoint(client):
    c, bridge = client
    bridge.record_fill("BTC/USDT", 1.0, 100.0, moment=T0)
    bridge.record_fill("BTC/USDT", -1.0, 110.0, moment=T0 + timedelta(hours=1))
    body = c.get("/api/journal").json()
    assert len(body["entries"]) == 1
    assert body["entries"][0]["net_pnl"] == pytest.approx(10.0)


def test_index_is_served(client):
    c, _ = client
    r = c.get("/")
    assert r.status_code == 200
    assert "GODALGO" in r.text


# --- proof of life ---------------------------------------------------------
#
# The defect these cover: connecting an API key produced no visible change
# anywhere in the interface, because the UI's credential store and the trading
# path were two disconnected systems. Saving a key wrote a file nothing read.


def test_saving_a_key_immediately_tests_it(client, monkeypatch):
    """Saving a key and seeing nothing happen is indistinguishable from the
    key being ignored -- which is exactly what used to happen."""
    c, _ = client
    _fake_venue(monkeypatch)

    body = c.post("/api/connections", json={
        "exchange_id": "binance", "api_key": "k", "api_secret": "s",
    }).json()

    assert body["ok"] is True
    names = [check["name"] for check in body["checks"]]
    assert names == ["reachable", "market_data", "credentials"]
    assert all(check["ok"] for check in body["checks"])


def test_saving_a_key_writes_to_the_event_log(client, monkeypatch):
    c, bridge = client
    _fake_venue(monkeypatch)
    c.post("/api/connections", json={
        "exchange_id": "binance", "api_key": "k", "api_secret": "s",
    })

    messages = [e["message"] for e in bridge.events.entries()]
    assert any("added binance key" in m for m in messages)
    assert any("verified" in m for m in messages)


def test_a_saved_key_is_never_returned_over_http(client, monkeypatch):
    """The whole store is served masked; the browser must never see key
    material, in any response."""
    c, _ = client
    _fake_venue(monkeypatch)
    secret = "SUPER_SECRET_VALUE_12345"

    save = c.post("/api/connections", json={
        "exchange_id": "binance", "api_key": "KEYVALUE_98765", "api_secret": secret,
    })
    for response in (save, c.get("/api/connections"), c.get("/api/state"),
                     c.get("/api/events")):
        assert secret not in response.text
        assert "KEYVALUE_98765" not in response.text


def test_the_read_only_venue_check_needs_no_key(client, monkeypatch):
    """Settles 'is this actually reaching Binance' before any decision to fund."""
    c, _ = client
    _fake_venue(monkeypatch)

    body = c.post("/api/venue/check", json={"exchange_id": "binance"}).json()
    assert body["ok"] is True
    assert body["credential_tested"] is False
    assert body["checks"][1]["data"]["price"] == 64000.0


def test_venue_failures_surface_as_events_not_errors(client, monkeypatch):
    """A venue outage must be a red lamp, never a 500."""
    import ccxt
    c, bridge = client
    _fake_venue(monkeypatch, raises={"load_markets": ccxt.NetworkError("down")})

    response = c.post("/api/venue/check", json={"exchange_id": "binance"})
    assert response.status_code == 200
    assert response.json()["ok"] is False
    assert any(e["level"] == "error" for e in bridge.events.entries())


def test_trade_permission_is_a_separate_deliberate_action(client, monkeypatch):
    """Consent to place orders is recorded per key, and is not bundled into
    saving one."""
    c, bridge = client
    _fake_venue(monkeypatch)
    c.post("/api/connections", json={
        "exchange_id": "binance", "api_key": "k", "api_secret": "s",
    })
    assert bridge.credentials.tradeable() is None

    assert c.post("/api/connections/binance/trade-enabled",
                  json={"enabled": True}).status_code == 200
    assert bridge.credentials.tradeable() is not None

    assert c.post("/api/connections/binance/trade-enabled",
                  json={"enabled": False}).status_code == 200
    assert bridge.credentials.tradeable() is None


def test_events_are_carried_in_the_state_snapshot(client, monkeypatch):
    """Carried rather than polled separately, so the log cannot drift out of
    step with the state it describes."""
    c, bridge = client
    bridge.events.good("venue", "binance reachable", "1200 markets")

    body = c.get("/api/state").json()
    assert body["events"][0]["message"] == "binance reachable"
    assert "venue" in body


def test_the_events_endpoint_supports_incremental_reads(client):
    c, bridge = client
    bridge.events.info("venue", "first")
    mark = c.get("/api/events").json()["latest"]
    bridge.events.info("venue", "second")

    body = c.get(f"/api/events?since={mark}").json()
    assert [e["message"] for e in body["events"]] == ["second"]


def test_telegram_is_configurable_from_the_terminal(client, monkeypatch):
    """Environment-only was a wall for a double-clicked executable."""
    c, bridge = client

    async def fake_verify():
        return True, "connected as @godalgo_bot"

    async def fake_send(text, **kwargs):
        return True

    monkeypatch.setattr(bridge.telegram, "verify", fake_verify)
    monkeypatch.setattr(bridge.telegram, "send", fake_send)

    body = c.post("/api/telegram", json={"token": "123:ABC", "chat_id": "555"}).json()
    assert body["ok"] is True
    assert bridge.telegram.configured is True


def test_the_telegram_token_never_leaves_the_process(client, monkeypatch):
    """The send URL embeds the token, so neither it nor the URL may appear in
    any response or event."""
    c, bridge = client

    async def fake_verify():
        return True, "connected"

    async def fake_send(text, **kwargs):
        return True

    monkeypatch.setattr(bridge.telegram, "verify", fake_verify)
    monkeypatch.setattr(bridge.telegram, "send", fake_send)

    token = "7654321:AAHtokenvalue"
    save = c.post("/api/telegram", json={"token": token, "chat_id": "5551234"})
    for response in (save, c.get("/api/connections"), c.get("/api/events"),
                     c.get("/api/state")):
        assert token not in response.text
    # The chat id is shown only as a tail, enough to identify it.
    assert c.get("/api/connections").json()["telegram"]["chat_id_tail"] == "1234"


def _fake_venue(monkeypatch, *, raises=None):
    """Point ccxt.async_support.binance at a stub, so no test needs a network."""
    import ccxt.async_support as accxt

    class _Fake:
        def __init__(self, *a, **k):
            pass

        async def load_markets(self):
            if raises and "load_markets" in raises:
                raise raises["load_markets"]
            return {"BTC/USDT": {}}

        async def fetch_ticker(self, symbol):
            if raises and "fetch_ticker" in raises:
                raise raises["fetch_ticker"]
            return {"last": 64000.0, "bid": 63999.0, "ask": 64001.0}

        async def fetch_balance(self):
            if raises and "fetch_balance" in raises:
                raise raises["fetch_balance"]
            return {"total": {}, "free": {}}

        def set_sandbox_mode(self, on):
            pass

        async def close(self):
            pass

    monkeypatch.setattr(accxt, "binance", _Fake, raising=False)
