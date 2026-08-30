"""Mode switching.

Swapping the broker under a running engine is the most dangerous operation in
the system: an order is bounded, but a bad mode switch silently invalidates
every position the engine believes it holds. These tests are almost entirely
about refusal.
"""

import asyncio

import pytest
from fastapi.testclient import TestClient

from godalgo.execution.broker import DryRunBroker, PaperBroker
from godalgo.execution.mode import ModeController, ModeSwitchError
from godalgo.execution.types import TradingMode
from godalgo.ui.server import UIBridge, create_app

ARM = "I_UNDERSTAND_THIS_TRADES_REAL_MONEY"


@pytest.fixture(autouse=True)
def unarmed(monkeypatch):
    """Every test starts from an unarmed environment unless it says otherwise."""
    monkeypatch.delenv("GODALGO_ARM_LIVE", raising=False)
    monkeypatch.delenv("GODALGO_API_KEY", raising=False)
    monkeypatch.delenv("GODALGO_API_SECRET", raising=False)


# --- defaults --------------------------------------------------------------

def test_starts_in_dry_run():
    """The harmless mode is the default; escalation is always deliberate."""
    c = ModeController()
    assert c.mode is TradingMode.DRY_RUN
    assert isinstance(c.broker, DryRunBroker)


def test_simulated_modes_switch_freely():
    c = ModeController()
    asyncio.run(c.switch(TradingMode.PAPER))
    assert c.mode is TradingMode.PAPER
    assert isinstance(c.broker, PaperBroker)


def test_switching_to_the_current_mode_is_a_noop():
    c = ModeController()
    change = asyncio.run(c.switch(TradingMode.DRY_RUN))
    assert change.note == "no change"
    assert c.history == []


# --- live gates ------------------------------------------------------------

def test_live_refused_without_arming():
    """The interface can request live; it cannot authorise it."""
    c = ModeController()
    with pytest.raises(ModeSwitchError, match="GODALGO_ARM_LIVE"):
        asyncio.run(c.switch(TradingMode.LIVE, confirm="GO LIVE"))


def test_live_refused_without_credentials(monkeypatch):
    monkeypatch.setenv("GODALGO_ARM_LIVE", ARM)
    c = ModeController()
    with pytest.raises(ModeSwitchError, match="GODALGO_API_KEY"):
        asyncio.run(c.switch(TradingMode.LIVE, confirm="GO LIVE"))


def test_live_refused_without_the_confirmation_phrase(monkeypatch):
    """A button that arms real money on one click arms it by accident."""
    monkeypatch.setenv("GODALGO_ARM_LIVE", ARM)
    monkeypatch.setenv("GODALGO_API_KEY", "k")
    monkeypatch.setenv("GODALGO_API_SECRET", "s")
    c = ModeController()
    for attempt in (None, "", "yes", "go live please"):
        with pytest.raises(ModeSwitchError, match="confirmation phrase"):
            asyncio.run(c.switch(TradingMode.LIVE, confirm=attempt))


def test_confirmation_is_forgiving_about_case_only(monkeypatch):
    """Case and whitespace are forgiven; content is not.

    Typing the phrase is the deliberate act, and punishing capitalisation adds
    friction without adding safety.
    """
    monkeypatch.setenv("GODALGO_ARM_LIVE", ARM)
    monkeypatch.setenv("GODALGO_API_KEY", "k")
    monkeypatch.setenv("GODALGO_API_SECRET", "s")
    controller = ModeController()

    # Reaches the broker build, which proves the phrase itself was accepted.
    controller._assert_live_permitted("  go live  ")

    with pytest.raises(ModeSwitchError, match="confirmation phrase"):
        controller._assert_live_permitted("golive")


def test_credentials_alone_do_not_imply_consent(monkeypatch):
    """The presence of an API key is not consent to use it."""
    monkeypatch.setenv("GODALGO_API_KEY", "k")
    monkeypatch.setenv("GODALGO_API_SECRET", "s")
    c = ModeController()
    assert c.live_armed is False
    assert c.status()["live_available"] is False


def test_status_reports_capability_without_secrets(monkeypatch):
    import json

    monkeypatch.setenv("GODALGO_ARM_LIVE", ARM)
    monkeypatch.setenv("GODALGO_API_KEY", "SECRET_KEY_VALUE")
    monkeypatch.setenv("GODALGO_API_SECRET", "SECRET_SECRET_VALUE")
    status = ModeController().status()
    blob = json.dumps(status)
    assert "SECRET_KEY_VALUE" not in blob
    assert "SECRET_SECRET_VALUE" not in blob
    assert status["live_armed"] is True


# --- flattening ------------------------------------------------------------

def test_every_switch_flattens_first():
    """A paper position is not a real one.

    Switching while holding one leaves the engine convinced it owns something
    the venue has never heard of.
    """
    calls = []

    async def flatten():
        calls.append("flattened")

    c = ModeController(flatten=flatten)
    change = asyncio.run(c.switch(TradingMode.PAPER))
    assert calls == ["flattened"]
    assert change.flattened is True


def test_a_failed_flatten_refuses_the_switch():
    """Staying put keeps whatever is open under the broker that owns it."""
    async def flatten():
        raise RuntimeError("venue down")

    c = ModeController(flatten=flatten)
    with pytest.raises(ModeSwitchError, match="could not be closed"):
        asyncio.run(c.switch(TradingMode.PAPER))
    assert c.mode is TradingMode.DRY_RUN, "mode changed despite a failed flatten"


def test_a_synchronous_flatten_is_accepted():
    calls = []
    c = ModeController(flatten=lambda: calls.append("x"))
    asyncio.run(c.switch(TradingMode.PAPER))
    assert calls == ["x"]


# --- rebinding and audit ---------------------------------------------------

def test_the_new_broker_is_handed_back():
    seen = []
    c = ModeController(on_broker_change=seen.append)
    asyncio.run(c.switch(TradingMode.PAPER))
    assert len(seen) == 1
    assert isinstance(seen[0], PaperBroker)
    assert seen[0] is c.broker


def test_every_change_is_recorded():
    c = ModeController()
    asyncio.run(c.switch(TradingMode.PAPER))
    asyncio.run(c.switch(TradingMode.DRY_RUN))
    assert [h.to_dict()["to"] for h in c.history] == ["paper", "dry_run"]


# --- HTTP surface ----------------------------------------------------------

@pytest.fixture
def client(tmp_path):
    from godalgo.ui.credentials import CredentialStore
    from godalgo.ui.journal import TradingJournal

    bridge = UIBridge(
        journal=TradingJournal(path=tmp_path / "j.jsonl", summary_path=tmp_path / "s.jsonl"),
        credentials=CredentialStore(directory=tmp_path),
        mode_controller=ModeController(),
    )
    return TestClient(create_app(bridge)), bridge


def test_mode_endpoint_reports_state(client):
    c, _ = client
    body = c.get("/api/mode").json()
    assert body["available"] is True
    assert body["mode"] == "dry_run"
    assert body["live_armed"] is False


def test_switching_to_paper_over_http(client):
    c, bridge = client
    r = c.post("/api/mode", json={"mode": "paper"})
    assert r.status_code == 200
    assert r.json()["mode"] == "paper"
    assert bridge.mode == "paper"


def test_live_over_http_is_refused_with_conflict(client):
    """409 rather than 400: the request is well-formed, the environment does
    not permit it."""
    c, _ = client
    r = c.post("/api/mode", json={"mode": "live", "confirm": "GO LIVE"})
    assert r.status_code == 409
    assert "GODALGO_ARM_LIVE" in r.json()["detail"]


def test_unknown_mode_is_rejected(client):
    c, _ = client
    r = c.post("/api/mode", json={"mode": "turbo"})
    assert r.status_code == 400
    assert "turbo" in r.json()["detail"]


def test_mode_control_unavailable_without_a_session(tmp_path):
    """A control that silently does nothing is worse than one that says so."""
    from godalgo.ui.credentials import CredentialStore
    from godalgo.ui.journal import TradingJournal

    bridge = UIBridge(
        journal=TradingJournal(path=tmp_path / "j.jsonl", summary_path=tmp_path / "s.jsonl"),
        credentials=CredentialStore(directory=tmp_path),
    )
    c = TestClient(create_app(bridge))
    assert c.get("/api/mode").json()["available"] is False
    assert c.post("/api/mode", json={"mode": "paper"}).status_code == 409
