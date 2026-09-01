"""That the terminal is actually assembled, not just assemblable.

Every bug these cover shipped. The plumbing existed and was unit-tested in
isolation; no entry point was connected to it, and nothing failed until someone
pressed the button:

* ``run-terminal.py`` -- the file the packaged executable runs -- built a
  bridge with no ``ModeController`` at all, so every mode button was disabled
  and LIVE could never be reached.
* ``cmd_ui`` built a controller without ``tradeable_credential``, so the
  credential store was never consulted and live stayed unavailable however the
  key was configured.
* Nothing attached an engine, so a successful mode switch changed a label and
  traded nothing.

So these assert the wiring itself, from the entry points inward.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from godalgo.execution.types import TradingMode
from godalgo.ui.credentials import CredentialStore, ExchangeCredential
from godalgo.ui.journal import TradingJournal
from godalgo.ui.server import build_terminal, create_app


@pytest.fixture
def terminal(tmp_path, monkeypatch):
    monkeypatch.delenv("GODALGO_ARM_LIVE", raising=False)
    monkeypatch.delenv("GODALGO_API_KEY", raising=False)
    monkeypatch.delenv("GODALGO_API_SECRET", raising=False)

    bridge = build_terminal(symbol="BTC/USDT", equity=1000.0)
    bridge.credentials = CredentialStore(directory=tmp_path)
    bridge.journal = TradingJournal(
        path=tmp_path / "j.jsonl", summary_path=tmp_path / "s.jsonl"
    )
    bridge.market_feed_enabled = False
    # Rebind the controller to the swapped-in store, as build_terminal does.
    bridge.mode_controller.tradeable_credential = bridge.credentials.tradeable
    return TestClient(create_app(bridge)), bridge


def _add_key(bridge, *, trade_enabled=False, testnet=False, exchange="binance"):
    bridge.credentials.add(ExchangeCredential(
        exchange_id=exchange, api_key="k", api_secret="s", label=exchange,
        trade_enabled=trade_enabled, testnet=testnet,
    ))


# --- the controller exists at all ------------------------------------------

def test_the_terminal_has_a_mode_controller():
    """It did not. The packaged executable reported every mode unavailable,
    which is indistinguishable from a broken build."""
    assert build_terminal().mode_controller is not None


def test_the_mode_endpoint_reports_itself_available(terminal):
    c, _ = terminal
    assert c.get("/api/mode").json()["available"] is True


def test_the_controller_consults_the_credential_store(terminal):
    """The other half of the same bug: a controller that exists but reads
    nothing is a LIVE button that can never light up."""
    _, bridge = terminal
    assert bridge.mode_controller.tradeable_credential is not None
    assert bridge.mode_controller._credential() is None

    _add_key(bridge, trade_enabled=True)
    assert bridge.mode_controller._credential() is not None


def test_a_switch_can_flatten(terminal):
    """Without a flatten hook a mode switch strands whatever is open under a
    broker nothing holds any more."""
    _, bridge = terminal
    assert bridge.mode_controller.flatten is not None


# --- live becomes available, and only for the right reason -----------------

def test_live_is_unavailable_with_no_key(terminal):
    c, _ = terminal
    body = c.get("/api/mode").json()
    assert body["live_available"] is False
    assert "allow this key to place orders" in body["live_blockers"][0]


def test_live_stays_unavailable_for_a_stored_but_unticked_key(terminal):
    """Storing a key is not consent to trade with it."""
    c, bridge = terminal
    _add_key(bridge, trade_enabled=False)
    assert c.get("/api/mode").json()["live_available"] is False


def test_ticking_the_key_makes_live_available(terminal):
    """The end of the chain the user actually presses."""
    c, bridge = terminal
    _add_key(bridge, trade_enabled=True)

    body = c.get("/api/mode").json()
    assert body["live_available"] is True
    assert body["live_blockers"] == []
    assert body["live_source"] == "binance"


def test_the_confirmation_phrase_is_still_required(terminal):
    """Making live reachable must not make it accidental."""
    c, bridge = terminal
    _add_key(bridge, trade_enabled=True)

    r = c.post("/api/mode", json={"mode": "live"})
    assert r.status_code == 409
    assert "GO LIVE" in r.json()["detail"]


def test_a_testnet_key_is_reported_as_testnet(terminal):
    """The modal says REAL money or TESTNET off this flag; getting it backwards
    is the worst possible label on that dialog."""
    c, bridge = terminal
    _add_key(bridge, trade_enabled=True, testnet=True)
    assert c.get("/api/mode").json()["live_testnet"] is True


# --- the session exists and the switch drives it ---------------------------

def test_a_trading_session_is_attached(terminal):
    """Otherwise a successful mode switch changes a label and trades nothing."""
    c, bridge = terminal
    assert bridge.session is not None
    assert c.get("/api/session").json()["attached"] is True


def test_switching_mode_rebrokers_the_session(terminal, monkeypatch):
    c, bridge = terminal
    started: list[str] = []

    async def fake_start(broker):
        started.append(type(broker).__name__)

    async def fake_stop():
        pass

    monkeypatch.setattr(bridge.session, "start", fake_start)
    monkeypatch.setattr(bridge.session, "stop", fake_stop)

    assert c.post("/api/mode", json={"mode": "paper"}).status_code == 200
    assert started == ["PaperBroker"]


def test_a_session_that_fails_to_restart_is_reported_not_swallowed(
    terminal, monkeypatch
):
    """A mode switch that half-worked must not look like one that worked."""
    c, bridge = terminal

    async def boom(broker):
        raise RuntimeError("no data feed")

    async def fake_stop():
        pass

    monkeypatch.setattr(bridge.session, "start", boom)
    monkeypatch.setattr(bridge.session, "stop", fake_stop)

    assert c.post("/api/mode", json={"mode": "paper"}).status_code == 200
    messages = [e["message"] for e in bridge.events.entries()]
    assert any("did not restart" in m for m in messages)


# --- the engine reaches the display ----------------------------------------

def test_a_fill_reaches_the_cluster(terminal):
    """The observer is the only path from a trade to the neurons."""
    _, bridge = terminal
    bridge.session._on_fill("BTC/USDT", 0.5, 64000.0, 1.2)

    assert len(bridge.tracker.open_positions) == 1
    assert any("bought" in e["message"] for e in bridge.events.entries())


def test_the_engine_calls_the_observer_on_a_fill():
    from godalgo.execution.engine import LiveEngine, LiveEngineConfig
    from godalgo.execution.types import (
        Order,
        OrderResult,
        OrderSide,
        OrderStatus,
        OrderType,
        TimeInForce,
    )
    from godalgo.strategies.mean_reversion import MeanReversionStrategy
    from godalgo.strategies.momentum import MomentumStrategy

    engine = LiveEngine(
        SimpleNamespace(), MomentumStrategy(), MeanReversionStrategy(),
        LiveEngineConfig(symbol="BTC/USDT"),
    )
    seen: list[tuple] = []
    engine.on_fill = lambda *args: seen.append(args)

    engine._notify_fill(OrderResult(
        order=Order(symbol="BTC/USDT", side=OrderSide.BUY,
                    order_type=OrderType.MARKET, amount=2.0,
                    time_in_force=TimeInForce.IOC),
        status=OrderStatus.FILLED, filled_amount=2.0,
        average_price=100.0, fee=0.2,
    ))
    assert seen == [("BTC/USDT", 2.0, 100.0, 0.2)]


def test_an_observer_that_raises_cannot_break_the_engine():
    """A display failure must never reach the trading loop: by the time this
    runs the order has already settled at the venue."""
    from godalgo.execution.engine import LiveEngine, LiveEngineConfig
    from godalgo.execution.types import (
        Order,
        OrderResult,
        OrderSide,
        OrderStatus,
        OrderType,
        TimeInForce,
    )
    from godalgo.strategies.mean_reversion import MeanReversionStrategy
    from godalgo.strategies.momentum import MomentumStrategy

    engine = LiveEngine(
        SimpleNamespace(), MomentumStrategy(), MeanReversionStrategy(),
        LiveEngineConfig(symbol="BTC/USDT"),
    )

    def explode(*_):
        raise RuntimeError("the UI is on fire")

    engine.on_fill = explode
    engine._notify_fill(OrderResult(
        order=Order(symbol="BTC/USDT", side=OrderSide.BUY,
                    order_type=OrderType.MARKET, amount=1.0,
                    time_in_force=TimeInForce.IOC),
        status=OrderStatus.FILLED, filled_amount=1.0, average_price=1.0,
    ))  # must not raise


def test_an_unfilled_order_notifies_nothing():
    from godalgo.execution.engine import LiveEngine, LiveEngineConfig
    from godalgo.execution.types import (
        Order,
        OrderResult,
        OrderSide,
        OrderStatus,
        OrderType,
        TimeInForce,
    )
    from godalgo.strategies.mean_reversion import MeanReversionStrategy
    from godalgo.strategies.momentum import MomentumStrategy

    engine = LiveEngine(
        SimpleNamespace(), MomentumStrategy(), MeanReversionStrategy(),
        LiveEngineConfig(symbol="BTC/USDT"),
    )
    seen = []
    engine.on_fill = lambda *a: seen.append(a)
    engine._notify_fill(OrderResult(
        order=Order(symbol="BTC/USDT", side=OrderSide.BUY,
                    order_type=OrderType.MARKET, amount=1.0,
                    time_in_force=TimeInForce.IOC),
        status=OrderStatus.REJECTED, filled_amount=0.0,
    ))
    assert seen == []


# --- the session reports what it is doing ----------------------------------

def test_warming_up_is_distinguished_from_running(terminal):
    """They look identical from outside and mean completely different things:
    one resolves itself, the other is the bot deciding not to trade."""
    _, bridge = terminal
    assert bridge.session.warmed_up is False

    bridge.session._engine = SimpleNamespace(
        bars=SimpleNamespace(n_complete=5000), _warmup=250,
        state=SimpleNamespace(snapshot=dict),
    )
    assert bridge.session.warmed_up is True


def test_the_mode_is_read_from_the_broker_in_hand():
    """Not passed alongside it, so the label cannot disagree with where the
    orders are actually going."""
    from godalgo.execution.broker import DryRunBroker, PaperBroker
    from godalgo.ui.session import _mode_of

    assert _mode_of(PaperBroker()) is TradingMode.PAPER
    assert _mode_of(DryRunBroker()) is TradingMode.DRY_RUN


def test_seed_timeframe_never_exceeds_the_bar_interval():
    """Seeding with coarser bars than the engine trades hands it history that
    is not the data it is about to see."""
    from godalgo.ui.session import _timeframe

    assert _timeframe(60) == "1m"
    assert _timeframe(300) == "5m"
    assert _timeframe(3600) == "1h"
    # Anything between named steps rounds up to the next, never down.
    assert _timeframe(120) == "3m"


def test_stopping_a_session_that_never_started_is_harmless(terminal):
    _, bridge = terminal
    asyncio.run(bridge.session.stop())
    assert bridge.session.running is False


# --- the LIVE button must be reachable, not merely enabled -----------------

def test_header_controls_cannot_shrink_below_their_content():
    """A shipped bug, and a subtle one.

    Adding a pill to the header pushed the nowrap flex row past its width.
    ``.mode-switch`` had no ``flex-shrink: 0``, so it shrank to 2px while its
    buttons kept rendering at full size *outside* their own box -- landing
    underneath the pill that followed. The LIVE button was enabled, visible,
    and could not be clicked. Verified in Chromium at 1024-1920px; this
    asserts the rule that makes it hold.
    """
    from pathlib import Path

    css = Path("src/godalgo/ui/static/style.css").read_text()

    def block(selector: str) -> str:
        """The base rule, not an override nested in a media query.

        Media-query rules are indented, so anchoring to the start of a line
        picks the top-level declaration -- which is the one that has to carry
        the constraint.
        """
        start = css.index("\n" + selector) + 1
        return css[start : css.index("}", start)]

    for selector in (".mode-switch {", ".mode-pill {", ".lamps {"):
        assert "flex: 0 0 auto" in block(selector), (
            f"{selector} may shrink; its content would then render outside its "
            f"own box and cover whatever follows it in the header"
        )

    # The row itself must clip rather than overflow, so a control can never be
    # pushed out of the viewport by a long readout.
    assert "overflow: hidden" in block("#header {")
