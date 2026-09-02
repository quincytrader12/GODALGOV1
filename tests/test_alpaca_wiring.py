"""The terminal points at Alpaca paper, from the entry points inward.

Testing the wiring rather than the components, because isolation is what let a
dead mode switch ship once already: every part was correct and none of them
were connected. A venue switch is exactly the kind of change where that
happens again -- the client works, the broker works, and the terminal still
opens on the old venue.
"""

from __future__ import annotations

import inspect

import pytest

from godalgo.execution.types import TradingMode
from godalgo.ui.credentials import CredentialStore, ExchangeCredential
from godalgo.ui.journal import TradingJournal
from godalgo.ui.server import (
    BINANCE_UNIVERSE,
    DEFAULT_SYMBOL,
    DEFAULT_UNIVERSE,
    DEFAULT_VENUE,
    UIBridge,
    build_terminal,
    universe_for,
)


def test_the_default_venue_is_alpaca():
    assert DEFAULT_VENUE == "alpaca"
    assert UIBridge().exchange_id == "alpaca"


def test_the_default_symbol_trades_around_the_clock():
    """An equity default shows a bot that looks broken for most of every week,
    and cannot trade at all on a small account once the day-trade budget is
    spent. Crypto has neither problem."""
    from godalgo.venues.alpaca import is_crypto

    assert is_crypto(DEFAULT_SYMBOL)


def test_the_default_universe_is_alpaca_symbols():
    """Symbols do not travel between venues. A watchlist of BTC/USDT against
    Alpaca is twelve rows that never get a price, which reads as a dead feed."""
    assert "BTC/USD" in DEFAULT_UNIVERSE
    assert not [s for s in DEFAULT_UNIVERSE if s.endswith("/USDT")]


def test_the_universe_spans_both_asset_classes():
    """The whole point of this venue: not crypto-only."""
    from godalgo.venues.alpaca import is_crypto

    assert any(is_crypto(s) for s in DEFAULT_UNIVERSE)
    assert any(not is_crypto(s) for s in DEFAULT_UNIVERSE)


def test_switching_venue_switches_the_symbols():
    assert universe_for("alpaca") == list(DEFAULT_UNIVERSE)
    assert universe_for("binance") == list(BINANCE_UNIVERSE)


def test_binance_still_works_it_is_just_not_the_default():
    """Removing it would be destructive for no gain; it is simply not first."""
    bridge = build_terminal(exchange_id="binance", symbol="BTC/USDT")
    assert bridge.exchange_id == "binance"


def test_the_packaged_entry_point_builds_an_alpaca_terminal(tmp_path):
    """run-terminal.py is the file the executable runs. It built a bridge with
    no controller once, so every mode button was disabled; assert the venue
    the same way."""
    bridge = build_terminal()
    assert bridge.exchange_id == "alpaca"
    assert bridge.symbol == DEFAULT_SYMBOL
    assert bridge.mode_controller is not None
    assert bridge.session is not None


def _alpaca_credential(**kw) -> ExchangeCredential:
    base = {
        "exchange_id": "alpaca", "label": "paper", "api_key": "PK123",
        "api_secret": "s3cret", "testnet": True, "trade_enabled": True,
    }
    return ExchangeCredential(**{**base, **kw})


def test_live_mode_builds_an_alpaca_broker_not_a_ccxt_one(tmp_path):
    """ccxt's Alpaca binding is crypto-only. Going down that path would work,
    and would silently cut the universe from thousands of instruments to a
    couple of dozen pairs."""
    from godalgo.execution.mode import ModeController
    from godalgo.venues.alpaca_broker import AlpacaBroker

    credential = _alpaca_credential()
    controller = ModeController(
        mode=TradingMode.DRY_RUN, equity=1000.0,
        tradeable_credential=lambda: credential,
    )
    broker = controller._build(TradingMode.LIVE)
    assert isinstance(broker, AlpacaBroker)


def test_a_paper_credential_points_the_broker_at_the_paper_endpoint(tmp_path):
    """The single most important assertion here: LIVE mode against a paper
    credential must not reach real money."""
    from godalgo.execution.mode import ModeController

    controller = ModeController(
        mode=TradingMode.DRY_RUN, equity=1000.0,
        tradeable_credential=_alpaca_credential,
    )
    broker = controller._build(TradingMode.LIVE)
    assert broker.client.config.paper is True
    assert "paper-api" in broker.client.config.trading_base


def test_a_live_credential_is_needed_to_reach_real_money():
    from godalgo.execution.mode import ModeController

    controller = ModeController(
        mode=TradingMode.DRY_RUN, equity=1000.0,
        tradeable_credential=lambda: _alpaca_credential(testnet=False),
    )
    broker = controller._build(TradingMode.LIVE)
    assert broker.client.config.paper is False


def test_a_key_without_permission_still_cannot_trade():
    """The consent record survives the venue change. Pasting a key into a form
    does not by itself authorise orders, on any venue."""
    from godalgo.execution.live import ArmingError
    from godalgo.execution.mode import ModeController

    controller = ModeController(
        mode=TradingMode.DRY_RUN, equity=1000.0,
        tradeable_credential=lambda: _alpaca_credential(trade_enabled=False),
    )
    with pytest.raises(ArmingError, match="not permitted"):
        controller._build(TradingMode.LIVE)


def test_the_probe_routes_alpaca_natively():
    """Not through ccxt, for the crypto-only reason."""
    from godalgo.ui import venue

    source = inspect.getsource(venue.VenueProbe.check_public)
    assert "alpaca_probes" in source
    source = inspect.getsource(venue.VenueProbe.check_credentials)
    assert "alpaca_probes" in source


def test_the_stored_credential_decides_the_feed_venue(tmp_path):
    from godalgo.ui.server import _feed_exchange

    store = CredentialStore(directory=tmp_path)
    store.add(_alpaca_credential())
    bridge = UIBridge(
        credentials=store,
        journal=TradingJournal(path=tmp_path / "j.jsonl",
                               summary_path=tmp_path / "s.jsonl"),
        market_feed_enabled=False,
    )
    assert _feed_exchange(bridge) == "alpaca"


def test_the_feed_without_a_key_says_what_to_do_rather_than_looking_dead(tmp_path):
    """Alpaca serves market data only to authenticated requests. An empty
    panel with no explanation is the failure mode this whole project has been
    fighting."""
    import asyncio

    from godalgo.ui.feed import MarketFeed

    bridge = UIBridge(
        credentials=CredentialStore(directory=tmp_path),
        journal=TradingJournal(path=tmp_path / "j.jsonl",
                               summary_path=tmp_path / "s.jsonl"),
        market_feed_enabled=False,
    )
    feed = MarketFeed(bridge, exchange_id="alpaca", symbols=["BTC/USD"])
    asyncio.run(feed._tick())

    status = bridge.venue_status["market_data"]
    assert status["ok"] is False
    assert status["kind"] == "needs_key"
    assert "Connections" in status["detail"]


def test_the_paper_endpoint_is_never_reached_by_accident():
    """One string separates the two. It is never the default."""
    from godalgo.venues.alpaca import AlpacaConfig

    assert AlpacaConfig().paper is True
