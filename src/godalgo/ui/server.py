"""Local UI server.

Binds to **127.0.0.1 only**, and refuses any other host. This is not a
configuration preference: the process holds exchange API keys and has no
authentication, so exposing it on a LAN or public interface would hand order
authority to anyone who can reach the port. The bind address is validated rather
than documented, because a documented rule is one nobody reads.

The server is a *view*. It never places orders and never mutates engine state;
it reads a snapshot and streams it. A UI that can trade is a second, untested
path to the exchange, and the whole point of the execution layer is that there
is exactly one.
"""

from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import json
import logging
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from godalgo.build_info import describe
from godalgo.execution.mode import ModeController, ModeSwitchError
from godalgo.execution.types import TradingMode
from godalgo.ui.credentials import CredentialStore, ExchangeCredential
from godalgo.ui.events import EventLog
from godalgo.ui.journal import TradingJournal
from godalgo.ui.state import (
    PositionTracker,
    TerminalHealth,
    UISnapshot,
    WatchedSymbol,
)
from godalgo.ui.telegram import TelegramNotifier
from godalgo.ui.venue import ProbeResult, VenueProbe

__all__ = ["UIBridge", "build_terminal", "create_app", "run_server"]

logger = logging.getLogger(__name__)

def _static_dir() -> Path:
    """Locate the UI's static files, frozen or not.

    PyInstaller unpacks bundled data to a temporary directory and points
    ``sys._MEIPASS`` at it; the path relative to this module does not exist in
    a onefile build. Getting this wrong produces a binary that serves the API
    and 404s its own page, which reads as a server fault rather than a
    packaging one.
    """
    bundled = getattr(sys, "_MEIPASS", None)
    if bundled:
        candidate = Path(bundled) / "godalgo" / "ui" / "static"
        if candidate.exists():
            return candidate
    return Path(__file__).parent / "static"


_STATIC = _static_dir()
_LOOPBACK_ONLY = "the UI holds exchange credentials and has no authentication"

DEFAULT_UNIVERSE: tuple[str, ...] = (
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
    "ADA/USDT", "AVAX/USDT", "LINK/USDT", "DOT/USDT", "DOGE/USDT",
    "LTC/USDT", "ATOM/USDT",
)
"""What the terminal watches out of the box.

Twelve liquid majors: enough that the panel shows the bot surveying a
market rather than staring at one symbol, and few enough that the whole
set arrives in a single batched request.
"""


@dataclass
class UIBridge:
    """Shared state between the trading engine and the UI.

    Deliberately decoupled from ``LiveEngine``: the UI must keep rendering when
    the engine halts, and the engine must never block on a websocket. The bridge
    is fed by whoever is driving -- a live engine, a paper run, or the built-in
    simulator -- and the front end cannot tell the difference.
    """

    tracker: PositionTracker = field(default_factory=PositionTracker)
    journal: TradingJournal = field(default_factory=TradingJournal)
    credentials: CredentialStore = field(default_factory=CredentialStore)
    telegram: TelegramNotifier = field(init=False)
    events: EventLog = field(default_factory=EventLog)
    """Everything the terminal is doing, as a readable stream.

    The reason this exists: a bot that is working and a bot that is silently
    doing nothing look identical from the outside, and the second is the
    expensive case.
    """

    venue_status: dict[str, Any] = field(default_factory=dict)
    """Last probe result per check name. Rendered as the connection lamps."""

    last_price: float = 0.0
    last_price_at: datetime | None = None
    prices: dict[str, float] = field(default_factory=dict)
    watchlist: dict[str, WatchedSymbol] = field(default_factory=dict)
    """What the bot is currently tracking, keyed by symbol.

    A dict rather than a list so a tick updates in place: rebuilding the
    collection each poll would churn objects the snapshot serialises once a
    second, for a set of symbols that changes almost never.
    """

    exchange_id: str = "binance"
    """Default venue for market data, when no credential names one."""

    universe: list[str] = field(default_factory=lambda: list(DEFAULT_UNIVERSE))
    """Symbols the feed polls. The scanner narrows this to what it will trade;
    the watchlist shows the whole set so the operator can see what was
    considered, not only what was chosen."""

    market_feed_enabled: bool = True
    """Whether to poll public market data on startup.

    Off in tests and in demo mode, where a real network call would make the
    suite depend on an exchange being up.
    """

    _feed_task: Any = field(default=None, init=False, repr=False)

    session: Any = None
    """The running trading loop, if the terminal owns one.

    Optional so the UI still runs as a pure viewer in demo mode. When present,
    a mode switch starts, stops or re-brokers it -- which is what makes the
    mode switch mean anything at all.
    """

    mode_controller: ModeController | None = None
    """Owns the broker and performs guarded mode switches.

    Optional so the UI runs standalone in demo mode, where there is nothing to
    switch. When absent the mode endpoints report the control as unavailable
    rather than pretending to work.
    """

    starting_equity: float = 10_000.0
    equity: float = 10_000.0
    peak_equity: float = 10_000.0
    mode: str = "dry_run"
    symbol: str = "BTC/USDT"
    regime: str = "indeterminate"
    conviction: float = 0.0
    target_weight: float = 0.0
    current_weight: float = 0.0

    max_neurons: int = 140
    """Cap on rendered positions.

    The cluster runs physics and a render pass per position every frame; an
    unbounded session would eventually make the panel unusable. Open positions
    are never trimmed, and the true total is reported so the cap is visible.
    The journal keeps the complete record regardless.
    """

    connected: bool = False
    last_data_at: datetime | None = None
    halted: bool = False
    halt_reason: str | None = None
    reconnects: int = 0
    errors: int = 0
    decisions_run: int = 0

    def __post_init__(self) -> None:
        # The notifier persists its token in the same owner-only file as the
        # exchange keys, so it needs the store handed to it at construction.
        self.telegram = TelegramNotifier(store=self.credentials)

    @property
    def probe(self) -> VenueProbe:
        return VenueProbe(self.events)

    def update_watchlist(self, rows: dict[str, dict[str, float]]) -> None:
        """Fold a poll into the watchlist.

        Held and active are recomputed every tick rather than tracked, because
        a position can open or close between polls and a stale marker on a
        watchlist is worse than no marker: it says the bot is in something it
        is not.
        """
        held = {n.symbol for n in self.tracker.open_positions.values()}
        for symbol, row in rows.items():
            self.watchlist[symbol] = WatchedSymbol(
                symbol=symbol,
                price=row.get("price", 0.0),
                change_pct=row.get("change_pct", 0.0),
                quote_volume=row.get("quote_volume", 0.0),
                spread_bps=row.get("spread_bps", 0.0),
                held=symbol in held,
                active=symbol == self.symbol,
                stale=False,
            )

        # Anything the venue did not return this tick is marked stale rather
        # than dropped. A row that vanishes and reappears makes the panel
        # flicker and loses the operator's place; a greyed row says plainly
        # that this one number is old.
        for symbol, existing in self.watchlist.items():
            if symbol not in rows and not existing.stale:
                self.watchlist[symbol] = replace(existing, stale=True)

    def watchlist_rows(self) -> list[WatchedSymbol]:
        """Ordered for reading: what the bot is in, then what it is on, then
        by turnover. Sorting by price or by symbol would bury the row that
        matters under whatever happens to be alphabetically first."""
        return sorted(
            self.watchlist.values(),
            key=lambda w: (not w.held, not w.active, -w.quote_volume, w.symbol),
        )

    def record_probe(self, results: list[ProbeResult]) -> None:
        """Fold probe results into the status lamps."""
        for result in results:
            self.venue_status[result.name] = result.to_dict()
            if result.name == "market_data" and result.ok:
                self.last_price = float(result.data.get("price") or 0.0)
                self.last_price_at = datetime.now(UTC)

    def record_fill(
        self, symbol: str, signed_quantity: float, price: float, fee: float = 0.0, **kwargs: Any
    ) -> None:
        """Route a fill into the tracker and journal a completed round trip."""
        closed = self.tracker.on_fill(symbol, signed_quantity, price, fee, **kwargs)
        if closed is not None:
            self.journal.record(closed, equity=self.equity)

    def update_equity(self, equity: float) -> None:
        self.equity = equity
        self.peak_equity = max(self.peak_equity, equity)

    @property
    def drawdown(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return max(0.0, 1.0 - self.equity / self.peak_equity)

    def health(self) -> TerminalHealth:
        age = (
            (datetime.now(UTC) - self.last_data_at).total_seconds()
            if self.last_data_at else 1e9
        )
        return TerminalHealth(
            connected=self.connected,
            data_age_seconds=age,
            halted=self.halted,
            halt_reason=self.halt_reason,
            reconnects=self.reconnects,
            errors=self.errors,
            decisions_run=self.decisions_run,
            drawdown=self.drawdown,
        )

    def snapshot(self) -> UISnapshot:
        return UISnapshot(
            timestamp=datetime.now(UTC),
            neurons=self.tracker.neurons(limit=self.max_neurons),
            health=self.health(),
            equity=self.equity,
            starting_equity=self.starting_equity,
            realised_pnl=self.tracker.realised_total,
            unrealised_pnl=self.tracker.unrealised_total,
            win_rate=self.tracker.win_rate,
            profit_factor=self.tracker.profit_factor,
            open_count=len(self.tracker.open_positions),
            closed_count=len(self.tracker.closed_positions),
            rendered_count=len(self.tracker.neurons(limit=self.max_neurons)),
            total_count=self.tracker.total_tracked,
            mode=self.mode,
            symbol=self.symbol,
            regime=self.regime,
            conviction=self.conviction,
            target_weight=self.target_weight,
            current_weight=self.current_weight,
            last_price=self.last_price,
            watchlist=self.watchlist_rows(),
            has_keys=len(self.credentials) > 0,
            build=describe(),
            venue=dict(self.venue_status),
            events=self.events.entries(limit=40),
        )

    async def check_daily_rollover(self) -> None:
        """Roll the journal at UTC midnight and push the summary to Telegram."""
        summary = self.journal.check_rollover(self.equity)
        if summary is None:
            return
        logger.info("daily rollover: %s", summary.day)
        await self.telegram.send(summary.as_telegram())


def build_terminal(
    *,
    symbol: str = "BTC/USDT",
    equity: float = 10_000.0,
    bar_seconds: int = 60,
    exchange_id: str = "binance",
) -> UIBridge:
    """Assemble a terminal that can actually trade.

    One constructor, called by every entry point, because the previous
    arrangement had each of them wiring this by hand and they drifted: the
    packaged executable built a bridge with no controller at all, so every
    mode button was disabled, and the CLI built a controller that never
    consulted the credential store, so live was never available even with a
    key marked as permitted. Both were invisible until someone pressed the
    button.

    Returns a bridge with:

    * a ``ModeController`` that reads the credential store, so a key ticked
      in the interface is the key live uses;
    * a ``TradingSession`` the mode switch starts, stops and re-brokers;
    * a flatten hook, so no switch can strand an open position.
    """
    from godalgo.ui.session import TradingSession

    bridge = UIBridge(
        starting_equity=equity, equity=equity, exchange_id=exchange_id,
    )
    bridge.symbol = symbol

    session = TradingSession(
        bridge, symbol=symbol, bar_seconds=bar_seconds, exchange_id=exchange_id,
    )
    bridge.session = session

    async def flatten() -> None:
        """Close everything before a mode switch.

        Stopping the session routes through the engine's halt, which cancels
        resting orders and flattens. A switch that skipped this would leave
        the old broker holding a position nothing is managing.
        """
        await session.stop()

    controller = ModeController(
        mode=TradingMode.DRY_RUN,
        equity=equity,
        flatten=flatten,
        # The whole point: live reads the key the operator ticked, rather than
        # an environment variable a double-clicked executable cannot set.
        tradeable_credential=bridge.credentials.tradeable,
    )
    bridge.mode_controller = controller
    bridge.mode = controller.mode.value
    return bridge


async def _restart_feed(bridge: UIBridge) -> None:
    """Rebind the background poller to the current venue."""
    from godalgo.ui.feed import MarketFeed

    task = getattr(bridge, "_feed_task", None)
    if task is not None:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task
    if not bridge.market_feed_enabled:
        return
    bridge._feed_task = asyncio.create_task(
        MarketFeed(
            bridge,
            exchange_id=_feed_exchange(bridge),
            symbols=_watch_symbols(bridge),
        ).run()
    )


def _feed_exchange(bridge: UIBridge) -> str:
    """Which venue the watchlist follows.

    A stored credential wins over the default: someone who added binance.us
    keys is telling us where their account lives, and polling binance.com for
    prices they cannot trade against would be showing them the wrong market.
    """
    from godalgo.ui.venue import known_exchange, normalise_exchange_id

    stored = bridge.credentials.tradeable()
    if stored is None:
        entries = bridge.credentials.items()
        stored = entries[0][1] if entries else None

    candidate = normalise_exchange_id(
        getattr(stored, "exchange_id", None) or bridge.exchange_id
    )
    # Never hand an unusable id to the feed. A stored typo would otherwise
    # take market data down entirely rather than degrading to the default.
    return candidate if known_exchange(candidate) else "binance"


def _watch_symbols(bridge: UIBridge) -> list[str]:
    """The universe, with the traded symbol guaranteed present.

    A bot deciding on a symbol the watchlist does not carry would show its
    headline price as blank while claiming to be trading it.
    """
    symbols = list(bridge.universe)
    if bridge.symbol and bridge.symbol not in symbols:
        symbols.insert(0, bridge.symbol)
    return symbols


def create_app(bridge: UIBridge) -> FastAPI:
    """Build the FastAPI application."""

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        """Run the public market feed for the life of the server.

        Started here rather than lazily on first page load, so the terminal is
        already showing live prices by the time anyone opens it, and so a
        headless run still logs whether the venue is reachable.
        """
        if bridge.session is not None and bridge.mode_controller is not None:
            # Dry run is a real session: the full pipeline runs and the broker
            # discards the orders. Starting it here means the operator sees
            # what the bot would do against live prices without choosing
            # anything first.
            with contextlib.suppress(Exception):
                await bridge.session.start(bridge.mode_controller.broker)

        task: asyncio.Task | None = None
        if bridge.market_feed_enabled:
            from godalgo.ui.feed import MarketFeed

            task = asyncio.create_task(
                MarketFeed(
                    bridge,
                    exchange_id=_feed_exchange(bridge),
                    symbols=_watch_symbols(bridge),
                ).run()
            )
            bridge._feed_task = task
        try:
            yield
        finally:
            if bridge.session is not None:
                with contextlib.suppress(Exception):
                    await bridge.session.stop()
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    app = FastAPI(
        title="GODALGO", docs_url=None, redoc_url=None, lifespan=lifespan,
    )

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(_STATIC / "index.html")

    @app.get("/api/state")
    async def state() -> JSONResponse:
        return JSONResponse(bridge.snapshot().to_dict())

    @app.get("/api/positions/{position_id}")
    async def position(position_id: str) -> JSONResponse:
        for neuron in bridge.tracker.neurons():
            if neuron.id == position_id:
                return JSONResponse(neuron.to_dict())
        raise HTTPException(status_code=404, detail="position not found")

    @app.get("/api/journal")
    async def journal(limit: int = 50) -> JSONResponse:
        entries = bridge.journal.entries()[-limit:]
        return JSONResponse({
            "entries": list(reversed(entries)),
            "summaries": bridge.journal.summaries(limit=14),
        })

    @app.get("/api/connections")
    async def connections() -> JSONResponse:
        """Masked credential listing. Never returns key material."""
        return JSONResponse({
            "exchanges": bridge.credentials.listing(),
            "telegram": bridge.telegram.status,
            "store_path": str(bridge.credentials.path),
            # protection() shells out to icacls on Windows, so it is polled
            # here rather than included in the once-a-second snapshot.
            "protection": await asyncio.to_thread(bridge.credentials.protection),
        })

    @app.post("/api/connections")
    async def add_connection(payload: dict[str, Any]) -> JSONResponse:
        required = ("exchange_id", "api_key", "api_secret")
        missing = [k for k in required if not payload.get(k)]
        if missing:
            raise HTTPException(status_code=400, detail=f"missing: {', '.join(missing)}")

        credential = ExchangeCredential(
            exchange_id=str(payload["exchange_id"]).strip(),
            api_key=str(payload["api_key"]).strip(),
            api_secret=str(payload["api_secret"]).strip(),
            label=str(payload.get("label") or "").strip(),
            passphrase=str(payload.get("passphrase") or "").strip(),
            testnet=bool(payload.get("testnet", False)),
            # Trading stays off unless explicitly requested, so pasting a
            # key into a form cannot by itself authorise orders.
            trade_enabled=bool(payload.get("trade_enabled", False)),
        )
        # Off the loop: on Windows the store's ACL restriction spawns icacls,
        # and a subprocess launch there costs tens of milliseconds during which
        # the websocket, the market feed and the trading loop all stall.
        await asyncio.to_thread(bridge.credentials.add, credential)
        bridge.events.info(
            "credentials",
            f"added {credential.exchange_id} key"
            + (" (testnet)" if credential.testnet else ""),
            "may place orders" if credential.trade_enabled else "read-only",
        )

        # Test it immediately. Saving a key and seeing nothing happen is the
        # single most confusing state this interface can be in -- it is
        # indistinguishable from the key being ignored, which it used to be.
        results = await bridge.probe.check_credentials(credential, bridge.symbol)
        bridge.record_probe(results)

        # Follow the venue the key names. Someone adding binance.us keys is
        # saying where their account lives; polling binance.com for prices
        # they cannot trade against would be showing the wrong market.
        if credential.exchange_id != bridge.exchange_id:
            bridge.exchange_id = credential.exchange_id
            bridge.watchlist.clear()
            await _restart_feed(bridge)

        return JSONResponse({
            "ok": True,
            "exchanges": bridge.credentials.listing(),
            "checks": [r.to_dict() for r in results],
        })

    @app.post("/api/connections/{key}/test")
    async def test_connection(key: str) -> JSONResponse:
        """Re-run the read-only checks against a stored key.

        Reads only: reachability, a ticker, and the account balance. Nothing
        here can place, amend or cancel an order.
        """
        credential = bridge.credentials.get(key)
        if credential is None:
            raise HTTPException(status_code=404, detail="no such connection")
        results = await bridge.probe.check_credentials(credential, bridge.symbol)
        bridge.record_probe(results)
        return JSONResponse({
            "ok": all(r.ok for r in results),
            "checks": [r.to_dict() for r in results],
        })

    @app.post("/api/connections/{key}/trade-enabled")
    async def set_trade_enabled(key: str, payload: dict[str, Any]) -> JSONResponse:
        """Grant or revoke this key's permission to place orders.

        This flag is the consent record the live switch checks, so it is a
        deliberate action of its own rather than something bundled into
        saving a key.
        """
        enabled = bool(payload.get("enabled", False))
        if not await asyncio.to_thread(
            bridge.credentials.set_trade_enabled, key, enabled
        ):
            raise HTTPException(status_code=404, detail="no such connection")
        bridge.events.record(
            "warn" if enabled else "info", "credentials",
            f"{key} {'may now place orders' if enabled else 'is now read-only'}",
        )
        return JSONResponse({"ok": True, "exchanges": bridge.credentials.listing()})

    @app.post("/api/venue/check")
    async def venue_check(payload: dict[str, Any] | None = None) -> JSONResponse:
        """Prove the terminal can reach the venue and read prices.

        Needs no key and no funds, which is the point: it settles "is this
        actually talking to the exchange" before any decision to deposit.
        """
        from godalgo.ui.venue import normalise_exchange_id

        body = payload or {}
        exchange_id = normalise_exchange_id(
            str(body.get("exchange_id") or "binance")
        )
        symbol = str(body.get("symbol") or bridge.symbol).strip()

        results = await bridge.probe.check_public(exchange_id, symbol)
        stored = bridge.credentials.first_for(exchange_id)
        if stored is not None and results[0].ok:
            results = await bridge.probe.check_credentials(stored, symbol)
        bridge.record_probe(results)
        return JSONResponse({
            "ok": all(r.ok for r in results),
            "checks": [r.to_dict() for r in results],
            "credential_tested": stored is not None,
        })

    @app.post("/api/watchlist/refresh")
    async def watchlist_refresh(payload: dict[str, Any] | None = None) -> JSONResponse:
        """Poll the venue once, now, and report exactly what came back.

        The poller runs on its own cadence, so without this the only way to
        find out why the list is empty is to wait and guess. Switching venue
        here also rebinds the background poller, which matters when the
        default venue is blocked in your region and another is not.
        """
        from godalgo.ui.feed import MarketFeed
        from godalgo.ui.venue import known_exchange, normalise_exchange_id

        body = payload or {}
        requested = normalise_exchange_id(str(body.get("exchange_id") or ""))
        if requested and not known_exchange(requested):
            from godalgo.ui.venue import suggest_exchange_ids

            hint = suggest_exchange_ids(str(body.get("exchange_id") or ""))
            raise HTTPException(
                status_code=400,
                detail=f"{body.get('exchange_id')!r} is not an exchange ccxt knows"
                + (f" — did you mean {' or '.join(hint)}?" if hint else ""),
            )
        if requested and requested != bridge.exchange_id:
            bridge.exchange_id = requested
            bridge.watchlist.clear()
            bridge.events.info("data", f"market data venue set to {requested}")
            await _restart_feed(bridge)

        feed = MarketFeed(
            bridge,
            exchange_id=_feed_exchange(bridge),
            symbols=_watch_symbols(bridge),
        )
        try:
            await feed._tick()
        finally:
            await feed._close()

        status = bridge.venue_status.get("market_data") or {}
        return JSONResponse({
            "ok": bool(status.get("ok")),
            "exchange_id": _feed_exchange(bridge),
            "detail": status.get("detail", ""),
            "kind": status.get("kind", ""),
            "rows": len(bridge.watchlist),
        })

    @app.post("/api/diagnostics")
    async def diagnostics(payload: dict[str, Any] | None = None) -> JSONResponse:
        """Find out which layer is failing, and say so in plain terms.

        "Could not reach the venue" covers DNS failure, a proxy, TLS
        interception, a geo-block and a firewall. This tests each layer
        separately and reports the raw error from every one, so the answer is
        a specific remedy rather than a guess.
        """
        from godalgo.ui.diagnostics import run_diagnostics
        from godalgo.ui.venue import normalise_exchange_id

        body = payload or {}
        exchange_id = normalise_exchange_id(
            str(body.get("exchange_id") or "") or _feed_exchange(bridge)
        )
        report = await run_diagnostics(exchange_id)

        bridge.events.record(
            "good" if report["ok"] else "error", "venue",
            f"diagnostics on {exchange_id}: "
            + ("all layers reachable" if report["ok"] else "a layer failed"),
            report["verdict"][:300],
        )
        return JSONResponse(report)

    @app.get("/api/session")
    async def session_status() -> JSONResponse:
        """What the trading loop is doing.

        Separate from /api/mode because "which broker" and "is the loop
        actually running" are different questions, and the second is the one
        that goes unanswered when a bot appears to do nothing.
        """
        if bridge.session is None:
            return JSONResponse({
                "attached": False,
                "reason": "this terminal is a viewer; no trading loop is attached",
            })
        return JSONResponse({"attached": True, **bridge.session.status()})

    @app.get("/api/events")
    async def events(limit: int = 80, since: int = 0) -> JSONResponse:
        return JSONResponse({
            "events": bridge.events.entries(limit=limit, since=since),
            "latest": bridge.events.latest_sequence,
            "counts": bridge.events.counts(),
        })

    @app.delete("/api/connections/{key}")
    async def remove_connection(key: str) -> JSONResponse:
        if not bridge.credentials.remove(key):
            raise HTTPException(status_code=404, detail="no such connection")
        return JSONResponse({"ok": True, "exchanges": bridge.credentials.listing()})

    @app.get("/api/mode")
    async def mode_status() -> JSONResponse:
        if bridge.mode_controller is None:
            return JSONResponse({
                "available": False,
                "mode": bridge.mode,
                "reason": "no trading session attached (demo view)",
            })
        return JSONResponse({"available": True, **bridge.mode_controller.status()})

    @app.post("/api/mode")
    async def set_mode(payload: dict[str, Any]) -> JSONResponse:
        """Switch trading mode.

        Every switch flattens first, and live additionally requires the arming
        environment variable, credentials, and a typed confirmation. The
        interface can request live; it cannot authorise it.
        """
        if bridge.mode_controller is None:
            raise HTTPException(status_code=409, detail="no trading session attached")

        requested = str(payload.get("mode") or "").strip().lower()
        try:
            target = TradingMode(requested)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"unknown mode {requested!r}; expected dry_run, paper or live",
            ) from None

        try:
            change = await bridge.mode_controller.switch(
                target, confirm=payload.get("confirm"), note="changed from the terminal",
            )
        except ModeSwitchError as exc:
            # 409 rather than 400: the request is well-formed, the system state
            # or environment does not permit it.
            bridge.events.error("mode", f"refused switch to {target.value}", str(exc))
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        bridge.mode = target.value
        bridge.events.record(
            "warn" if target is TradingMode.LIVE else "info", "mode",
            f"mode is now {target.value}",
            "positions were flattened first" if change.flattened else "",
        )

        # Re-broker the running loop, or the switch would change a label and
        # nothing else -- which is what it used to do.
        if bridge.session is not None:
            try:
                await bridge.session.start(bridge.mode_controller.broker)
            except Exception as exc:
                logger.exception("could not restart the trading session")
                bridge.events.error(
                    "engine", "mode changed but the session did not restart",
                    str(exc)[:200],
                )
        return JSONResponse({"ok": True, "change": change.to_dict(),
                             **bridge.mode_controller.status()})

    @app.post("/api/telegram")
    async def telegram_configure(payload: dict[str, Any]) -> JSONResponse:
        """Save the bot token and chat id, then prove they work.

        Verified immediately for the same reason exchange keys are: a settings
        form that accepts anything and reports nothing teaches you it worked
        when it did not.
        """
        token = str(payload.get("token") or "").strip()
        chat_id = str(payload.get("chat_id") or "").strip()
        if not token or not chat_id:
            raise HTTPException(status_code=400, detail="token and chat_id required")

        bridge.telegram.configure(token, chat_id)
        ok, message = await bridge.telegram.verify()
        if ok:
            await bridge.telegram.send(
                "GODALGO terminal connected. Daily summaries will arrive here."
            )
            bridge.events.good("telegram", "connected", message)
        else:
            # Never the token, and never the URL, which embeds it.
            bridge.events.error("telegram", "could not connect", message)
        return JSONResponse({
            "ok": ok, "message": message, "telegram": bridge.telegram.status,
        })

    @app.delete("/api/telegram")
    async def telegram_clear() -> JSONResponse:
        bridge.telegram.clear()
        bridge.events.info("telegram", "disconnected")
        return JSONResponse({"ok": True, "telegram": bridge.telegram.status})

    @app.post("/api/telegram/test")
    async def telegram_test() -> JSONResponse:
        ok, message = await bridge.telegram.verify()
        if ok:
            await bridge.telegram.send("GODALGO terminal connected.")
        bridge.events.record(
            "good" if ok else "error", "telegram",
            "test message sent" if ok else "test failed", message,
        )
        return JSONResponse({"ok": ok, "message": message})

    @app.post("/api/telegram/digest")
    async def telegram_digest() -> JSONResponse:
        """Send today's summary now, rather than waiting for the rollover."""
        today = datetime.now(UTC).date()
        summary = bridge.journal.summarise(today, bridge.starting_equity, bridge.equity)
        sent = await bridge.telegram.send(summary.as_telegram())
        return JSONResponse({"ok": sent, "summary": summary.to_dict()})

    @app.websocket("/ws")
    async def websocket(socket: WebSocket) -> None:
        """Stream snapshots at a fixed cadence.

        Push rather than poll, and a fixed interval rather than on-change: the
        cluster animates continuously, so a steady frame rate is what the client
        actually needs, and it bounds the work regardless of trade volume.
        """
        await socket.accept()
        try:
            while True:
                await socket.send_text(json.dumps(bridge.snapshot().to_dict()))
                await bridge.check_daily_rollover()
                await asyncio.sleep(1.0)
        except WebSocketDisconnect:
            pass
        except Exception:
            logger.exception("websocket stream failed")
            with contextlib.suppress(Exception):
                await socket.close()

    if _STATIC.exists():
        app.mount("/static", StaticFiles(directory=_STATIC), name="static")

    return app


def _assert_loopback(host: str) -> None:
    """Refuse any bind address that is not loopback.

    Enforced rather than documented. The process holds credentials that can move
    money and has no authentication; binding it to 0.0.0.0 would publish order
    authority to the network.
    """
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        if host.lower() in {"localhost", "localhost.localdomain"}:
            return
        raise ValueError(
            f"refusing to bind to {host!r}: {_LOOPBACK_ONLY}. Use 127.0.0.1."
        ) from None
    if not address.is_loopback:
        raise ValueError(
            f"refusing to bind to {host!r}: {_LOOPBACK_ONLY}. Use 127.0.0.1."
        )


def port_owner(host: str, port: int, timeout: float = 1.5) -> str:
    """Who holds this port: ``free``, ``godalgo``, or ``other``.

    Identified by asking, not by guessing. Anything can be listening on a
    loopback port; only our own terminal answers ``/api/state`` with a
    snapshot, so a probe distinguishes "the app is already running" from "some
    unrelated program has the port" -- and those need opposite responses.
    """
    import socket
    import urllib.error
    import urllib.request

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(timeout)
        if probe.connect_ex((host, port)) != 0:
            return "free"

    try:
        with urllib.request.urlopen(
            f"http://{host}:{port}/api/state", timeout=timeout
        ) as response:
            body = response.read(4096)
    except (urllib.error.URLError, OSError, ValueError):
        return "other"
    return "godalgo" if b'"health"' in body and b'"pnl"' in body else "other"


def find_free_port(host: str, start: int, attempts: int = 20) -> int | None:
    """The first free port at or after ``start``."""
    import socket

    for candidate in range(start, start + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind((host, candidate))
            except OSError:
                continue
            return candidate
    return None


def run_server(
    bridge: UIBridge,
    host: str = "127.0.0.1",
    port: int = 8787,
    *,
    open_browser: bool = True,
) -> None:
    """Serve the UI. Blocks until interrupted.

    A second launch does not crash. Previously it died with

        [Errno 10048] only one usage of each socket address ... is normally
        permitted

    which is the operating system's phrasing for "something already has this
    port" and says nothing about what to do. Worse, the failure is silent in
    practice: the window closes, the *older* copy is still serving, and the
    browser shows a terminal that looks fine while being a build behind. That
    is how a fixed bug appears not to be fixed.
    """
    import uvicorn

    _assert_loopback(host)

    owner = port_owner(host, port)
    if owner == "godalgo":
        # Already running. Show that one rather than failing; two copies on one
        # account would size against the same buying power anyway.
        url = f"http://{host}:{port}"
        print(f"GODALGO is already running at {url} — opening it.")
        print("This window can be closed. To run a second copy, pass --port.")
        print("If it was started at sign-in, stop it with:")
        print("  godalgo-terminal.exe service stop")
        if open_browser:
            import webbrowser

            webbrowser.open(url)
        return

    if owner == "other":
        alternative = find_free_port(host, port + 1)
        if alternative is None:
            print(
                f"Port {port} is in use by another program, and no free port "
                f"was found nearby. Close whatever is using it, or pass "
                f"--port with a free one.",
                file=sys.stderr,
            )
            return
        print(f"Port {port} is in use by another program; using {alternative}.")
        port = alternative

    app = create_app(bridge)
    url = f"http://{host}:{port}"

    if open_browser:
        import threading
        import webbrowser

        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    logger.info("GODALGO terminal on %s (loopback only)", url)
    try:
        uvicorn.run(app, host=host, port=port, log_level="warning")
    except OSError as exc:
        # The check above is a probe, so a race is possible: something can take
        # the port between asking and binding. Report it in words rather than
        # as a raw errno.
        print(
            f"Could not start on {url}: {exc}\n"
            f"Something took the port after it was checked. Try again, or pass "
            f"--port with a different one.",
            file=sys.stderr,
        )


def main_cli() -> int:
    """Console-script entry point for ``godalgo-terminal``."""
    import argparse

    parser = argparse.ArgumentParser(description="GODALGO terminal")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--equity", type=float, default=10_000.0)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    bridge = UIBridge(starting_equity=args.equity, equity=args.equity)
    if args.demo:
        import threading

        from godalgo.ui.simulator import Simulator

        simulator = Simulator(bridge)
        simulator.seed_history()
        threading.Thread(target=lambda: asyncio.run(simulator.run()), daemon=True).start()

    run_server(bridge, port=args.port, open_browser=not args.no_browser)
    return 0
