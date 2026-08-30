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
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from godalgo.ui.credentials import CredentialStore, ExchangeCredential
from godalgo.ui.journal import TradingJournal
from godalgo.ui.state import PositionTracker, TerminalHealth, UISnapshot
from godalgo.ui.telegram import TelegramNotifier

__all__ = ["UIBridge", "create_app", "run_server"]

logger = logging.getLogger(__name__)

_STATIC = Path(__file__).parent / "static"
_LOOPBACK_ONLY = "the UI holds exchange credentials and has no authentication"


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
    telegram: TelegramNotifier = field(default_factory=TelegramNotifier)

    starting_equity: float = 10_000.0
    equity: float = 10_000.0
    peak_equity: float = 10_000.0
    mode: str = "dry_run"
    symbol: str = "BTC/USDT"
    regime: str = "indeterminate"
    conviction: float = 0.0
    target_weight: float = 0.0
    current_weight: float = 0.0

    connected: bool = False
    last_data_at: datetime | None = None
    halted: bool = False
    halt_reason: str | None = None
    reconnects: int = 0
    errors: int = 0
    decisions_run: int = 0

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
            neurons=self.tracker.neurons(),
            health=self.health(),
            equity=self.equity,
            starting_equity=self.starting_equity,
            realised_pnl=self.tracker.realised_total,
            unrealised_pnl=self.tracker.unrealised_total,
            win_rate=self.tracker.win_rate,
            profit_factor=self.tracker.profit_factor,
            open_count=len(self.tracker.open_positions),
            closed_count=len(self.tracker.closed_positions),
            mode=self.mode,
            symbol=self.symbol,
            regime=self.regime,
            conviction=self.conviction,
            target_weight=self.target_weight,
            current_weight=self.current_weight,
        )

    async def check_daily_rollover(self) -> None:
        """Roll the journal at UTC midnight and push the summary to Telegram."""
        summary = self.journal.check_rollover(self.equity)
        if summary is None:
            return
        logger.info("daily rollover: %s", summary.day)
        await self.telegram.send(summary.as_telegram())


def create_app(bridge: UIBridge) -> FastAPI:
    """Build the FastAPI application."""
    app = FastAPI(title="GODALGO", docs_url=None, redoc_url=None)

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
        })

    @app.post("/api/connections")
    async def add_connection(payload: dict[str, Any]) -> JSONResponse:
        required = ("exchange_id", "api_key", "api_secret")
        missing = [k for k in required if not payload.get(k)]
        if missing:
            raise HTTPException(status_code=400, detail=f"missing: {', '.join(missing)}")

        bridge.credentials.add(
            ExchangeCredential(
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
        )
        return JSONResponse({"ok": True, "exchanges": bridge.credentials.listing()})

    @app.delete("/api/connections/{key}")
    async def remove_connection(key: str) -> JSONResponse:
        if not bridge.credentials.remove(key):
            raise HTTPException(status_code=404, detail="no such connection")
        return JSONResponse({"ok": True, "exchanges": bridge.credentials.listing()})

    @app.post("/api/telegram/test")
    async def telegram_test() -> JSONResponse:
        ok, message = await bridge.telegram.verify()
        if ok:
            await bridge.telegram.send("GODALGO terminal connected.")
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


def run_server(
    bridge: UIBridge,
    host: str = "127.0.0.1",
    port: int = 8787,
    *,
    open_browser: bool = True,
) -> None:
    """Serve the UI. Blocks until interrupted."""
    import uvicorn

    _assert_loopback(host)
    app = create_app(bridge)
    url = f"http://{host}:{port}"

    if open_browser:
        import threading
        import webbrowser

        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    logger.info("GODALGO terminal on %s (loopback only)", url)
    uvicorn.run(app, host=host, port=port, log_level="warning")


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
