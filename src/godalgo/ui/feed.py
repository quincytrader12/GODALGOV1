"""Live public market data, running whatever the mode is.

The terminal used to show nothing until a trading session was attached, which
meant the first thing a new operator saw was a dead screen -- indistinguishable
from a broken build. This poller fixes that, and it does so using the part of
the exchange API that needs no credentials and costs nothing:

* it requires **no API key**, so it runs before anything is configured;
* it requires **no funds**, so it proves the data path before any decision to
  deposit;
* it places **no orders**, in dry run, paper and live alike.

That combination is the answer to a reasonable question -- *is this thing
actually talking to Binance?* -- which previously could only be settled by
funding an account and watching.

Failures are events, not exceptions. A venue outage must show up as a red lamp
and a log line, never as a crashed background task that leaves the screen
frozen on its last good value.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from godalgo.ui.server import UIBridge

__all__ = ["MarketFeed"]

logger = logging.getLogger(__name__)


def _unsupported_errors() -> tuple[type[BaseException], ...]:
    """Exceptions that mean 'this venue will not batch', and nothing else.

    Resolved once at import rather than inline, so the tuple is not rebuilt
    on every poll.
    """
    import ccxt

    return (NotImplementedError, ccxt.NotSupported, ccxt.BadRequest, TypeError)


_UNSUPPORTED = _unsupported_errors()


@dataclass
class MarketFeed:
    """Polls public tickers and pushes them into the bridge.

    Polling rather than a websocket, deliberately. This drives status lamps and
    a headline price, not trading decisions -- the execution layer has its own
    websocket driver for that. A second socket here would double the
    reconnection logic to maintain for a display that updates once a second.

    Args:
        bridge: Where prices and events land.
        exchange_id: Venue to poll.
        symbols: Instruments to track. The first is the headline.
        interval: Seconds between polls.
    """

    bridge: UIBridge
    exchange_id: str = "binance"
    symbols: list[str] = field(default_factory=lambda: ["BTC/USDT"])
    interval: float = 5.0
    _exchange: Any = field(default=None, init=False, repr=False)
    _consecutive_failures: int = field(default=0, init=False)
    _batched: bool | None = field(default=None, init=False)
    """Whether the venue supports one-call ticker fetches. Probed once."""

    async def run(self) -> None:
        """Poll until cancelled. Never raises."""
        self.bridge.events.info(
            "data", f"watching {len(self.symbols)} symbol(s) on {self.exchange_id}",
            "public market data — no API key used",
        )
        try:
            while True:
                await self._tick()
                await asyncio.sleep(self.interval)
        finally:
            # Runs on cancellation too, which is the case that matters: the
            # aiohttp session must be closed or shutdown warns and leaks it.
            await self._close()

    async def _tick(self) -> None:
        try:
            exchange = await self._client()
            tickers = await self._fetch(exchange)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - reported, never propagated
            self._on_failure(exc)
            return

        if not tickers:
            return

        self._on_success(tickers)

    async def _client(self) -> Any:
        if self._exchange is None:
            import ccxt.async_support as accxt

            # ccxt leaves aiohttp's trust_env off, so it ignores HTTPS_PROXY and the
            # system proxy entirely. On a machine behind a corporate proxy, a VPN
            # client, or antivirus that intercepts TLS, every request fails while the
            # browser beside it works -- which reads as "the exchange is down".
            self._exchange = getattr(accxt, self.exchange_id)({
                "enableRateLimit": True, "timeout": 15_000,
                "aiohttp_trust_env": True,
            })
        return self._exchange

    async def _fetch(self, exchange: Any) -> dict[str, dict[str, float]]:
        """Read every watched symbol, in one request where the venue allows it.

        ``fetch_tickers`` is a single call covering the whole list;
        ``fetch_ticker`` is one call per symbol. With a dozen symbols on a five
        second cadence that is the difference between 12 requests and 1, and
        the loop form is also sequential -- twelve round trips of latency
        stacked inside one tick, which is what makes a watchlist feel slow and
        what trips venue rate limits.
        """
        if self._batched is None:
            self._batched = bool(getattr(exchange, "has", {}).get("fetchTickers"))

        if self._batched:
            try:
                raw = await exchange.fetch_tickers(self.symbols)
                return {s: _row(t) for s, t in raw.items() if t}
            except _UNSUPPORTED:
                # Narrow on purpose. Some venues advertise fetchTickers and
                # still reject an explicit symbol list, which is worth falling
                # back for -- but catching every exception here would read a
                # network blip as "batching unsupported" and permanently
                # downgrade the feed to one request per symbol per tick, which
                # is the exact cost this method exists to avoid. A transport
                # failure belongs to the outage handler, so it propagates.
                self._batched = False
                logger.debug("fetch_tickers unsupported; falling back", exc_info=True)

        out: dict[str, dict[str, float]] = {}
        for symbol in self.symbols:
            out[symbol] = _row(await exchange.fetch_ticker(symbol))
        return out

    def _on_success(self, tickers: dict[str, dict[str, float]]) -> None:
        recovered = self._consecutive_failures > 0
        self._consecutive_failures = 0

        prices = {s: r["price"] for s, r in tickers.items() if r.get("price")}
        # The headline follows the engine's symbol rather than whatever the
        # venue returned first, which is dict order and effectively arbitrary.
        headline = prices.get(self.bridge.symbol) or next(iter(prices.values()), 0.0)
        self.bridge.last_price = headline
        self.bridge.last_price_at = datetime.now(UTC)
        self.bridge.prices.update(prices)
        self.bridge.update_watchlist(tickers)
        self.bridge.connected = True
        self.bridge.last_data_at = datetime.now(UTC)
        self.bridge.venue_status["market_data"] = {
            "name": "market_data", "ok": True,
            "detail": f"{len(prices)} symbol(s) live",
            "data": {"price": headline, "symbols": len(prices)},
        }
        # A successful ticker is proof of reachability too. Leaving the venue
        # lamp grey while prices stream would be its own kind of lie.
        self.bridge.venue_status["reachable"] = {
            "name": "reachable", "ok": True,
            "detail": f"{self.exchange_id} responding",
        }
        if recovered:
            self.bridge.events.good(
                "data", "market data recovered", f"{self.exchange_id} responding again"
            )

    def _on_failure(self, exc: BaseException) -> None:
        from godalgo.ui.venue import classify_error, raw_error

        self._consecutive_failures += 1
        self.bridge.connected = False
        kind, explanation = classify_error(exc)
        self.bridge.venue_status["market_data"] = {
            "name": "market_data", "ok": False, "detail": explanation,
            "kind": kind, "raw": raw_error(exc),
        }
        # Only a transport-level failure implicates the venue itself. A bad
        # symbol or a rate limit means we reached it perfectly well, and
        # reddening the venue lamp for those would misdirect the operator.
        if kind in ("unreachable", "timeout", "geo_blocked"):
            self.bridge.venue_status["reachable"] = {
                "name": "reachable", "ok": False,
                "detail": explanation, "kind": kind,
            }

        # Only the first failure and then every twelfth is logged. A venue that
        # is down for an hour would otherwise produce 720 identical events and
        # push everything else out of the operator's view.
        if self._consecutive_failures == 1 or self._consecutive_failures % 12 == 0:
            self.bridge.events.warn(
                "data",
                f"market data unavailable ({self._consecutive_failures}x)",
                f"{explanation} [{raw_error(exc)}]",
            )

        # Force a fresh client next tick: a poisoned aiohttp session survives
        # the outage and keeps failing after the venue recovers.
        exchange, self._exchange = self._exchange, None
        if exchange is not None:
            with contextlib.suppress(Exception):
                asyncio.get_running_loop().create_task(exchange.close())

    async def _close(self) -> None:
        if self._exchange is not None:
            with contextlib.suppress(Exception):
                await self._exchange.close()
            self._exchange = None


def _row(ticker: dict[str, Any]) -> dict[str, float]:
    """Reduce a ccxt ticker to the handful of numbers the panel renders.

    A ccxt ticker carries a nested ``info`` blob of raw venue JSON. Passing
    that through would put kilobytes per symbol into a frame sent once a
    second, for fields nothing reads.
    """
    price = ticker.get("last") or ticker.get("close") or 0.0
    bid, ask = ticker.get("bid"), ticker.get("ask")

    spread_bps = 0.0
    if bid and ask and ask > 0:
        mid = (bid + ask) / 2.0
        if mid > 0:
            spread_bps = (ask - bid) / mid * 10_000.0

    # ccxt reports percentage as a percent, not a fraction.
    percentage = ticker.get("percentage")
    change = float(percentage) / 100.0 if percentage is not None else 0.0

    return {
        "price": float(price or 0.0),
        "change_pct": change,
        "quote_volume": float(ticker.get("quoteVolume") or 0.0),
        "spread_bps": spread_bps,
    }
