"""Alpaca: the whole tradable universe, and a paper account that is real.

Written natively rather than through ccxt, for one decisive reason: ccxt's
Alpaca binding covers the crypto endpoints only. Going through it would mean
adopting a broker and seeing a couple of dozen crypto pairs, when the account
can trade thousands of US equities and ETFs besides. The point of moving here
is the breadth, so the client has to speak the whole API.

The other reason to be here at all is the paper account. It is not a
simulation bolted on by us -- it is Alpaca's own matching against real market
data, reached with real keys over the real API, at ``paper-api.alpaca.markets``
instead of ``api.alpaca.markets``. One base URL apart. That makes it possible
to prove the entire pipeline -- key, data, decision, order, fill, journal --
before any money exists anywhere, which was never possible on a venue with no
test network.

Three things about this API that shape everything below:

* **Two asset classes with different rules.** Equities trade during market
  hours, honour a pattern-day-trader rule, and are commission-free. Crypto
  trades continuously, has no PDT rule, and charges a real fee. They also live
  on different data endpoints. ``is_crypto`` is the discriminator, and it is
  the presence of a ``/`` in the symbol: ``AAPL`` versus ``BTC/USD``.
* **There is no post-only.** The order type this project defaults to does not
  exist here. That is not a detail: on crypto it means the maker fee tier
  cannot be guaranteed, so the round-trip cost assumption changes and the
  feasibility floor moves with it. The mapping is explicit and documented
  rather than silent.
* **The market is usually closed.** For most of the week, an equities order is
  not a bad trade -- it is an impossible one. The clock is part of the client,
  not an afterthought.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

__all__ = [
    "AlpacaAsset",
    "AlpacaClient",
    "AlpacaConfig",
    "AlpacaError",
    "MarketClock",
    "is_crypto",
]

logger = logging.getLogger(__name__)

TRADING_PAPER = "https://paper-api.alpaca.markets"
TRADING_LIVE = "https://api.alpaca.markets"
DATA = "https://data.alpaca.markets"

_CRYPTO_PREFIX = "/v1beta3/crypto/us"


class AlpacaError(RuntimeError):
    """A request the venue refused, carrying what it said.

    Holds the status and the venue's own message because the remedy differs
    completely between them: 401 is the key, 403 is usually paper keys against
    the live endpoint, 422 is the order itself, and 429 is rate limiting that
    will pass on its own.
    """

    def __init__(self, status: int, message: str, *, path: str = "") -> None:
        super().__init__(f"HTTP {status} from {path or 'alpaca'}: {message}")
        self.status = status
        self.message = message
        self.path = path

    @property
    def is_auth(self) -> bool:
        return self.status in (401, 403)


def is_crypto(symbol: str) -> bool:
    """Whether this is a crypto pair rather than an equity.

    Alpaca names equities bare (``AAPL``) and crypto as a pair (``BTC/USD``),
    so the slash is the discriminator. Everything downstream branches on this:
    which data endpoint serves it, whether the market being closed matters,
    whether the PDT rule applies, and what a round trip costs.
    """
    return "/" in symbol


@dataclass(frozen=True, slots=True)
class MarketClock:
    """The venue's own clock, never this machine's.

    Asking the venue rather than computing from a timezone is deliberate:
    holidays, half days and early closes are the venue's business, and a bot
    that decides for itself that the market is open will be wrong a dozen times
    a year -- always on the days when being wrong is most expensive.
    """

    is_open: bool
    timestamp: datetime
    next_open: datetime | None = None
    next_close: datetime | None = None

    def tradeable(self, symbol: str) -> bool:
        """Whether this instrument can be traded right now.

        Crypto ignores the clock entirely; that is most of why it is the
        sensible thing to run first.
        """
        return True if is_crypto(symbol) else self.is_open

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_open": self.is_open,
            "timestamp": self.timestamp.isoformat(),
            "next_open": self.next_open.isoformat() if self.next_open else None,
            "next_close": self.next_close.isoformat() if self.next_close else None,
        }


@dataclass(frozen=True, slots=True)
class AlpacaAsset:
    """One tradable instrument, as the venue describes it."""

    symbol: str
    name: str = ""
    asset_class: str = "us_equity"
    tradable: bool = True
    shortable: bool = False
    fractionable: bool = False
    marginable: bool = False
    exchange: str = ""

    @property
    def is_crypto(self) -> bool:
        return self.asset_class == "crypto" or is_crypto(self.symbol)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol, "name": self.name,
            "asset_class": self.asset_class, "tradable": self.tradable,
            "shortable": self.shortable, "fractionable": self.fractionable,
            "exchange": self.exchange,
        }


@dataclass
class AlpacaConfig:
    """Where to connect and as whom."""

    key_id: str = ""
    secret_key: str = ""
    paper: bool = True
    """Paper by default, everywhere, always.

    The live endpoint is one string away, and that string is never the
    default. Nothing in this project should reach real money because a field
    was left unset.
    """

    feed: str = "iex"
    """Equity data feed.

    ``iex`` is what a free account gets and it is genuinely partial -- one
    venue's prints, a few percent of consolidated volume, with gaps where a
    symbol simply did not trade there. ``sip`` is the full consolidated tape
    and needs a paid subscription. Naming the default here rather than burying
    it matters, because thin data does not announce itself: it looks like a
    quiet market, and a strategy will happily fit to the gaps.
    """

    timeout: float = 15.0

    @property
    def trading_base(self) -> str:
        return TRADING_PAPER if self.paper else TRADING_LIVE

    @property
    def configured(self) -> bool:
        return bool(self.key_id and self.secret_key)


@dataclass
class AlpacaClient:
    """Async access to the trading and market-data APIs.

    One client for both asset classes and both endpoints, so callers never
    have to know which host answers what.
    """

    config: AlpacaConfig = field(default_factory=AlpacaConfig)
    _client: Any = field(default=None, init=False, repr=False)

    # -- plumbing ----------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        return {
            "APCA-API-KEY-ID": self.config.key_id,
            "APCA-API-SECRET-KEY": self.config.secret_key,
            "accept": "application/json",
        }

    async def _session(self) -> Any:
        if self._client is None:
            import httpx

            # trust_env so a machine behind a proxy works. ccxt's failure to do
            # this cost a full debugging round on the previous venue: every
            # request failed while the browser beside it worked.
            self._client = httpx.AsyncClient(
                timeout=self.config.timeout, trust_env=True,
                headers=self._headers(),
            )
        return self._client

    async def _request(
        self, method: str, base: str, path: str, **kwargs: Any
    ) -> Any:
        import httpx

        client = await self._session()
        url = f"{base}{path}"
        try:
            response = await client.request(method, url, **kwargs)
        except httpx.HTTPError as exc:
            raise AlpacaError(0, f"{type(exc).__name__}: {exc}", path=path) from exc

        if response.status_code >= 400:
            # The venue's own message, not a generic one. Alpaca is specific
            # about why an order was refused and discarding that would throw
            # away the only useful part.
            try:
                detail = response.json().get("message") or response.text
            except ValueError:
                detail = response.text
            raise AlpacaError(response.status_code, str(detail)[:300], path=path)

        if not response.content:
            return None
        return response.json()

    async def close(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            try:
                await client.aclose()
            except Exception:  # noqa: BLE001 - closing must not raise
                logger.debug("could not close the Alpaca client", exc_info=True)

    # -- account -----------------------------------------------------------

    async def account(self) -> dict[str, Any]:
        """Equity, buying power, and the flags that constrain trading."""
        return await self._request("GET", self.config.trading_base, "/v2/account")

    async def clock(self) -> MarketClock:
        """Whether the equities market is open, from the venue itself."""
        raw = await self._request("GET", self.config.trading_base, "/v2/clock")
        return MarketClock(
            is_open=bool(raw.get("is_open")),
            timestamp=_parse_time(raw.get("timestamp")) or datetime.now(UTC),
            next_open=_parse_time(raw.get("next_open")),
            next_close=_parse_time(raw.get("next_close")),
        )

    async def assets(
        self, asset_class: str = "us_equity", *, tradable_only: bool = True
    ) -> list[AlpacaAsset]:
        """Every instrument of a class the account may trade.

        This is the call that makes the universe the account's real universe
        rather than a list someone typed. It is thousands of rows, so callers
        should hold the result rather than asking per decision.
        """
        raw = await self._request(
            "GET", self.config.trading_base, "/v2/assets",
            params={"status": "active", "asset_class": asset_class},
        )
        assets = [
            AlpacaAsset(
                symbol=row.get("symbol", ""),
                name=row.get("name", ""),
                asset_class=row.get("class", asset_class),
                tradable=bool(row.get("tradable")),
                shortable=bool(row.get("shortable")),
                fractionable=bool(row.get("fractionable")),
                marginable=bool(row.get("marginable")),
                exchange=row.get("exchange", ""),
            )
            for row in (raw or [])
        ]
        return [a for a in assets if a.tradable] if tradable_only else assets

    async def universe(self) -> list[AlpacaAsset]:
        """Everything tradable, both asset classes, in one list.

        Asked concurrently: two independent round trips that would otherwise be
        serialised for no reason.
        """
        equities, crypto = await asyncio.gather(
            self.assets("us_equity"), self.assets("crypto"),
            return_exceptions=True,
        )
        out: list[AlpacaAsset] = []
        for result in (equities, crypto):
            if isinstance(result, BaseException):
                logger.debug("an asset class could not be listed", exc_info=result)
                continue
            out.extend(result)
        return out

    # -- market data -------------------------------------------------------

    async def snapshots(self, symbols: list[str]) -> dict[str, dict[str, float]]:
        """Latest trade, quote and daily bar per symbol.

        Split by asset class because they are served by different endpoints,
        then requested concurrently and merged, so a watchlist mixing AAPL and
        BTC/USD costs two round trips rather than one per symbol.
        """
        equities = [s for s in symbols if not is_crypto(s)]
        crypto = [s for s in symbols if is_crypto(s)]

        tasks = []
        if equities:
            tasks.append(self._stock_snapshots(equities))
        if crypto:
            tasks.append(self._crypto_snapshots(crypto))
        if not tasks:
            return {}

        out: dict[str, dict[str, float]] = {}
        for result in await asyncio.gather(*tasks, return_exceptions=True):
            if isinstance(result, BaseException):
                logger.debug("a snapshot request failed", exc_info=result)
                continue
            out.update(result)
        return out

    async def _stock_snapshots(self, symbols: list[str]) -> dict[str, dict[str, float]]:
        raw = await self._request(
            "GET", DATA, "/v2/stocks/snapshots",
            params={"symbols": ",".join(symbols), "feed": self.config.feed},
        )
        rows = raw.get("snapshots", raw) if isinstance(raw, dict) else {}
        return {s: _row(v) for s, v in rows.items() if isinstance(v, dict)}

    async def _crypto_snapshots(self, symbols: list[str]) -> dict[str, dict[str, float]]:
        raw = await self._request(
            "GET", DATA, f"{_CRYPTO_PREFIX}/snapshots",
            params={"symbols": ",".join(symbols)},
        )
        rows = (raw or {}).get("snapshots", {})
        return {s: _row(v) for s, v in rows.items() if isinstance(v, dict)}

    async def bars(
        self, symbol: str, timeframe: str = "1Min", limit: int = 1000
    ) -> list[dict[str, float]]:
        """Recent OHLCV, for seeding the engine so it can signal now.

        Returned as plain rows; turning them into a frame is the caller's job,
        which keeps pandas out of the client.
        """
        if is_crypto(symbol):
            path, params = f"{_CRYPTO_PREFIX}/bars", {}
        else:
            path, params = "/v2/stocks/bars", {"feed": self.config.feed}
        params |= {"symbols": symbol, "timeframe": timeframe, "limit": limit}

        raw = await self._request("GET", DATA, path, params=params)
        rows = (raw or {}).get("bars", {})
        series = rows.get(symbol, []) if isinstance(rows, dict) else []
        return [
            {
                "timestamp": _parse_time(b.get("t")),
                "open": float(b.get("o", 0.0)), "high": float(b.get("h", 0.0)),
                "low": float(b.get("l", 0.0)), "close": float(b.get("c", 0.0)),
                "volume": float(b.get("v", 0.0)),
            }
            for b in series
        ]

    # -- positions and orders ---------------------------------------------

    async def positions(self) -> list[dict[str, Any]]:
        return await self._request(
            "GET", self.config.trading_base, "/v2/positions"
        ) or []

    async def orders(self, status: str = "open") -> list[dict[str, Any]]:
        return await self._request(
            "GET", self.config.trading_base, "/v2/orders",
            params={"status": status, "limit": 500},
        ) or []

    async def submit_order(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._request(
            "POST", self.config.trading_base, "/v2/orders", json=payload,
        )

    async def cancel_order(self, order_id: str) -> None:
        await self._request(
            "DELETE", self.config.trading_base, f"/v2/orders/{order_id}"
        )


def _parse_time(value: Any) -> datetime | None:
    """Alpaca timestamps, tolerantly.

    They arrive as RFC 3339, sometimes with nanosecond precision that
    ``fromisoformat`` rejects on older Pythons, and a bad timestamp must not
    take down a price update.
    """
    if not isinstance(value, str) or not value:
        return None
    text = value.replace("Z", "+00:00")
    if "." in text:
        head, _, tail = text.partition(".")
        digits = "".join(c for c in tail if c.isdigit())[:6]
        offset = tail[len(digits):] if len(tail) > len(digits) else ""
        offset = offset.lstrip("0123456789")
        text = f"{head}.{digits or '0'}{offset}"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _row(snapshot: dict[str, Any]) -> dict[str, float]:
    """Reduce a snapshot to the handful of numbers the terminal renders.

    A snapshot carries the latest trade, latest quote, minute bar, daily bar
    and previous daily bar. Passing all of that through per symbol, once a
    second, would be a frame far larger than the panel needs.
    """
    trade = snapshot.get("latestTrade") or {}
    quote = snapshot.get("latestQuote") or {}
    daily = snapshot.get("dailyBar") or {}
    previous = snapshot.get("prevDailyBar") or {}

    price = float(trade.get("p") or daily.get("c") or 0.0)
    bid = float(quote.get("bp") or 0.0)
    ask = float(quote.get("ap") or 0.0)

    spread_bps = 0.0
    if bid > 0 and ask > 0:
        mid = (bid + ask) / 2.0
        if mid > 0:
            spread_bps = (ask - bid) / mid * 10_000.0

    # Against the previous session's close, which is what "24h change" means
    # for an instrument that does not trade overnight.
    reference = float(previous.get("c") or 0.0) or float(daily.get("o") or 0.0)
    change = (price - reference) / reference if reference > 0 and price > 0 else 0.0

    volume = float(daily.get("v") or 0.0)
    return {
        "price": price,
        "change_pct": change,
        "quote_volume": volume * price,
        "spread_bps": spread_bps,
    }
