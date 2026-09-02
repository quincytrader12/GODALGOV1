"""Read-only checks against a real exchange.

This module answers one question the rest of the system could not: *is this
thing actually talking to Binance?* It does so without placing an order and
without needing a funded account, which matters because the alternative is to
fund an account in order to find out whether funding it was a good idea.

Three probes, in increasing order of what they prove:

1. **Reachability** -- ``load_markets`` against the public API. No key. Proves
   the venue is up, the region is not blocked, and the network path works.
2. **Market data** -- ``fetch_ticker``. Still no key. Proves live prices are
   arriving, which is the whole input to the strategy stack.
3. **Credentials** -- ``fetch_balance``. Uses the key, reads only. Proves the
   key, the secret, the signature, the clock, and the IP allow-list all work.

Nothing here can place, amend or cancel an order. That is a property of the
module, not a convention: no call in it takes a side, a size or a price.

Error classification is the substance. ``ccxt`` surfaces most failures as one
of a handful of exception types, but the *actionable* distinction -- a bad
secret, a clock that has drifted, an IP that is not on the allow-list -- lives
in the venue's own numeric code. Reporting "authentication failed" for all
three sends the operator to re-paste a key that was always correct.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from godalgo.ui.credentials import ExchangeCredential
    from godalgo.ui.events import EventLog

__all__ = [
    "ProbeResult", "VenueProbe", "classify_error", "normalise_exchange_id",
    "raw_error", "suggest_exchange_ids",
]

logger = logging.getLogger(__name__)

_DEFAULT_SYMBOL = "BTC/USDT"

# Venue error codes worth naming. The message a user needs is different in
# every one of these cases, and ccxt collapses several of them into a single
# exception type.
_BINANCE_CODES: dict[str, tuple[str, str]] = {
    "-2015": (
        "ip_not_allowed",
        (
            "Binance rejected the key for this IP address. Add the address "
            "shown in Connections to the key's allow-list, or check the key "
            "has 'Enable Reading' and Spot trading enabled. A domestic "
            "connection is usually dynamic, so an address whitelisted "
            "correctly can stop matching days later."
        ),
    ),
    "-2014": ("bad_key_format", "The API key format is not valid — re-copy it."),
    "-2008": ("bad_key", "Binance does not recognise this API key."),
    "-1022": (
        "bad_signature",
        (
            "The signature was rejected, which almost always means the API "
            "secret is wrong or has a stray space. Re-paste the secret."
        ),
    ),
    "-1021": (
        "clock_skew",
        (
            "Your PC's clock is outside Binance's tolerance. Sync it: "
            "Settings → Time & language → Date & time → 'Sync now'."
        ),
    ),
    "-1003": ("rate_limited", "Too many requests — the venue is rate-limiting."),
    "-1013": ("filter_failure", "The order breached a venue filter."),
}


@dataclass(frozen=True, slots=True)
class ProbeResult:
    """The outcome of one check."""

    name: str
    ok: bool
    detail: str = ""
    elapsed_ms: float = 0.0
    kind: str = ""
    """Machine-readable failure class, empty on success. See ``classify_error``."""

    data: dict[str, Any] = field(default_factory=dict)
    """Small facts worth showing: a price, a balance count, a market count."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# What people type, and what ccxt calls it. ccxt ids are lowercase and have no
# punctuation; "Binance" fails with a message that names an attribute rather
# than the mistake, which is not something an operator can be expected to
# translate.
_ALIASES: dict[str, str] = {
    "binance.us": "binanceus",
    "binance us": "binanceus",
    "binanceus.com": "binanceus",
    "coinbasepro": "coinbase",
    "coinbase pro": "coinbase",
    "gdax": "coinbase",
    "kucoin.com": "kucoin",
    "okex": "okx",
    "bybit.com": "bybit",
}


def normalise_exchange_id(value: str) -> str:
    """Turn what a person typed into the id ccxt expects.

    ccxt exchange ids are lowercase with no spaces or dots. Passing "Binance"
    through raises ``module 'ccxt.async_support' has no attribute 'Binance'``,
    which names an attribute rather than the mistake and is not something an
    operator should have to decode. Worse, an id like that can be *stored* --
    and then every later lookup fails identically, so a single capital letter
    reads as the exchange being down.
    """
    cleaned = (value or "").strip().lower()
    cleaned = _ALIASES.get(cleaned, cleaned)
    # ccxt ids contain only letters and digits.
    return "".join(c for c in cleaned if c.isalnum())


def suggest_exchange_ids(value: str, limit: int = 3) -> list[str]:
    """Ids close to what was typed, for an error message worth reading."""
    import difflib

    import ccxt

    return difflib.get_close_matches(
        normalise_exchange_id(value), ccxt.exchanges, n=limit, cutoff=0.5
    )


def known_exchange(exchange_id: str) -> bool:
    import ccxt

    return exchange_id in ccxt.exchanges


def raw_error(exc: BaseException) -> str:
    """The exception, as thrown, for a bug report.

    The friendly explanations below are for acting on; this is for diagnosing.
    Reporting only the friendly text was a mistake -- "could not reach the
    venue" covers DNS failure, a proxy, TLS interception, a geo-block and a
    firewall, and discards the one string that tells them apart.
    """
    return f"{type(exc).__name__}: {exc}"[:300]


def classify_error(exc: BaseException) -> tuple[str, str]:
    """Turn an exchange exception into ``(kind, human explanation)``.

    The venue's numeric code is checked before the exception type, because
    ccxt maps several genuinely different failures onto ``AuthenticationError``
    and the remedy differs for each.
    """
    import ccxt

    text = str(exc)
    for code, (kind, explanation) in _BINANCE_CODES.items():
        # The code appears as ``"code":-2015`` in the JSON body ccxt attaches.
        if f'"code":{code}' in text.replace(" ", "") or f"code={code}" in text:
            return kind, explanation

    if isinstance(exc, ccxt.AuthenticationError):
        return "auth_failed", (
            "The venue rejected these credentials. Check the key and secret, "
            "and that the key has read permission."
        )
    if isinstance(exc, ccxt.PermissionDenied):
        return "permission_denied", (
            "The key is valid but not permitted to do this. Check its "
            "permissions and IP allow-list."
        )
    if isinstance(exc, ccxt.RateLimitExceeded):
        return "rate_limited", "The venue is rate-limiting this key."
    if isinstance(exc, ccxt.BadSymbol):
        return "bad_symbol", "The venue does not list that symbol."
    if isinstance(exc, ccxt.NetworkError):
        # Covers timeouts, DNS failures, proxies and TLS errors alike.
        if "451" in text:
            return "geo_blocked", (
                "The venue refused the connection for this region (HTTP 451). "
                "Binance blocks some countries; binance.us or another venue "
                "may be required."
            )
        return "unreachable", (
            "Could not reach the venue. Check the internet connection, and "
            "whether a firewall or VPN is intercepting the request."
        )
    if isinstance(exc, ccxt.ExchangeError):
        return "venue_error", f"The venue returned an error: {text[:200]}"
    return "unknown", text[:200]


@dataclass
class VenueProbe:
    """Runs read-only checks and reports them as events.

    Args:
        events: Where results are recorded. The probe never raises to the
            caller; a failure is an event and a returned result.
        timeout: Per-call ceiling in seconds. Short, because this runs behind
            a button the operator is waiting on.
    """

    events: EventLog
    timeout: float = 20.0

    async def check_public(
        self, exchange_id: str = "binance", symbol: str = _DEFAULT_SYMBOL
    ) -> list[ProbeResult]:
        """Reachability and live market data. Requires no credentials.

        This is the check worth running first and worth running always: it
        proves the data path the strategies consume, and it costs nothing and
        risks nothing.
        """
        import ccxt.async_support as accxt

        requested, exchange_id = exchange_id, normalise_exchange_id(exchange_id)
        results: list[ProbeResult] = []

        if not known_exchange(exchange_id):
            hint = suggest_exchange_ids(requested)
            detail = (
                f"{requested!r} is not an exchange ccxt knows"
                + (f" — did you mean {' or '.join(hint)}?" if hint else "")
            )
            self.events.error("venue", "unknown exchange", detail)
            return [ProbeResult("reachable", False, detail, kind="bad_exchange")]

        exchange = getattr(accxt, exchange_id)({
            "enableRateLimit": True,
            "timeout": int(self.timeout * 1000),
            # ccxt ignores the system proxy without this.
            "aiohttp_trust_env": True,
        })

        try:
            results.append(await self._market_count(exchange, exchange_id))
            if results[-1].ok:
                results.append(await self._ticker(exchange, exchange_id, symbol))
        finally:
            # ccxt.async_support holds an aiohttp session; not closing it leaks
            # a connector per probe and warns on shutdown.
            await _close(exchange)
        return results

    async def check_credentials(
        self, credential: ExchangeCredential, symbol: str = _DEFAULT_SYMBOL
    ) -> list[ProbeResult]:
        """Prove the key works, by reading the account balance.

        ``fetch_balance`` is the right call for this: it is authenticated, so
        it exercises the key, the secret, the signature and the clock; it is
        rejected by an IP allow-list, so it exercises that too; and it cannot
        move anything. A zero balance is a pass -- an empty account still
        proves the credential is good.
        """
        import ccxt.async_support as accxt

        results = await self.check_public(credential.exchange_id, symbol)
        if not results[0].ok:
            # No point testing a key against a venue we cannot reach; the
            # failure would be reported as an auth problem it is not.
            return results

        exchange_id = normalise_exchange_id(credential.exchange_id)
        if not known_exchange(exchange_id):
            return results
        try:
            exchange = getattr(accxt, exchange_id)({
                "apiKey": credential.api_key,
                "secret": credential.api_secret,
                "password": credential.passphrase or None,
                "enableRateLimit": True,
                "timeout": int(self.timeout * 1000),
                # ccxt ignores the system proxy without this.
                "aiohttp_trust_env": True,
            })
        except AttributeError:
            return results

        if credential.testnet:
            # The venue's test network: real API, fake money. The one way to
            # exercise the full order path on Binance without funding anything.
            with contextlib.suppress(Exception):
                exchange.set_sandbox_mode(True)

        try:
            results.append(await self._balance(exchange, credential))
        finally:
            await _close(exchange)
        return results

    # -- individual probes -------------------------------------------------

    async def _market_count(self, exchange: Any, exchange_id: str) -> ProbeResult:
        started = time.monotonic()
        try:
            markets = await asyncio.wait_for(
                exchange.load_markets(), timeout=self.timeout
            )
        except Exception as exc:  # noqa: BLE001 - classified and reported
            kind, explanation = _classify(exc)
            self.events.error(
                "venue", f"cannot reach {exchange_id}",
                f"{explanation} [{raw_error(exc)}]",
            )
            return ProbeResult(
                "reachable", False, explanation,
                (time.monotonic() - started) * 1000, kind,
            )

        elapsed = (time.monotonic() - started) * 1000
        self.events.good(
            "venue", f"{exchange_id} reachable",
            f"{len(markets)} markets listed, {elapsed:.0f}ms",
        )
        return ProbeResult(
            "reachable", True, f"{len(markets)} markets", elapsed,
            data={"markets": len(markets)},
        )

    async def _ticker(
        self, exchange: Any, exchange_id: str, symbol: str
    ) -> ProbeResult:
        started = time.monotonic()
        try:
            ticker = await asyncio.wait_for(
                exchange.fetch_ticker(symbol), timeout=self.timeout
            )
        except Exception as exc:  # noqa: BLE001 - classified and reported
            kind, explanation = _classify(exc)
            self.events.warn(
                "data", f"no market data for {symbol}",
                f"{explanation} [{raw_error(exc)}]",
            )
            return ProbeResult(
                "market_data", False, explanation,
                (time.monotonic() - started) * 1000, kind,
            )

        elapsed = (time.monotonic() - started) * 1000
        price = ticker.get("last") or ticker.get("close") or 0.0
        self.events.good(
            "data", f"{symbol} @ {price:,.2f}",
            f"live from {exchange_id}, {elapsed:.0f}ms",
        )
        return ProbeResult(
            "market_data", True, f"{symbol} @ {price:,.2f}", elapsed,
            data={"symbol": symbol, "price": price,
                  "bid": ticker.get("bid"), "ask": ticker.get("ask")},
        )

    async def _balance(
        self, exchange: Any, credential: ExchangeCredential
    ) -> ProbeResult:
        started = time.monotonic()
        try:
            balance = await asyncio.wait_for(
                exchange.fetch_balance(), timeout=self.timeout
            )
        except Exception as exc:  # noqa: BLE001 - classified and reported
            kind, explanation = _classify(exc)
            self.events.error(
                "credentials",
                f"{credential.exchange_id} rejected the key",
                f"{explanation} [{raw_error(exc)}]",
            )
            return ProbeResult(
                "credentials", False, explanation,
                (time.monotonic() - started) * 1000, kind,
            )

        elapsed = (time.monotonic() - started) * 1000
        totals = balance.get("total") or {}
        funded = {k: v for k, v in totals.items() if v}
        free_usdt = float((balance.get("free") or {}).get("USDT") or 0.0)

        # A zero balance is still a pass. The credential is proven either way,
        # and saying so is the point of this whole module.
        note = (
            f"{len(funded)} asset(s) with a balance"
            if funded else "account is empty — the key still works"
        )
        self.events.good(
            "credentials",
            f"{credential.exchange_id} key verified"
            + (" (testnet)" if credential.testnet else ""),
            f"{note}, {elapsed:.0f}ms",
        )
        return ProbeResult(
            "credentials", True, note, elapsed,
            data={
                "funded_assets": sorted(funded)[:12],
                "free_usdt": free_usdt,
                "testnet": credential.testnet,
                # Never the balance figures themselves beyond USDT: this is
                # rendered in a browser and screenshotted.
            },
        )


async def _close(exchange: Any) -> None:
    try:
        await exchange.close()
    except Exception:
        logger.debug("failed to close exchange session", exc_info=True)


def _classify(exc: BaseException) -> tuple[str, str]:
    """``classify_error`` with a timeout case ccxt does not model."""
    if isinstance(exc, TimeoutError | asyncio.TimeoutError):
        return "timeout", (
            "The venue did not respond in time. This is usually the network, "
            "not the key."
        )
    return classify_error(exc)
