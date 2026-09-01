"""Why market data is not arriving, layer by layer.

"Could not reach the venue" is true and useless. It covers DNS failure, a
corporate proxy, TLS interception by antivirus, a geo-block, a firewall, and a
broken ccxt bundle -- six problems with six different remedies, and no way to
tell them apart from the message.

This module tests the layers independently and reports the **raw** exception
from each. The ordering is the point: the first failure locates the fault.

    environment   what is installed, and whether a proxy is configured
    dns           can this machine resolve the venue's hostname
    stdlib https  a plain GET, using the system trust store and proxy settings
    aiohttp       the client ccxt actually uses, which by default ignores
                  proxy environment variables entirely
    ccxt          the library call the terminal makes

If stdlib succeeds and aiohttp fails, it is a proxy or TLS problem inside the
client rather than a network one. If both fail, the network or the region is
blocking it, and no change to this program will help. If everything up to ccxt
succeeds and ccxt fails, the fault is ours.

Nothing here is friendly-ified. Every failure carries its exception type and
message verbatim, because that is what makes a report actionable.
"""

from __future__ import annotations

import asyncio
import logging
import os
import platform
import socket
import sys
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.request import Request, urlopen

__all__ = ["DiagnosticStep", "run_diagnostics"]

logger = logging.getLogger(__name__)

_TIMEOUT = 15.0

# The public, unauthenticated liveness endpoint for each venue we default to.
_PING: dict[str, str] = {
    "binance": "https://api.binance.com/api/v3/ping",
    "binanceus": "https://api.binance.us/api/v3/ping",
    "kraken": "https://api.kraken.com/0/public/SystemStatus",
    "coinbase": "https://api.exchange.coinbase.com/time",
    "bybit": "https://api.bybit.com/v5/market/time",
    "okx": "https://www.okx.com/api/v5/public/time",
}

_PROXY_VARS = (
    "HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy",
    "ALL_PROXY", "all_proxy", "NO_PROXY", "no_proxy",
)


@dataclass
class DiagnosticStep:
    """One layer's result."""

    name: str
    ok: bool
    detail: str = ""
    error_type: str = ""
    elapsed_ms: float = 0.0
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "ok": self.ok, "detail": self.detail,
            "error_type": self.error_type,
            "elapsed_ms": round(self.elapsed_ms, 1), "data": self.data,
        }


def _host_of(url: str) -> str:
    return url.split("://", 1)[-1].split("/", 1)[0]


def _environment(exchange_id: str) -> DiagnosticStep:
    """What is installed and how the network is configured.

    Proxy variables are reported by name and presence, never by value: a proxy
    URL can embed credentials, and this output is meant to be pasted into a
    bug report.
    """
    data: dict[str, Any] = {
        "platform": f"{platform.system()} {platform.release()}",
        "python": sys.version.split()[0],
        "frozen": bool(getattr(sys, "frozen", False)),
        "exchange_id": exchange_id,
    }
    for name, module in (("ccxt", "ccxt"), ("aiohttp", "aiohttp")):
        try:
            data[name] = __import__(module).__version__
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            data[name] = f"IMPORT FAILED: {type(exc).__name__}: {exc}"

    proxies = [v for v in _PROXY_VARS if os.environ.get(v)]
    data["proxy_vars_set"] = proxies
    return DiagnosticStep(
        "environment", True,
        f"{data['platform']}, ccxt {data.get('ccxt')}"
        + (f", proxy configured ({', '.join(proxies)})" if proxies else ", no proxy"),
        data=data,
    )


def _dns(host: str) -> DiagnosticStep:
    started = time.monotonic()
    try:
        infos = socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        return DiagnosticStep(
            "dns", False,
            f"cannot resolve {host}: {exc}", type(exc).__name__,
            (time.monotonic() - started) * 1000,
        )
    addresses = sorted({i[4][0] for i in infos})[:4]
    return DiagnosticStep(
        "dns", True, f"{host} -> {', '.join(addresses)}",
        elapsed_ms=(time.monotonic() - started) * 1000,
        data={"addresses": addresses},
    )


def _stdlib_https(url: str) -> DiagnosticStep:
    """A plain GET with the standard library.

    Uses the system trust store and *does* honour proxy environment variables,
    which is exactly what makes it worth comparing against aiohttp.
    """
    started = time.monotonic()
    try:
        request = Request(url, headers={"User-Agent": "godalgo-diagnostics"})
        with urlopen(request, timeout=_TIMEOUT) as response:
            status = response.status
            body = response.read(200).decode("utf-8", "replace")
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        return DiagnosticStep(
            "stdlib_https", False, f"{exc}", type(exc).__name__,
            (time.monotonic() - started) * 1000,
        )
    elapsed = (time.monotonic() - started) * 1000
    if not 200 <= status < 300:
        return DiagnosticStep(
            "stdlib_https", False,
            f"HTTP {status} — a reply from something in the way, not from the "
            f"venue: {body[:100]!r}",
            f"HTTP{status}", elapsed,
        )
    return DiagnosticStep(
        "stdlib_https", True, f"HTTP {status}, {body[:80]!r}", elapsed_ms=elapsed,
    )


async def _aiohttp_https(url: str, *, trust_env: bool) -> DiagnosticStep:
    """The same GET through the client ccxt uses.

    Run twice by the caller -- with and without ``trust_env`` -- because the
    difference between those two runs *is* the diagnosis when a proxy is in
    play: ccxt defaults to ignoring proxy environment variables.
    """
    name = f"aiohttp_https{'_trust_env' if trust_env else ''}"
    started = time.monotonic()
    try:
        import aiohttp

        timeout = aiohttp.ClientTimeout(total=_TIMEOUT)
        async with (
            aiohttp.ClientSession(timeout=timeout, trust_env=trust_env) as session,
            session.get(url) as response,
        ):
            body = (await response.text())[:120]
            status = response.status
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        return DiagnosticStep(
            name, False, f"{exc}", type(exc).__name__,
            (time.monotonic() - started) * 1000,
        )

    elapsed = (time.monotonic() - started) * 1000
    # A response is not a success. An intercepting proxy answers with its own
    # 403 block page, and counting that as "reachable" would put a tick beside
    # the exact thing preventing the connection.
    if not 200 <= status < 300:
        return DiagnosticStep(
            name, False,
            f"HTTP {status} — a reply from something in the way, not from the "
            f"venue: {body!r}",
            f"HTTP{status}", elapsed,
        )
    return DiagnosticStep(
        name, True, f"HTTP {status}, {body[:80]!r}", elapsed_ms=elapsed,
    )


async def _ccxt(exchange_id: str) -> DiagnosticStep:
    started = time.monotonic()
    exchange = None
    try:
        import ccxt.async_support as accxt

        klass = getattr(accxt, exchange_id, None)
        if klass is None:
            return DiagnosticStep(
                "ccxt", False,
                f"ccxt has no exchange named {exchange_id!r}", "AttributeError",
                (time.monotonic() - started) * 1000,
            )
        exchange = klass({
            "enableRateLimit": True,
            "timeout": int(_TIMEOUT * 1000),
            # Honour the system proxy, which ccxt otherwise ignores.
            "aiohttp_trust_env": True,
        })
        markets = await exchange.load_markets()
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        return DiagnosticStep(
            "ccxt", False, f"{exc}"[:400], type(exc).__name__,
            (time.monotonic() - started) * 1000,
        )
    finally:
        if exchange is not None:
            try:
                await exchange.close()
            except Exception:
                logger.debug("failed closing diagnostic session", exc_info=True)

    return DiagnosticStep(
        "ccxt", True, f"{len(markets)} markets loaded",
        elapsed_ms=(time.monotonic() - started) * 1000,
        data={"markets": len(markets)},
    )


def _verdict(steps: list[DiagnosticStep]) -> str:
    """Name the fault and the remedy, from where the chain first broke."""
    by_name = {s.name: s for s in steps}

    def failed(name: str) -> bool:
        step = by_name.get(name)
        return step is not None and not step.ok

    if failed("dns"):
        return (
            "This machine cannot resolve the venue's hostname. That is DNS or "
            "a firewall, not the bot — a VPN, a corporate network or a DNS "
            "filter is the usual cause."
        )

    proxied = bool(by_name["environment"].data.get("proxy_vars_set"))
    if failed("stdlib_https") and failed("aiohttp_https_trust_env"):
        return (
            "Nothing on this machine can reach the venue over HTTPS, so no "
            "change to the bot will help. Most often the region is blocked "
            "(Binance restricts several countries — try binanceus or kraken), "
            "or a firewall/antivirus is intercepting the connection."
        )

    if by_name.get("aiohttp_https_trust_env") and by_name["aiohttp_https_trust_env"].ok \
            and failed("aiohttp_https"):
        return (
            "The connection only works when proxy settings are honoured. ccxt "
            "ignores them by default; the terminal now enables them, so this "
            "should be fixed — if data is still missing, report this output."
        )

    if failed("ccxt"):
        detail = by_name["ccxt"].detail
        if proxied:
            return (
                "Plain HTTPS works but ccxt fails, and a proxy is configured. "
                f"The raw error is: {detail[:200]}"
            )
        return (
            "The network is fine and ccxt itself is failing. This is a bug in "
            f"the terminal, not your setup. Raw error: {detail[:200]}"
        )

    return "Everything works: DNS, HTTPS and ccxt all reached the venue."


async def run_diagnostics(exchange_id: str = "binance") -> dict[str, Any]:
    """Run every layer and return the results with a verdict.

    Never raises. A diagnostic that can fail is not a diagnostic.
    """
    url = _PING.get(exchange_id) or f"https://api.{exchange_id}.com/"
    host = _host_of(url)

    steps = [_environment(exchange_id)]
    steps.append(await asyncio.to_thread(_dns, host))

    if steps[-1].ok:
        # Off the loop: urlopen is synchronous and would block everything
        # sharing it for the whole timeout.
        steps.append(await asyncio.to_thread(_stdlib_https, url))
        steps.append(await _aiohttp_https(url, trust_env=False))
        steps.append(await _aiohttp_https(url, trust_env=True))
        steps.append(await _ccxt(exchange_id))

    return {
        "exchange_id": exchange_id,
        "url": url,
        "steps": [s.to_dict() for s in steps],
        "ok": all(s.ok for s in steps),
        "verdict": _verdict(steps),
    }
