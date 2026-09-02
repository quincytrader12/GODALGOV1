"""What IP address the exchange sees.

Binance rejects a key used from an address that is not on its allow-list, and
returns ``-2015`` for it. That error names the problem but not the fact you
need, which is *what this machine's address currently is* -- and on a domestic
connection that is a moving target. ISPs reassign dynamic addresses on a router
reboot, a line drop, or their own schedule, so a key that was whitelisted
correctly stops working days later with nothing having been changed. Looking it
up in a browser to compare against the allow-list is a chore the terminal can
simply do.

Three things this has to get right:

* **IPv4, forced.** The allow-list takes IPv4. A machine that prefers IPv6 will
  otherwise report an address that cannot be entered there, which is worse than
  reporting nothing -- it looks like an answer.
* **More than one provider.** These are free third-party echo services; any one
  of them can be down, rate-limited, or blocked by a corporate network. Falling
  through a short list turns a provider outage into a delay rather than a
  failure.
* **Never raise, and never block for long.** This is a convenience readout. It
  must not be able to stall a panel or take down a request handler.

Nothing here sends anything. Each provider is asked to echo back the address it
sees, which is what any web server already learns from the connection.
"""

from __future__ import annotations

import logging
import socket
import time
from dataclasses import dataclass
from urllib.request import Request, urlopen

__all__ = ["PublicAddress", "public_ip"]

logger = logging.getLogger(__name__)

_PROVIDERS = (
    "https://api.ipify.org",
    "https://ifconfig.me/ip",
    "https://icanhazip.com",
)

_TIMEOUT = 4.0
_CACHE_SECONDS = 300.0
"""How long a looked-up address is reused.

Long enough that opening the panel repeatedly does not hammer a free service,
short enough that a reconnection which changes the address is noticed within
the same sitting.
"""

_cache: tuple[float, "PublicAddress"] | None = None


@dataclass(frozen=True, slots=True)
class PublicAddress:
    """The address, or why it could not be found."""

    ip: str | None
    source: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {"ip": self.ip, "source": self.source, "error": self.error}


def _looks_like_ipv4(text: str) -> bool:
    """Whether this is an address that can actually be pasted into Binance.

    Checked rather than trusted: these endpoints are third parties, and one
    returning an error page, an IPv6 address or an HTML block page must not end
    up displayed as though it were an answer.
    """
    try:
        socket.inet_aton(text)
    except OSError:
        return False
    return text.count(".") == 3


def public_ip(*, force: bool = False) -> PublicAddress:
    """Look up this machine's public IPv4 address.

    Blocking. Call it from a worker thread -- there is a network round trip in
    here and the event loop must not wait on it.

    Args:
        force: Ignore the cache and ask again. What a "recheck" button wants,
            since the reason to press it is a suspicion the address moved.
    """
    global _cache

    if not force and _cache is not None:
        looked_up_at, cached = _cache
        if time.monotonic() - looked_up_at < _CACHE_SECONDS:
            return cached

    errors: list[str] = []
    for url in _PROVIDERS:
        try:
            request = Request(url, headers={"User-Agent": "godalgo"})
            with urlopen(request, timeout=_TIMEOUT) as response:
                if not 200 <= response.status < 300:
                    errors.append(f"{url}: HTTP {response.status}")
                    continue
                text = response.read(64).decode("utf-8", "replace").strip()
        except Exception as exc:  # noqa: BLE001 - try the next provider
            errors.append(f"{url}: {type(exc).__name__}")
            continue

        if _looks_like_ipv4(text):
            found = PublicAddress(ip=text, source=url.split("/")[2])
            _cache = (time.monotonic(), found)
            return found
        errors.append(f"{url}: not an IPv4 address")

    logger.debug("could not determine the public address: %s", "; ".join(errors))
    # Not cached. A failure is usually transient -- a provider blip or a network
    # that is briefly down -- and caching it would keep the panel empty for five
    # minutes after the connection came back.
    return PublicAddress(
        ip=None,
        error=(
            "Could not reach any address-lookup service. Check the connection, "
            "or read it from https://ifconfig.me in a browser."
        ),
    )
