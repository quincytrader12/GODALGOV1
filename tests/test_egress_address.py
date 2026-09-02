"""Which address the venue actually saw.

Reported with the key test showing:

    reachable: 4611 markets            ok
    market data: BTC/USDT @ 77,195.62  ok
    credentials: -2015                 rejected for this IP address

and the address in the panel matching the one on the allow-list. Both facts
can be true at once, because they are about different connections.

ccxt builds its aiohttp connector with ``family=AF_UNSPEC`` and Happy Eyeballs
racing both families with zero delay. On a dual-stack connection the request
can leave over IPv6 while the panel's separate lookup reports the machine's
IPv4 address. The operator whitelists an address the venue never saw, and
every piece of evidence on screen says the address is right -- because it is
right, for a connection that is not the one being made.

Public market data succeeding does not contradict any of this: no allow-list
applies to it.

So the terminal measures the address through ccxt's own session instead of
inferring it, and says what the answer means in each case.
"""

from __future__ import annotations

import asyncio

from godalgo.ui.venue import _address_verdict, egress_address


class _Response:
    def __init__(self, status: int, text: str) -> None:
        self.status = status
        self._text = text

    async def text(self) -> str:
        return self._text

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False


class _Session:
    """Stands in for the session ccxt built, which is the whole point: the
    measurement must travel the same connector the venue call did."""

    def __init__(self, *answers) -> None:
        self._answers = list(answers)
        self.urls: list[str] = []

    def get(self, url, **_kwargs):
        self.urls.append(url)
        answer = self._answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


class _Exchange:
    def __init__(self, session) -> None:
        self.session = session


def test_reports_the_address_seen_through_ccxts_own_session():
    session = _Session(_Response(200, "102.135.244.201\n"))
    assert asyncio.run(egress_address(_Exchange(session))) == "102.135.244.201"
    assert session.urls, "it must actually go through the exchange's session"


def test_falls_through_when_a_provider_is_down():
    session = _Session(_Response(503, "nope"), _Response(200, "198.51.100.9"))
    assert asyncio.run(egress_address(_Exchange(session))) == "198.51.100.9"


def test_a_raising_session_returns_nothing_rather_than_propagating():
    """This runs inside an error path. It must not replace the venue's error
    with one of its own."""
    session = _Session(OSError("connection reset"), OSError("connection reset"))
    assert asyncio.run(egress_address(_Exchange(session))) is None


def test_an_exchange_with_no_session_is_handled():
    assert asyncio.run(egress_address(_Exchange(None))) is None
    assert asyncio.run(egress_address(object())) is None


def test_an_ipv6_egress_names_the_mismatch():
    """The case the report matches: address on screen correct, venue seeing
    a different one."""
    verdict = _address_verdict("2a0d:6fc2:6100:1::42")
    assert "IPv6" in verdict
    assert "2a0d:6fc2:6100:1::42" in verdict
    assert "allow-list" in verdict


def test_an_ipv4_egress_sends_the_operator_to_permissions():
    """If the measured address is the whitelisted one, the address is not the
    problem and saying 'check your IP' again wastes their time."""
    verdict = _address_verdict("102.135.244.201")
    assert "102.135.244.201" in verdict
    assert "Enable" in verdict and "Spot" in verdict


def test_an_unmeasurable_address_offers_the_manual_route():
    verdict = _address_verdict(None)
    assert "ifconfig.me" in verdict


def test_the_2015_message_names_all_three_causes():
    """One code, three causes. Leading with only the IP sent me chasing an
    address that was already correct."""
    from godalgo.ui.venue import _BINANCE_CODES

    _, explanation = _BINANCE_CODES["-2015"]
    assert "allow-list" in explanation
    assert "Enable Reading" in explanation
    assert "testnet" in explanation
    # And it uses what the other probes already established.
    assert "market data" in explanation
