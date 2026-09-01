"""Layered connection diagnostics.

The motivating failure: "could not reach the venue" was reported for a DNS
failure, a corporate proxy, TLS interception, a geo-block and a firewall
alike -- five problems, five different remedies, one useless message. These
tests cover the property that makes the diagnostic worth having: the first
failing layer locates the fault, and every failure carries its raw exception.
"""

from __future__ import annotations

import asyncio

import pytest

from godalgo.ui.diagnostics import DiagnosticStep, _verdict, run_diagnostics
from godalgo.ui.venue import raw_error


def _steps(**outcomes: bool) -> list[DiagnosticStep]:
    """Build a step list, defaulting anything unnamed to passing."""
    order = ["environment", "dns", "stdlib_https", "aiohttp_https",
             "aiohttp_https_trust_env", "ccxt"]
    return [
        DiagnosticStep(name, outcomes.get(name, True), detail=f"{name} detail")
        for name in order
    ]


# --- the verdict points at the right layer ---------------------------------

def test_dns_failure_is_not_blamed_on_the_bot():
    verdict = _verdict(_steps(dns=False))
    assert "DNS" in verdict
    assert "not the bot" in verdict


def test_a_total_https_failure_suggests_another_venue():
    """Binance blocks several regions. No change to this program fixes that,
    and saying so saves the operator from debugging our code."""
    verdict = _verdict(_steps(stdlib_https=False, aiohttp_https=False,
                              aiohttp_https_trust_env=False, ccxt=False))
    assert "no change to the bot will help" in verdict
    assert "binanceus" in verdict


def test_a_proxy_only_failure_is_identified_as_such():
    """The signature is precise: works with proxy settings honoured, fails
    without. ccxt ignores them by default, which is why this case exists."""
    verdict = _verdict(_steps(aiohttp_https=False))
    assert "proxy" in verdict.lower()


def test_ccxt_failing_alone_is_reported_as_our_bug():
    """If DNS, HTTPS and the proxy all work and only ccxt fails, it is not the
    operator's network and they should not be sent to debug it."""
    verdict = _verdict(_steps(ccxt=False))
    assert "bug in" in verdict
    assert "not your setup" in verdict


def test_everything_passing_says_so():
    assert "Everything works" in _verdict(_steps())


# --- it never raises, whatever the network does ----------------------------

def test_diagnostics_never_raise_on_an_unknown_exchange():
    """A diagnostic that can fail is not a diagnostic."""
    report = asyncio.run(run_diagnostics("not_a_real_venue"))
    assert "steps" in report
    assert report["verdict"]


def test_dns_failure_short_circuits_the_later_layers(monkeypatch):
    """No point testing HTTPS against a host that does not resolve; the
    failure would be reported as something it is not."""
    import godalgo.ui.diagnostics as diag

    def no_dns(host):
        return DiagnosticStep("dns", False, f"cannot resolve {host}", "socket.gaierror")

    monkeypatch.setattr(diag, "_dns", no_dns)
    report = asyncio.run(run_diagnostics("binance"))

    names = [s["name"] for s in report["steps"]]
    assert names == ["environment", "dns"]
    assert report["ok"] is False


# --- what it reports, and what it must not ---------------------------------

def test_a_proxy_url_is_never_included_in_the_report(monkeypatch):
    """Proxy URLs embed credentials, and this output exists to be pasted into
    a bug report. Only the variable names are reported."""
    secret = "http://user:hunter2@proxy.internal:8080"
    monkeypatch.setenv("HTTPS_PROXY", secret)

    import godalgo.ui.diagnostics as diag

    step = diag._environment("binance")
    blob = str(step.to_dict())
    assert "hunter2" not in blob
    assert "proxy.internal" not in blob
    assert "HTTPS_PROXY" in blob


def test_the_environment_step_reports_versions_and_platform():
    import godalgo.ui.diagnostics as diag

    data = diag._environment("binance").data
    assert data["exchange_id"] == "binance"
    assert data["python"]
    assert "ccxt" in data


def test_raw_error_keeps_the_exception_type_and_message():
    """The friendly explanation is for acting on; this is for diagnosing.
    Reporting only the friendly text is what made the original bug opaque."""
    text = raw_error(ValueError("connection reset by peer"))
    assert "ValueError" in text
    assert "connection reset by peer" in text


def test_raw_error_is_bounded():
    """A ccxt error body can run to kilobytes and this goes on screen."""
    assert len(raw_error(RuntimeError("x" * 5000))) <= 300


@pytest.mark.parametrize("exchange_id", ["binance", "binanceus", "kraken"])
def test_known_venues_have_a_public_ping_endpoint(exchange_id):
    """The probe must hit something unauthenticated, or a credential problem
    would masquerade as a network one."""
    from godalgo.ui.diagnostics import _PING

    assert _PING[exchange_id].startswith("https://")


def test_a_block_page_is_not_counted_as_reachable():
    """Found by running the diagnostic: an intercepting proxy answers with its
    own 403, and treating any HTTP reply as success put a tick beside the
    exact thing preventing the connection."""
    import godalgo.ui.diagnostics as diag

    class _Response:
        status = 403

        async def text(self):
            return "Host not in allowlist"

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

    class _Session:
        def __init__(self, *a, **k):
            pass

        def get(self, url):
            return _Response()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

    import aiohttp
    original = aiohttp.ClientSession
    aiohttp.ClientSession = _Session
    try:
        step = asyncio.run(diag._aiohttp_https("https://example.invalid", trust_env=False))
    finally:
        aiohttp.ClientSession = original

    assert step.ok is False
    assert "403" in step.detail
    assert "not from the venue" in step.detail or "in the way" in step.detail


def test_the_diagnostic_is_reachable_by_typing_a_url(tmp_path):
    """A control someone cannot find is a control that does not exist.

    The button lives in a scrolling panel and only appeared in a build the
    operator may never have run. A GET survives any layout, needs no scrolling,
    and its plain-text output can be pasted into a report without a screenshot.
    """
    from fastapi.testclient import TestClient

    from godalgo.ui.credentials import CredentialStore
    from godalgo.ui.journal import TradingJournal
    from godalgo.ui.server import UIBridge, create_app

    bridge = UIBridge(
        credentials=CredentialStore(directory=tmp_path),
        journal=TradingJournal(path=tmp_path / "j.jsonl",
                               summary_path=tmp_path / "s.jsonl"),
        market_feed_enabled=False,
    )
    response = TestClient(create_app(bridge)).get("/diagnose")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")

    body = response.text
    # The build stamp is on it: a diagnostic that does not say which build
    # produced it is how a fixed bug gets reported twice.
    assert "GODALGO diagnostics" in body
    for step in ("environment", "dns"):
        assert step in body
    assert body.strip().endswith(".")


def test_the_diagnose_page_accepts_a_venue(tmp_path):
    from fastapi.testclient import TestClient

    from godalgo.ui.credentials import CredentialStore
    from godalgo.ui.journal import TradingJournal
    from godalgo.ui.server import UIBridge, create_app

    bridge = UIBridge(
        credentials=CredentialStore(directory=tmp_path),
        journal=TradingJournal(path=tmp_path / "j.jsonl",
                               summary_path=tmp_path / "s.jsonl"),
        market_feed_enabled=False,
    )
    body = TestClient(create_app(bridge)).get("/diagnose?exchange=Kraken").text
    assert "venue: kraken" in body
