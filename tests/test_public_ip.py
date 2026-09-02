"""Showing the address the venue sees.

Binance's -2015 says a key was used from an address that is not on its
allow-list. It does not say what the address *is*, and on a domestic connection
that is a moving target: ISPs reassign dynamic addresses on a reboot or their
own schedule, so a key whitelisted correctly stops working days later with
nothing having changed. Looking it up in a browser to compare was the errand
this replaces.

The lookup goes to third-party echo services, so the tests run against a real
local HTTP server rather than a mock of one — what matters is the behaviour
when a provider is down, rate-limited, or answering with something that is not
an address at all, and a mock would only assert my own assumptions about that.
"""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from godalgo.ui import network
from godalgo.ui.network import PublicAddress, public_ip


@pytest.fixture(autouse=True)
def _no_cache(monkeypatch):
    """Each test starts cold; the cache is tested explicitly where it matters."""
    monkeypatch.setattr(network, "_cache", None)


def _serve(handler_body, status=200):
    """A one-endpoint HTTP server returning what the test asks for."""

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler's interface
            payload = handler_body().encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_):
            pass

    server = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_port}"


def test_reports_the_address_and_where_it_came_from(monkeypatch):
    server, url = _serve(lambda: "203.0.113.7")
    try:
        monkeypatch.setattr(network, "_PROVIDERS", (url,))
        found = public_ip()
        assert found.ip == "203.0.113.7"
        assert found.source is not None      # so a wrong answer is traceable
        assert found.error is None
    finally:
        server.shutdown()


def test_whitespace_is_stripped(monkeypatch):
    """These services commonly answer with a trailing newline, and an address
    with a newline in it does not match anything pasted into Binance."""
    server, url = _serve(lambda: "203.0.113.7\n")
    try:
        monkeypatch.setattr(network, "_PROVIDERS", (url,))
        assert public_ip().ip == "203.0.113.7"
    finally:
        server.shutdown()


def test_falls_through_to_the_next_provider(monkeypatch):
    """One free service being down must be a delay, not a failure."""
    good, good_url = _serve(lambda: "198.51.100.4")
    bad, bad_url = _serve(lambda: "unavailable", status=503)
    try:
        monkeypatch.setattr(network, "_PROVIDERS", (bad_url, good_url))
        assert public_ip().ip == "198.51.100.4"
    finally:
        good.shutdown()
        bad.shutdown()


def test_a_block_page_is_not_reported_as_an_address(monkeypatch):
    """The failure mode that would be worse than reporting nothing.

    A captive portal or corporate proxy answers 200 with HTML. Displaying that
    as the address gives the operator something to paste into Binance that
    cannot possibly work, and no reason to doubt it.
    """
    server, url = _serve(lambda: "<html>Access denied</html>")
    try:
        monkeypatch.setattr(network, "_PROVIDERS", (url,))
        found = public_ip()
        assert found.ip is None
        assert found.error is not None
    finally:
        server.shutdown()


def test_an_ipv6_answer_is_rejected(monkeypatch):
    """Binance's allow-list takes IPv4. A v6 address cannot be entered there,
    so reporting one looks like an answer and is not."""
    server, url = _serve(lambda: "2001:db8::1")
    try:
        monkeypatch.setattr(network, "_PROVIDERS", (url,))
        assert public_ip().ip is None
    finally:
        server.shutdown()


def test_every_provider_failing_names_the_manual_route(monkeypatch):
    monkeypatch.setattr(network, "_PROVIDERS", ("http://127.0.0.1:1/",))
    found = public_ip()
    assert found.ip is None
    assert "ifconfig.me" in found.error    # a way to get it without the app


def test_a_failure_is_not_cached(monkeypatch):
    """Caching a blip would keep the panel empty for five minutes after the
    connection came back."""
    monkeypatch.setattr(network, "_PROVIDERS", ("http://127.0.0.1:1/",))
    assert public_ip().ip is None
    assert network._cache is None


def test_the_answer_is_cached_but_recheck_ignores_it(monkeypatch):
    """Opening the panel repeatedly must not hammer a free service — but the
    reason to press recheck is a suspicion the address moved."""
    calls: list[int] = []

    def _body():
        calls.append(1)
        return "203.0.113.9"

    server, url = _serve(_body)
    try:
        monkeypatch.setattr(network, "_PROVIDERS", (url,))
        public_ip()
        public_ip()
        assert len(calls) == 1              # second call served from cache
        public_ip(force=True)
        assert len(calls) == 2              # recheck really rechecks
    finally:
        server.shutdown()


def test_lookup_never_raises(monkeypatch):
    """A convenience readout must not be able to break a request handler."""

    def _explode(*_a, **_k):
        raise RuntimeError("socket layer gone")

    monkeypatch.setattr(network, "urlopen", _explode)
    assert isinstance(public_ip(), PublicAddress)


def test_the_endpoint_reports_it(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from godalgo.ui.credentials import CredentialStore
    from godalgo.ui.journal import TradingJournal
    from godalgo.ui.server import UIBridge, create_app

    monkeypatch.setattr(
        network, "public_ip",
        lambda *, force=False: PublicAddress(ip="203.0.113.5", source="test"),
    )
    bridge = UIBridge(
        credentials=CredentialStore(directory=tmp_path),
        journal=TradingJournal(path=tmp_path / "j.jsonl",
                               summary_path=tmp_path / "s.jsonl"),
        market_feed_enabled=False,
    )
    with TestClient(create_app(bridge)) as client:
        body = client.get("/api/ip").json()
        assert body["ip"] == "203.0.113.5"

        # A recheck is worth a line in the activity log: it is the number the
        # operator is about to paste into an exchange.
        client.get("/api/ip?refresh=true")
        assert any(
            "203.0.113.5" in e["message"] for e in bridge.events.entries()
        )


def test_the_ip_error_points_at_the_panel():
    """-2015 must send the operator somewhere, not just name the problem."""
    from godalgo.ui.venue import _BINANCE_CODES

    kind, explanation = _BINANCE_CODES["-2015"]
    assert kind == "ip_not_allowed"
    assert "Connections" in explanation      # where the address is shown
    assert "dynamic" in explanation          # why a working key stopped working
