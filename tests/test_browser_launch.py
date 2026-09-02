"""The browser must open when the terminal is ready, not a second after launch.

Reported: the packaged executable "loaded after 5 refreshes", where earlier
builds opened first time. The cause was a fixed one-second timer -- the browser
was opened 1.0s after start whether or not uvicorn had bound yet. One second is
a guess about how long a cold start takes, and on the machine that matters it
is wrong: a 125MB unsigned onedir build has Windows Defender reading every
library before the server comes up. The tab then landed on a dead port, showed
a connection error, and only worked once a refresh happened to coincide with
the server being ready.

These tests pin the property that replaces the guess: when the browser is
opened, the terminal answers.
"""

from __future__ import annotations

import asyncio
import threading
import time
import types
import webbrowser

import pytest
import uvicorn

from godalgo.ui.credentials import CredentialStore
from godalgo.ui.journal import TradingJournal
from godalgo.ui.server import (
    UIBridge,
    create_app,
    find_free_port,
    port_owner,
    run_server,
    wait_until_serving,
)


@pytest.fixture
def bridge(tmp_path):
    return UIBridge(
        credentials=CredentialStore(directory=tmp_path),
        journal=TradingJournal(path=tmp_path / "j.jsonl",
                               summary_path=tmp_path / "s.jsonl"),
        market_feed_enabled=False,
    )


def test_the_browser_opens_only_once_the_terminal_answers(bridge, monkeypatch):
    """The regression itself, end to end against a real server."""
    port = find_free_port("127.0.0.1", 8940)
    seen: list[str] = []
    monkeypatch.setattr(
        webbrowser, "open", lambda url: seen.append(port_owner("127.0.0.1", port))
    )

    threading.Thread(
        target=lambda: run_server(bridge, port=port, open_browser=True),
        daemon=True,
    ).start()

    for _ in range(300):
        if seen:
            break
        time.sleep(0.05)

    # Not "the browser was opened" -- that was true before too. What was false
    # is that the thing it was pointed at existed yet.
    assert seen == ["godalgo"]


def test_waiting_actually_waits_for_a_slow_start(bridge):
    """A server that takes its time is waited for, not given up on.

    Deliberately started late, because a probe that returns true immediately
    would pass a test that merely checks the return value.
    """
    port = find_free_port("127.0.0.1", 8960)
    delay = 1.5

    def _late_server() -> None:
        time.sleep(delay)
        config = uvicorn.Config(create_app(bridge), host="127.0.0.1",
                                port=port, log_level="error")
        uvicorn.Server(config).run()

    threading.Thread(target=_late_server, daemon=True).start()

    started = time.monotonic()
    assert wait_until_serving("127.0.0.1", port, timeout=20.0) is True
    assert time.monotonic() - started >= delay


def test_waiting_gives_up_rather_than_hanging_forever():
    """Nothing is coming: return, so the caller can say so."""
    port = find_free_port("127.0.0.1", 8980)
    started = time.monotonic()
    assert wait_until_serving("127.0.0.1", port, timeout=1.0) is False
    elapsed = time.monotonic() - started
    assert 0.9 <= elapsed < 6.0


def test_a_bound_socket_alone_is_not_ready(bridge):
    """The gap this closes: accepting connections is not serving pages.

    A plain listening socket answers a TCP connect and nothing else, which is
    what a bare socket probe would have called success -- and pointing a
    browser at it produces exactly the error being fixed.
    """
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        assert wait_until_serving("127.0.0.1", port, timeout=1.0) is False


def test_a_slow_start_says_so_and_keeps_waiting(bridge, monkeypatch, capsys):
    """A slow start must be visibly slow, not apparently hung -- and must not
    give up.

    An earlier version stopped after 90 seconds and printed a note, which left
    the operator with a console message and no terminal. That is worse than the
    bug it replaced: a premature tab could at least be refreshed into working.
    """
    port = find_free_port("127.0.0.1", 8990)
    monkeypatch.setattr("godalgo.ui.server._READY_SLICE", 5.0)

    answers = iter([False, False, False, False, False, False, True])
    monkeypatch.setattr(
        "godalgo.ui.server.wait_until_serving",
        lambda *a, **k: next(answers, True),
    )
    opened: list[str] = []
    monkeypatch.setattr(webbrowser, "open", lambda url: opened.append(url))

    stop = threading.Event()
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: stop.wait(10.0))
    thread = threading.Thread(
        target=lambda: run_server(bridge, port=port, open_browser=True),
        daemon=True,
    )
    thread.start()

    said = ""
    for _ in range(200):
        said += capsys.readouterr().out
        if opened:
            break
        time.sleep(0.02)
    stop.set()
    thread.join(timeout=5.0)

    assert opened == [f"http://127.0.0.1:{port}"]   # it waited, then opened
    assert "still starting after 10s" in said       # and said so on the way
    assert f"127.0.0.1:{port}" in said              # naming where to look


def test_the_page_is_served_before_the_trading_session_is_ready(tmp_path):
    """The defect that made a launch look like a hang.

    Uvicorn serves nothing until lifespan startup returns, and starting a
    session seeds history from the venue -- a synchronous, paginated, retrying
    network call. Awaiting that in the lifespan meant the terminal could not
    serve its own page until the exchange answered, so a slow venue presented
    as an application that would not start.

    The UI exists to show what the bot is doing. It must be up before the bot
    is ready, or the moment you most need to see what is happening is the
    moment there is nothing on screen.
    """
    from godalgo.ui.server import create_app

    started = threading.Event()

    class _SlowSession:
        """Stands in for seeding against an unresponsive venue."""

        async def start(self, broker):
            await asyncio.sleep(30.0)
            started.set()

        async def stop(self): ...

    bridge = UIBridge(
        credentials=CredentialStore(directory=tmp_path),
        journal=TradingJournal(path=tmp_path / "j.jsonl",
                               summary_path=tmp_path / "s.jsonl"),
        market_feed_enabled=False,
    )
    bridge.session = _SlowSession()
    bridge.mode_controller = types.SimpleNamespace(broker=object())

    port = find_free_port("127.0.0.1", 9010)
    config = uvicorn.Config(create_app(bridge), host="127.0.0.1", port=port,
                            log_level="error")
    server = uvicorn.Server(config)
    threading.Thread(target=server.run, daemon=True).start()
    try:
        # Well inside the 30s the session takes, and the page must already
        # answer. Before the fix this timed out.
        assert wait_until_serving("127.0.0.1", port, timeout=10.0) is True
        assert not started.is_set()   # the session is genuinely still starting
    finally:
        server.should_exit = True


def test_a_session_that_fails_to_start_is_reported_not_swallowed(tmp_path):
    """Backgrounding the start must not turn a failure into silence."""
    from godalgo.ui.server import create_app

    class _BrokenSession:
        async def start(self, broker):
            raise RuntimeError("venue unreachable")

        async def stop(self): ...

    bridge = UIBridge(
        credentials=CredentialStore(directory=tmp_path),
        journal=TradingJournal(path=tmp_path / "j.jsonl",
                               summary_path=tmp_path / "s.jsonl"),
        market_feed_enabled=False,
    )
    bridge.session = _BrokenSession()
    bridge.mode_controller = types.SimpleNamespace(broker=object())

    port = find_free_port("127.0.0.1", 9030)
    config = uvicorn.Config(create_app(bridge), host="127.0.0.1", port=port,
                            log_level="error")
    server = uvicorn.Server(config)
    threading.Thread(target=server.run, daemon=True).start()
    try:
        assert wait_until_serving("127.0.0.1", port, timeout=10.0) is True
        for _ in range(100):
            text = " ".join(
                f"{e['message']} {e.get('detail', '')}"
                for e in bridge.events.entries()
            )
            if "could not start the trading session" in text:
                break
            time.sleep(0.05)
        assert "could not start the trading session" in text
        assert "venue unreachable" in text
    finally:
        server.should_exit = True
