"""A second launch must not fail silently.

This cost a full debugging round. Starting a second copy died with

    [Errno 10048] only one usage of each socket address (protocol/network
    address/port) is normally permitted

which is the operating system's phrasing for "something already has this
port". The window closed, the *older* copy kept serving, and the browser showed
a terminal that looked healthy while being several builds behind. A bug that
had already been fixed appeared not to be fixed, and nothing on screen could
have revealed why.

Two defences: a second launch opens the running one instead of dying, and the
build stamp is visible so "which version am I looking at" is answerable.
"""

from __future__ import annotations

import socket
import threading

import pytest
import uvicorn

from godalgo.build_info import build_stamp, describe
from godalgo.ui.credentials import CredentialStore
from godalgo.ui.journal import TradingJournal
from godalgo.ui.server import (
    UIBridge,
    create_app,
    find_free_port,
    port_owner,
    run_server,
)


@pytest.fixture
def bridge(tmp_path):
    return UIBridge(
        credentials=CredentialStore(directory=tmp_path),
        journal=TradingJournal(path=tmp_path / "j.jsonl",
                               summary_path=tmp_path / "s.jsonl"),
        market_feed_enabled=False,
    )


@pytest.fixture
def running_terminal(bridge):
    """A real terminal on a real port, so the probe is tested against the
    thing it exists to recognise rather than a mock of it."""
    port = find_free_port("127.0.0.1", 8900)
    config = uvicorn.Config(create_app(bridge), host="127.0.0.1", port=port,
                            log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    import time
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    else:
        pytest.fail("terminal did not start")

    yield port

    server.should_exit = True
    thread.join(timeout=5)


# --- identifying who holds the port ----------------------------------------

def test_a_free_port_is_reported_free():
    port = find_free_port("127.0.0.1", 8950)
    assert port_owner("127.0.0.1", port) == "free"


def test_our_own_terminal_is_recognised(running_terminal):
    """Recognised by asking, not guessing: only our terminal answers
    /api/state with a snapshot."""
    assert port_owner("127.0.0.1", running_terminal) == "godalgo"


def test_an_unrelated_listener_is_not_mistaken_for_us():
    """The distinction that matters. Treating any listener as "already
    running" would silently refuse to start because something unrelated
    happened to hold the port."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    port = sock.getsockname()[1]
    try:
        assert port_owner("127.0.0.1", port) == "other"
    finally:
        sock.close()


# --- what a second launch does ---------------------------------------------

def test_a_second_launch_opens_the_running_one_instead_of_dying(
    bridge, running_terminal, monkeypatch, capsys
):
    """The whole point. It must not raise, and must not start a rival copy:
    two on one account would size against the same buying power."""
    opened: list[str] = []
    monkeypatch.setattr("webbrowser.open", opened.append)

    run_server(bridge, port=running_terminal, open_browser=True)

    out = capsys.readouterr().out
    assert "already running" in out
    assert opened == [f"http://127.0.0.1:{running_terminal}"]


def test_the_message_says_how_to_stop_a_background_copy(
    bridge, running_terminal, capsys
):
    """The likeliest owner is the scheduled task from `service install`, and
    a message that does not mention it leaves someone hunting."""
    run_server(bridge, port=running_terminal, open_browser=False)
    assert "service stop" in capsys.readouterr().out


def test_an_unrelated_listener_moves_the_terminal_to_a_free_port(
    bridge, monkeypatch, capsys
):
    """Refusing to start because some other program holds 8787 would be a
    worse outcome than moving."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    port = sock.getsockname()[1]

    served: dict[str, int] = {}

    def fake_run(app, host, port, **kwargs):
        served["port"] = port

    monkeypatch.setattr("uvicorn.run", fake_run)
    try:
        run_server(bridge, port=port, open_browser=False)
    finally:
        sock.close()

    assert served["port"] != port
    assert "in use by another program" in capsys.readouterr().out


def test_a_race_after_the_probe_is_reported_in_words(bridge, monkeypatch, capsys):
    """The check is a probe, so something can take the port between asking
    and binding. That must not surface as a raw errno either."""
    def boom(app, host, port, **kwargs):
        raise OSError(10048, "only one usage of each socket address")

    monkeypatch.setattr("uvicorn.run", boom)
    run_server(bridge, port=find_free_port("127.0.0.1", 8970), open_browser=False)

    err = capsys.readouterr().err
    assert "Could not start" in err
    assert "--port" in err


# --- the build stamp --------------------------------------------------------

def test_the_build_is_identifiable():
    stamp = build_stamp()
    assert stamp["version"]
    assert stamp["commit"]
    assert stamp["source"] in {"packaged", "checkout"}


def test_describe_is_short_enough_for_a_header():
    assert len(describe()) < 60


def test_the_build_reaches_the_snapshot(bridge):
    """Visible in the interface, so "am I running the new one" is a question
    that can be answered rather than assumed."""
    assert bridge.snapshot().to_dict()["build"] == describe()


def test_the_stamp_never_raises_without_git_or_a_packaged_stamp(monkeypatch):
    """A version banner must never be what stops the program starting."""
    import godalgo.build_info as info

    def no_git(*args, **kwargs):
        raise OSError("git not found")

    monkeypatch.setattr(info.subprocess, "run", no_git)
    info.build_stamp.cache_clear()
    try:
        assert info.build_stamp()["commit"]
    finally:
        info.build_stamp.cache_clear()
