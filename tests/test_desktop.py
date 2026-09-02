"""The terminal as a window rather than a browser tab.

Everything here that can be checked on a headless Linux runner is checked
here: the port choice, the single-instance lock, the readiness wait, the
close warning, and the launcher's routing between window and browser. The
window itself cannot be -- opening one needs a desktop -- so the boundary is
drawn deliberately: this file proves that everything up to
``webview.create_window`` is right, and the packaging check in CI proves the
component is present in the binary. The window appearing is confirmed on
Windows and nowhere else.
"""

from __future__ import annotations

import importlib.util
import json
import socket
import sys
import threading
import types
from pathlib import Path

import pytest

from godalgo.ui import desktop
from godalgo.ui.server import UIBridge


def _launcher():
    """``run-terminal.py`` as a module. The hyphen makes it unimportable."""
    path = Path(__file__).resolve().parents[1] / "run-terminal.py"
    spec = importlib.util.spec_from_file_location("godalgo_launcher", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------
# port selection
# --------------------------------------------------------------------------

def test_free_port_is_actually_bindable():
    """The whole point: never fail with [Errno 10048] again."""
    port = desktop.free_port()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", port))  # raises if the OS lied


def test_free_port_is_not_the_old_fixed_one():
    """8787 is what a second launch used to collide on."""
    ports = {desktop.free_port() for _ in range(8)}
    assert 8787 not in ports
    # Ephemeral means the OS chooses, so they should not all be one number.
    assert len(ports) > 1


# --------------------------------------------------------------------------
# single instance
# --------------------------------------------------------------------------

def test_lock_round_trips_a_port(tmp_path):
    lock = desktop.SingleInstance(tmp_path / "terminal.lock")
    assert lock.read() is None
    lock.claim(9123)
    assert lock.read() == 9123
    lock.release()
    assert lock.read() is None


def test_lock_records_the_pid_so_a_human_can_kill_it(tmp_path):
    import os

    lock = desktop.SingleInstance(tmp_path / "terminal.lock")
    lock.claim(9123)
    assert json.loads((tmp_path / "terminal.lock").read_text())["pid"] == os.getpid()


def test_a_corrupt_lock_is_not_fatal(tmp_path):
    """A half-written file after a hard kill must not lock the operator out."""
    path = tmp_path / "terminal.lock"
    path.write_text('{"port": ')
    assert desktop.SingleInstance(path).read() is None
    assert desktop.SingleInstance(path).live_port() is None


def test_a_stale_lock_does_not_block_a_launch(tmp_path):
    """The exact failure this replaces: a dead copy holding the door shut.

    The lock names a port; if nothing is answering there, the previous copy is
    gone and the file is a leftover, not a running program.
    """
    lock = desktop.SingleInstance(tmp_path / "terminal.lock")
    lock.claim(desktop.free_port())  # free, so nothing is listening
    assert lock.live_port() is None


def test_a_live_lock_reports_the_running_port(tmp_path, monkeypatch):
    from godalgo.ui import server

    monkeypatch.setattr(server, "port_owner", lambda host, port: "godalgo")
    lock = desktop.SingleInstance(tmp_path / "terminal.lock")
    lock.claim(9123)
    assert lock.live_port() == 9123


def test_an_unrelated_program_on_the_port_is_not_us(tmp_path, monkeypatch):
    from godalgo.ui import server

    monkeypatch.setattr(server, "port_owner", lambda host, port: "other")
    lock = desktop.SingleInstance(tmp_path / "terminal.lock")
    lock.claim(9123)
    assert lock.live_port() is None


def test_releasing_a_missing_lock_is_silent(tmp_path):
    desktop.SingleInstance(tmp_path / "gone.lock").release()


class _FakeWindow:
    def __init__(self):
        self.events = types.SimpleNamespace(closing=_Signal())
        self.url = None


class _Signal:
    def __init__(self):
        self.handlers = []

    def __iadd__(self, handler):
        self.handlers.append(handler)
        return self


class _FakeWebview(types.ModuleType):
    """Stands in for pywebview: records the window, never opens one."""

    def __init__(self):
        super().__init__("webview")
        self.opened: list[tuple[str, str]] = []
        self.started = 0
        self.windows: list[_FakeWindow] = []

    def create_window(self, title, url, **kwargs):
        self.opened.append((title, url))
        window = _FakeWindow()
        window.url = url
        self.windows.append(window)
        return window

    def start(self):
        self.started += 1


@pytest.fixture
def fake_webview(monkeypatch):
    module = _FakeWebview()
    monkeypatch.setitem(sys.modules, "webview", module)
    return module


def test_a_second_launch_attaches_instead_of_starting_a_second_bot(
    tmp_path, monkeypatch, fake_webview,
):
    """Where [Errno 10048] used to be, there is now a window on the live one.

    Two engines on one account is the failure that matters here -- each would
    size against the same buying power while blind to the other's position --
    so the second launch must serve nothing and view what is already running.
    """
    monkeypatch.setattr(desktop, "data_dir", lambda: tmp_path)
    monkeypatch.setattr(desktop.SingleInstance, "live_port", lambda self: 9123)

    started: list[int] = []
    monkeypatch.setattr(
        desktop, "_ServerThread", lambda bridge, port: started.append(port),
    )

    assert desktop.DesktopApp(UIBridge()).run() == 0
    assert started == []                                  # no second server
    assert fake_webview.opened[0][1] == "http://127.0.0.1:9123"
    assert fake_webview.started == 1


def test_a_first_launch_serves_its_own_ephemeral_port(
    tmp_path, monkeypatch, fake_webview,
):
    monkeypatch.setattr(desktop, "data_dir", lambda: tmp_path)
    monkeypatch.setattr(desktop.SingleInstance, "live_port", lambda self: None)

    from godalgo.ui import server

    monkeypatch.setattr(server, "port_owner", lambda host, port: "free")

    class _Stub:
        def __init__(self, bridge, port):
            self.port = port

        def start(self): ...
        def wait_until_ready(self, timeout): return True

    monkeypatch.setattr(desktop, "_ServerThread", _Stub)

    assert desktop.DesktopApp(UIBridge()).run() == 0
    url = fake_webview.opened[0][1]
    assert url.startswith("http://127.0.0.1:")
    assert not url.endswith(":8787")     # ephemeral, never the fixed port


def test_attaching_does_not_delete_the_running_terminals_lock(
    tmp_path, monkeypatch, fake_webview,
):
    """Closing a viewer window must not make the real one look gone."""
    monkeypatch.setattr(desktop, "data_dir", lambda: tmp_path)
    lock = desktop.SingleInstance(tmp_path / "terminal.lock")
    lock.claim(9123)
    monkeypatch.setattr(desktop.SingleInstance, "live_port", lambda self: 9123)

    assert desktop.DesktopApp(UIBridge()).run() == 0
    assert lock.read() == 9123


def test_the_background_service_is_found_without_a_lock(tmp_path, monkeypatch):
    """The scheduled task serves the fixed port and writes no lock file."""
    from godalgo.ui import server

    monkeypatch.setattr(desktop, "data_dir", lambda: tmp_path)
    monkeypatch.setattr(
        server, "port_owner",
        lambda host, port: "godalgo" if port == desktop.SERVICE_PORT else "free",
    )
    app = desktop.DesktopApp(UIBridge())
    lock = desktop.SingleInstance(tmp_path / "terminal.lock")
    assert app._running_port(lock) == desktop.SERVICE_PORT


def test_an_unrelated_program_on_the_service_port_is_not_attached_to(
    tmp_path, monkeypatch,
):
    from godalgo.ui import server

    monkeypatch.setattr(desktop, "data_dir", lambda: tmp_path)
    monkeypatch.setattr(server, "port_owner", lambda host, port: "other")
    app = desktop.DesktopApp(UIBridge())
    assert app._running_port(desktop.SingleInstance(tmp_path / "x.lock")) is None


# --------------------------------------------------------------------------
# waiting for the server
# --------------------------------------------------------------------------

def test_ready_when_the_terminal_answers(tmp_path):
    """The window must not paint before the server can serve it."""
    import uvicorn

    from godalgo.ui.credentials import CredentialStore
    from godalgo.ui.journal import TradingJournal
    from godalgo.ui.server import create_app, find_free_port

    bridge = UIBridge(
        credentials=CredentialStore(directory=tmp_path),
        journal=TradingJournal(path=tmp_path / "j.jsonl",
                               summary_path=tmp_path / "s.jsonl"),
        market_feed_enabled=False,
    )
    port = find_free_port("127.0.0.1", 8920)
    config = uvicorn.Config(create_app(bridge), host="127.0.0.1", port=port,
                            log_level="error")
    server = uvicorn.Server(config)
    threading.Thread(target=server.run, daemon=True).start()
    try:
        assert desktop._ServerThread(bridge, port).wait_until_ready(10.0) is True
    finally:
        server.should_exit = True


def test_a_bare_listening_socket_is_not_ready():
    """A socket that accepts and serves nothing is what the browser used to
    be pointed at. Accepting a connection is not being ready."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        thread = desktop._ServerThread(UIBridge(), port)
        assert thread.wait_until_ready(timeout=1.0) is False


def test_not_ready_when_nothing_is_listening():
    thread = desktop._ServerThread(UIBridge(), desktop.free_port())
    assert thread.wait_until_ready(timeout=0.5) is False


def test_a_crashed_server_stops_the_wait_immediately():
    """A dead server must not be waited on for the full timeout."""
    thread = desktop._ServerThread(UIBridge(), desktop.free_port())
    thread.error = RuntimeError("bind failed")
    assert thread.wait_until_ready(timeout=30.0) is False


def test_a_server_that_never_comes_up_reports_where_the_log_is(tmp_path, monkeypatch):
    monkeypatch.setattr(desktop, "data_dir", lambda: tmp_path)
    monkeypatch.setattr(desktop.SingleInstance, "live_port", lambda self: None)

    class _Stub:
        def __init__(self, bridge, port): ...
        def start(self): ...
        def wait_until_ready(self, timeout): return False

    monkeypatch.setattr(desktop, "_ServerThread", _Stub)
    said: list[tuple[str, str]] = []
    monkeypatch.setattr(desktop, "_message", lambda t, b: said.append((t, b)))

    assert desktop.DesktopApp(UIBridge()).run() == 1
    assert "terminal.log" in said[0][1]
    # And the lock must not be left behind claiming a port nothing serves.
    assert desktop.SingleInstance(tmp_path / "terminal.lock").read() is None


def test_the_server_thread_is_a_daemon():
    """Closing the window must end the process, not leave a bot running."""
    thread = desktop._ServerThread(UIBridge(), 9123)
    assert thread.daemon is True
    assert isinstance(thread, threading.Thread)


# --------------------------------------------------------------------------
# closing with money in the market
# --------------------------------------------------------------------------

def _bridge_with(mode: str, positions: int) -> UIBridge:
    bridge = UIBridge()
    bridge.mode = mode
    bridge.tracker.open_positions.update(
        {f"p{i}": object() for i in range(positions)}
    )
    return bridge


def test_no_warning_in_dry_run():
    assert desktop.close_warning(_bridge_with("dry_run", 3)) is None


def test_no_warning_live_but_flat():
    assert desktop.close_warning(_bridge_with("live", 0)) is None


def test_warns_when_live_with_a_position():
    warning = desktop.close_warning(_bridge_with("live", 1))
    assert warning is not None
    assert "1 open position" in warning
    assert "It stays open" in warning   # singular reads correctly


def test_warning_counts_positions():
    warning = desktop.close_warning(_bridge_with("live", 4))
    assert "4 open positions" in warning
    assert "They stay open" in warning


def test_close_warning_survives_a_bridge_without_a_tracker():
    class _Bare:
        mode = "live"

    assert desktop.close_warning(_Bare()) is None


# --------------------------------------------------------------------------
# failing loudly
# --------------------------------------------------------------------------

def test_a_missing_window_component_is_reported_not_silent(monkeypatch, tmp_path):
    monkeypatch.setattr(desktop, "data_dir", lambda: tmp_path)
    monkeypatch.setattr(desktop, "available", lambda: False)
    said: list[tuple[str, str]] = []
    monkeypatch.setattr(desktop, "_message", lambda t, b: said.append((t, b)))

    assert desktop.run_desktop(UIBridge()) == 1
    assert "--browser" in said[0][1]     # names the way out


def test_an_unexpected_failure_names_the_log(monkeypatch, tmp_path):
    monkeypatch.setattr(desktop, "data_dir", lambda: tmp_path)
    monkeypatch.setattr(desktop, "available", lambda: True)

    def _boom(self):
        raise RuntimeError("no display")

    monkeypatch.setattr(desktop.DesktopApp, "run", _boom)
    said: list[tuple[str, str]] = []
    monkeypatch.setattr(desktop, "_message", lambda t, b: said.append((t, b)))

    assert desktop.run_desktop(UIBridge()) == 1
    assert "no display" in said[0][1]
    assert "terminal.log" in said[0][1]


def test_message_falls_back_to_stderr_off_windows(capsys, monkeypatch):
    """Note the monkeypatch: without it this test opened a real message box.

    On a Windows CI runner that dialog has nobody to click it, so MessageBoxW
    never returns and the job hangs with no failing test to point at. That is
    what the per-test timeout in CI now catches.
    """
    monkeypatch.setattr(sys, "platform", "linux")
    desktop._message("title", "body")
    assert "body" in capsys.readouterr().err


def test_message_uses_a_dialog_on_windows(monkeypatch):
    """The dialog is the point: a windowed process has no console to print to."""
    import ctypes
    import types as _types

    shown: list[tuple] = []
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(
        ctypes, "windll",
        _types.SimpleNamespace(
            user32=_types.SimpleNamespace(
                MessageBoxW=lambda *a: shown.append(a) or 1
            ),
        ),
        raising=False,
    )
    desktop._message("title", "body")
    assert shown and shown[0][1] == "body"


def test_a_broken_dialog_falls_back_instead_of_raising(monkeypatch, capsys):
    """A failure to report a failure must not become the failure."""
    import ctypes
    import types as _types

    def _boom(*_):
        raise OSError("no window station")

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(
        ctypes, "windll",
        _types.SimpleNamespace(user32=_types.SimpleNamespace(MessageBoxW=_boom)),
        raising=False,
    )
    desktop._message("title", "body")
    assert "body" in capsys.readouterr().err


def test_the_console_is_never_hidden_off_windows(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    assert desktop._owns_console() is False
    desktop._hide_console()   # must not raise


def test_data_dir_exists_after_the_call(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "AppData"))
    path = desktop.data_dir()
    assert path.is_dir()
    assert path.name == "GODALGO"


# --------------------------------------------------------------------------
# what the launcher chooses
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("argv", "expect_window"),
    [
        ([], False),                      # a double-click gets a browser tab
        (["--demo"], False),
        (["--browser"], False),           # the default, named explicitly
        (["--no-browser"], False),        # the background service
        (["--port", "8787"], False),
        (["--window"], True),             # the window is opt-in
        (["--window", "--demo"], True),
    ],
)
def test_launcher_routing(monkeypatch, argv, expect_window):
    launcher = _launcher()
    monkeypatch.setattr(sys, "argv", ["run-terminal.py", *argv])
    monkeypatch.setattr(desktop, "available", lambda: True)

    windowed: list[object] = []
    served: list[int] = []
    monkeypatch.setattr(desktop, "run_desktop", lambda bridge: windowed.append(bridge) or 0)

    from godalgo.ui import server

    monkeypatch.setattr(
        server, "run_server",
        lambda bridge, port=8787, open_browser=True: served.append(port),
    )

    assert launcher.main() == 0
    assert bool(windowed) is expect_window
    assert bool(served) is not expect_window


def test_launcher_falls_back_when_there_is_no_window_component(monkeypatch, capsys):
    """Asking for a window on a build without one gets a tab, and is told so
    rather than looking like the window failed to open."""
    launcher = _launcher()
    monkeypatch.setattr(sys, "argv", ["run-terminal.py", "--window"])
    monkeypatch.setattr(desktop, "available", lambda: False)

    from godalgo.ui import server

    served: list[int] = []
    monkeypatch.setattr(
        server, "run_server",
        lambda bridge, port=8787, open_browser=True: served.append(port),
    )

    assert launcher.main() == 0
    assert served == [8787]
    assert "browser tab" in capsys.readouterr().out


def test_check_window_flag_reports_availability(monkeypatch, capsys):
    """CI runs this against the built binary; it must be honest either way."""
    launcher = _launcher()
    monkeypatch.setattr(sys, "argv", ["run-terminal.py", "--check-window"])

    monkeypatch.setattr(desktop, "available", lambda: True)
    assert launcher.main() == 0
    assert "available" in capsys.readouterr().out

    monkeypatch.setattr(desktop, "available", lambda: False)
    assert launcher.main() == 1
    assert "missing" in capsys.readouterr().err


def test_available_never_raises():
    assert isinstance(desktop.available(), bool)
