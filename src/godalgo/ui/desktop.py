"""The terminal as a Windows application, not a browser tab.

The interface is unchanged -- the same HTML, CSS and canvas -- but it renders
inside a native window through the WebView2 runtime that ships with Edge on
every current Windows install. Nothing about the trading code changes.

Why this exists is worth recording, because none of the problems it solves were
trading problems:

* **A fixed port meant collisions.** Port 8787 was hardcoded, so a second copy
  -- or the scheduled task started at sign-in -- held it and the next launch
  died with ``[Errno 10048]``. The window vanished, the *older* copy kept
  serving, and the browser showed a terminal that looked healthy while being
  builds out of date. Here the port is ephemeral: the OS picks a free one, it
  is never typed and never seen, and there is nothing to collide with.
* **A browser tab is not an application.** There was no window to close, no
  taskbar entry, nothing in alt-tab, and closing the tab left the bot running
  invisibly with a live position.
* **Failures were silent.** A window that fails to open shows nothing at all,
  which after this project's history is unacceptable -- so startup is logged
  to a file and a fatal error raises a message box rather than disappearing.

Only one copy runs at a time, enforced by a lock file naming the live port. Two
terminals on one account would each size against the same buying power while
blind to the other's position, which is the same reason the scheduled task sets
``MultipleInstancesPolicy=IgnoreNew``.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "DesktopApp", "SingleInstance", "available", "close_warning",
    "data_dir", "free_port", "run_desktop",
]

logger = logging.getLogger(__name__)

_LOCK_NAME = "terminal.lock"

SERVICE_PORT = 8787
"""Where the background scheduled task serves, if it is installed.

Checked so a double-click attaches to the running bot instead of starting a
second one beside it. The window's own port is never this: it is whatever the
OS hands out.
"""


def data_dir() -> Path:
    """Where the lock and the log live, created if missing."""
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_STATE_HOME")
    root = Path(base) / "GODALGO" if base else Path.home() / ".godalgo"
    root.mkdir(parents=True, exist_ok=True)
    return root


def available() -> bool:
    """Whether a native window can be opened here.

    Asks the import system rather than importing: this runs on every launch
    and on a build check, and neither needs pywebview's module-level work
    done, let alone its GUI backend loaded.
    """
    import importlib.util

    try:
        return importlib.util.find_spec("webview") is not None
    except (ImportError, ValueError):
        return False


def free_port() -> int:
    """A port the OS says is free.

    Bound to port 0 and read back, which is the only way to ask without
    guessing. A fixed port is what made a second launch fatal.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@dataclass
class SingleInstance:
    """A lock file naming the port of the running terminal.

    Not a mutex: the useful thing to hold is *where* the live copy is, so a
    second launch can say so instead of failing. A stale file -- left by a
    crash or a hard kill -- is detected by probing the port it names, so it
    cannot lock the operator out of their own program.
    """

    path: Path

    def read(self) -> int | None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        port = data.get("port")
        return int(port) if isinstance(port, int) else None

    def live_port(self) -> int | None:
        """The port of a terminal that is actually answering, if any."""
        from godalgo.ui.server import port_owner

        port = self.read()
        if port is None:
            return None
        return port if port_owner("127.0.0.1", port) == "godalgo" else None

    def claim(self, port: int) -> None:
        self.path.write_text(
            json.dumps({"pid": os.getpid(), "port": port}), encoding="utf-8"
        )

    def release(self) -> None:
        try:
            self.path.unlink(missing_ok=True)
        except OSError:
            logger.debug("could not remove the lock file", exc_info=True)


def _owns_console() -> bool:
    """Whether this process created the console it is attached to.

    A double-clicked executable gets a console window of its own; the same
    executable started from ``cmd`` shares the operator's. Hiding the first is
    right -- an application should not ship with a stray black rectangle behind
    it -- and hiding the second would take away the window they were typing in.

    ``GetConsoleProcessList`` tells the two apart: it reports how many
    processes are attached, and one means the console exists only for us.
    """
    if sys.platform != "win32":
        return False
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        if not kernel32.GetConsoleWindow():
            return False
        buffer = (ctypes.c_uint * 2)()
        return int(kernel32.GetConsoleProcessList(buffer, 2)) == 1
    except Exception:
        logger.debug("could not inspect the console", exc_info=True)
        return False


def _hide_console() -> None:
    """Hide our own console window, if we own it.

    Kept instead of building a second ``--windowed`` executable. A windowed
    PyInstaller binary has no stdout at all, which would silence
    ``service install``, ``--browser`` and every diagnostic printed from the
    command line -- a poor trade for one hidden window.
    """
    if not _owns_console():
        return
    try:
        import ctypes

        ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), 0)
    except Exception:
        logger.debug("could not hide the console", exc_info=True)


def close_warning(bridge: object) -> str | None:
    """What to say before the window closes, or ``None`` to just close.

    Closing the window ends the process, and ending the process ends the bot.
    In dry run that costs nothing. In live mode with money in the market it
    leaves real positions open on the exchange with nothing watching their
    stops -- so that one case asks first, and no other case nags.
    """
    if getattr(bridge, "mode", "dry_run") != "live":
        return None
    tracker = getattr(bridge, "tracker", None)
    open_positions = getattr(tracker, "open_positions", None) or {}
    if not open_positions:
        return None

    count = len(open_positions)
    subject = (
        "1 open position" if count == 1 else f"{count} open positions"
    )
    verb = "stays" if count == 1 else "stay"
    it = "It" if count == 1 else "They"
    return (
        f"LIVE mode with {subject}.\n\n"
        f"Closing this window stops the bot. {it} {verb} open on the "
        "exchange with nothing managing the stops.\n\nClose anyway?"
    )


@dataclass
class DesktopApp:
    """Serves the terminal and shows it in a native window.

    Args:
        bridge: The assembled terminal.
        title: Window title.
    """

    bridge: object
    title: str = "GODALGO"
    width: int = 1500
    height: int = 950

    def run(self) -> int:
        """Open the window. Returns a process exit code."""
        from godalgo.build_info import describe

        lock = SingleInstance(data_dir() / _LOCK_NAME)

        # A terminal is already running -- another window, or the scheduled
        # task started at sign-in. Show *that* one rather than starting a
        # second bot beside it. Two engines on one account would each size
        # against the same buying power while blind to the other's position,
        # and refusing outright would leave a double-click doing nothing
        # visible, which is how this became hard to debug in the first place.
        port = self._running_port(lock)
        owned = port is None
        if port is None:
            port = free_port()
            lock.claim(port)
            logger.info("serving the terminal on 127.0.0.1:%d", port)

            server = _ServerThread(self.bridge, port)
            server.start()
            if not server.wait_until_ready(timeout=30.0):
                lock.release()
                _message(
                    "GODALGO could not start",
                    "The terminal did not come up within 30 seconds.\n\n"
                    f"The log is at:\n    {data_dir() / 'terminal.log'}",
                )
                return 1
        else:
            logger.info("attaching to the terminal already on 127.0.0.1:%d", port)

        # Imported here rather than at the top so the paths above -- the
        # already-running check and a server that fails to start -- work and
        # can be tested on a machine with no window component at all.
        import webview

        try:
            window = webview.create_window(
                f"{self.title} — {describe()}",
                f"http://127.0.0.1:{port}",
                width=self.width, height=self.height,
                min_size=(1024, 700),
            )
            window.events.closing += self._confirm_close
            # Only hide the console once the window is about to appear. Hiding
            # it earlier would swallow the message from a failed startup, which
            # is the output that matters most.
            _hide_console()
            # Blocks until the window is closed, which is what makes this an
            # application: closing it ends the program rather than orphaning a
            # bot with a live position.
            webview.start()
        finally:
            # Only if this process is the one serving. Closing a window that
            # merely attached to the background service must not delete the
            # lock that names where that service is.
            if owned:
                lock.release()
        return 0

    def _running_port(self, lock: SingleInstance) -> int | None:
        """Where a terminal is already answering, if one is.

        Two places to look, because a terminal can be started two ways: a
        window records its own port in the lock, and the scheduled task serves
        the fixed default with no lock at all.
        """
        from godalgo.ui.server import port_owner

        port = lock.live_port()
        if port is not None:
            return port
        if port_owner("127.0.0.1", SERVICE_PORT) == "godalgo":
            return SERVICE_PORT
        return None

    def _confirm_close(self) -> bool:
        """Veto the close while live money is exposed, unless confirmed."""
        import webview

        warning = close_warning(self.bridge)
        if warning is None:
            return True
        try:
            return bool(
                webview.windows[0].create_confirmation_dialog("GODALGO", warning)
            )
        except Exception:
            logger.exception("could not ask before closing")
            return True


class _ServerThread(threading.Thread):
    """The UI server, on a daemon thread under the window."""

    def __init__(self, bridge: object, port: int) -> None:
        super().__init__(daemon=True, name="godalgo-server")
        self._bridge = bridge
        self._port = port
        self.error: BaseException | None = None

    def run(self) -> None:
        from godalgo.ui.server import run_server

        try:
            run_server(self._bridge, port=self._port, open_browser=False)
        except BaseException as exc:  # noqa: BLE001 - surfaced to the window
            self.error = exc
            logger.exception("the UI server stopped")

    def wait_until_ready(self, timeout: float) -> bool:
        """Poll until the server answers, so the window never shows a blank
        page or a connection error on first paint.

        In slices, so a server that has already died is noticed at once rather
        than waited out for the whole timeout.
        """
        from godalgo.ui.server import wait_until_serving

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.error is not None:
                return False
            if wait_until_serving("127.0.0.1", self._port, timeout=0.5):
                return True
        return False


def _message(title: str, body: str) -> None:
    """Tell the operator something, with no console to print to.

    A windowed build has no stdout anyone will see. Given how much of this
    project's history was spent on failures that produced no visible output,
    a fatal error must not be one of them.
    """
    logger.error("%s: %s", title, body.replace("\n", " "))
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.user32.MessageBoxW(None, body, title, 0x40)
            return
        except Exception:  # noqa: BLE001 - a dialog must never be fatal
            logger.debug("could not show a message box", exc_info=True)
    print(f"{title}\n{body}", file=sys.stderr)


def _configure_logging() -> Path:
    """Log to a file, because a windowed process has nowhere else to write."""
    path = data_dir() / "terminal.log"
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    return path


def run_desktop(bridge: object, *, title: str = "GODALGO") -> int:
    """Entry point for the windowed application."""
    log_path = _configure_logging()
    logger.info("starting desktop terminal; log at %s", log_path)

    if not available():
        _message(
            "GODALGO could not open a window",
            "The window component is missing from this build.\n\n"
            "Run with --browser to use a browser tab instead.",
        )
        return 1

    try:
        return DesktopApp(bridge, title=title).run()
    except Exception as exc:  # noqa: BLE001 - the last line before silence
        logger.exception("desktop terminal failed")
        _message(
            "GODALGO stopped unexpectedly",
            f"{type(exc).__name__}: {exc}\n\nThe log is at:\n    {log_path}",
        )
        return 1
