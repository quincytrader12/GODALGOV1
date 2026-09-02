#!/usr/bin/env python3
"""GODALGO terminal launcher.

Double-clickable entry point, and the script PyInstaller wraps into a single
executable.

By default on Windows it opens a native window on a port the OS chooses, which
is the mode that exists because the browser-tab mode kept failing in ways that
had nothing to do with trading -- a hardcoded port a second launch could not
bind, an out-of-date copy left serving silently behind the failure, and a tab
that could be closed while the bot went on holding a position unseen.

    godalgo-terminal.exe                  # a window
    godalgo-terminal.exe --browser        # a browser tab on port 8787
    godalgo-terminal.exe --no-browser     # serve only; what the service uses
    godalgo-terminal.exe --check-window   # is the window component present?

Packaging notes:

    pip install pyinstaller
    pyinstaller --onefile --name godalgo-terminal \
        --add-data "src/godalgo/ui/static:godalgo/ui/static" run-terminal.py

The bundled static files need that --add-data mapping; without it the
executable starts, serves the API, and returns 404 for the page itself.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Support running from a source checkout without installing the package.
_SRC = Path(__file__).parent / "src"
if _SRC.exists():
    sys.path.insert(0, str(_SRC))


def main() -> int:
    # ``service`` is dispatched before argparse rather than added as a
    # subparser: every other invocation is bare flags, and introducing
    # subcommands would break `godalgo-terminal.exe --port 8787`, which is what
    # a double-click and every existing shortcut do.
    if len(sys.argv) > 1 and sys.argv[1] == "service":
        from godalgo.service import main as service_main

        return service_main(sys.argv[2:])

    parser = argparse.ArgumentParser(
        description="GODALGO terminal",
        epilog="run it in the background on Windows: "
               "godalgo-terminal service install",
    )
    parser.add_argument(
        "--port", type=int, default=None,
        help="serve on a fixed port and use a browser tab; the window mode "
             "picks a free port itself, so this is only needed for the "
             "background service or a second copy",
    )
    parser.add_argument("--equity", type=float, default=10_000.0)
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument(
        "--demo", action="store_true",
        help="populate with fabricated positions -- nothing is traded",
    )
    parser.add_argument(
        "--no-browser", action="store_true",
        help="serve only, open nothing; what the background service uses",
    )
    parser.add_argument(
        "--browser", action="store_true",
        help="open in a browser tab instead of a window",
    )
    parser.add_argument(
        "--check-window", action="store_true",
        help="report whether the window component is present, then exit; "
             "this is what CI runs to catch a build that silently lost it",
    )
    parser.add_argument(
        "--bar-seconds", type=int, default=60,
        help="decision cadence; the bot acts once per completed bar",
    )
    args = parser.parse_args()

    if args.check_window:
        from godalgo.ui import desktop

        if desktop.available():
            print("window component: available")
            return 0
        print("window component: missing", file=sys.stderr)
        return 1

    import threading

    from godalgo.ui.server import UIBridge, build_terminal, run_server
    from godalgo.ui.simulator import Simulator

    if args.demo:
        import asyncio

        bridge = UIBridge(
            starting_equity=args.equity, equity=args.equity,
            market_feed_enabled=False,
        )
        bridge.symbol = args.symbol
        simulator = Simulator(bridge)
        simulator.seed_history()
        threading.Thread(
            target=lambda: asyncio.run(simulator.run()), daemon=True
        ).start()
        print("demo mode: positions are fabricated, not traded")
    else:
        # The real thing: a mode controller reading the credential store, and
        # a trading loop the mode switch actually drives. Without both, the
        # LIVE button is disabled and switching modes changes only a label.
        bridge = build_terminal(
            symbol=args.symbol, equity=args.equity, bar_seconds=args.bar_seconds,
        )

    # A window unless something asks otherwise. Every recent failure here was
    # a browser-tab failure and none of them were trading failures: a fixed
    # port that a second launch could not bind, an older copy left serving
    # silently, a tab that could be closed while the bot kept running unseen.
    # A window has its own port, its own taskbar entry, and closing it stops
    # the program.
    from godalgo.ui import desktop

    wants_server = args.no_browser or args.browser or args.port is not None
    if not wants_server and desktop.available():
        return desktop.run_desktop(bridge)

    if not wants_server:
        print("no window component in this build; opening a browser tab")

    port = args.port if args.port is not None else 8787
    print(f"GODALGO terminal -> http://127.0.0.1:{port}")
    print("loopback only; this process holds credentials and has no auth")
    try:
        # Host is fixed, not exposed as a flag. run_server rejects non-loopback
        # binds anyway, but not offering the option is the stronger guarantee.
        run_server(bridge, port=port, open_browser=not args.no_browser)
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
