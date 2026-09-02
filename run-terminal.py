#!/usr/bin/env python3
"""GODALGO terminal launcher.

Double-clickable entry point, and the script PyInstaller wraps into a single
executable.

A browser tab is the default, and the browser is opened only once the server
actually answers -- see wait_until_serving. It used to open on a fixed
one-second timer, which is a guess, and on a cold start of the packaged build
it was the wrong one: the tab landed on a port nothing was listening to yet
and had to be refreshed until the server caught up.

    godalgo-terminal.exe                  # a browser tab on port 8787
    godalgo-terminal.exe --window         # a native window instead (Windows)
    godalgo-terminal.exe --no-browser     # serve only; what the service uses
    godalgo-terminal.exe --check-window   # is the window component present?

--window is opt-in rather than the default because it is the one mode that
cannot be verified in CI: a runner has no desktop, so only the presence of the
window component is checked, never the window opening.

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
        "--port", type=int, default=8787,
        help="port for the browser tab; --window picks a free one itself",
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
        help="open in a browser tab (the default; kept so existing shortcuts "
             "and scripts carry on working)",
    )
    parser.add_argument(
        "--window", action="store_true",
        help="open in a native window instead of a browser tab (Windows)",
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

    if args.window:
        from godalgo.ui import desktop

        if desktop.available():
            return desktop.run_desktop(bridge)
        print("no window component in this build; opening a browser tab")

    print(f"GODALGO terminal -> http://127.0.0.1:{args.port}")
    print("loopback only; this process holds credentials and has no auth")
    if not args.no_browser:
        print("the browser opens as soon as the terminal answers")
    try:
        # Host is fixed, not exposed as a flag. run_server rejects non-loopback
        # binds anyway, but not offering the option is the stronger guarantee.
        run_server(bridge, port=args.port, open_browser=not args.no_browser)
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
