#!/usr/bin/env python3
"""GODALGO terminal launcher.

Double-clickable entry point, and the script PyInstaller wraps into a single
executable:

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
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--equity", type=float, default=10_000.0)
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument(
        "--demo", action="store_true",
        help="populate with fabricated positions -- nothing is traded",
    )
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    import threading

    from godalgo.ui.server import UIBridge, run_server
    from godalgo.ui.simulator import Simulator

    bridge = UIBridge(starting_equity=args.equity, equity=args.equity)
    bridge.symbol = args.symbol

    if args.demo:
        import asyncio

        simulator = Simulator(bridge)
        simulator.seed_history()
        threading.Thread(
            target=lambda: asyncio.run(simulator.run()), daemon=True
        ).start()
        print("demo mode: positions are fabricated, not traded")

    print(f"GODALGO terminal -> http://127.0.0.1:{args.port}")
    print("loopback only; this process holds credentials and has no auth")
    try:
        # Host is fixed, not exposed as a flag. run_server rejects non-loopback
        # binds anyway, but not offering the option is the stronger guarantee.
        run_server(bridge, port=args.port, open_browser=not args.no_browser)
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
