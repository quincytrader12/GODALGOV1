#!/usr/bin/env python3
"""Build a standalone GODALGO terminal executable.

    pip install pyinstaller
    python build-exe.py

Produces ``dist/godalgo-terminal`` (``.exe`` on Windows) -- a single file that
runs the terminal with no Python installation on the target machine.

The two things that break a naive PyInstaller build of this project:

* **Static files.** The UI is served from ``godalgo/ui/static``. PyInstaller
  bundles imported modules, not data, so without an explicit mapping the
  executable starts, serves the API, and returns 404 for the page itself --
  which looks like a server fault rather than a packaging one.
* **Hidden imports.** uvicorn and ccxt resolve much of their machinery by name
  at runtime. Static analysis cannot see those, so they must be named here or
  the binary fails on first request rather than at build time.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
STATIC = ROOT / "src" / "godalgo" / "ui" / "static"
SEPARATOR = ";" if sys.platform == "win32" else ":"

# Resolved at runtime rather than imported, so PyInstaller cannot infer them.
HIDDEN = [
    "uvicorn.logging",
    "uvicorn.loops.auto",
    "uvicorn.loops.asyncio",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.http.h11_impl",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.protocols.websockets.websockets_impl",
    "uvicorn.lifespan.on",
    "ccxt.async_support",
    "ccxt.pro",
    "statsmodels.tsa.stattools",
    "scipy.special.cython_special",
]

# Excluded to keep the binary from carrying a plotting stack it never uses.
EXCLUDE = ["matplotlib", "tkinter", "PyQt5", "PySide2", "IPython", "notebook"]


def main() -> int:
    if not STATIC.exists():
        print(f"error: static files not found at {STATIC}", file=sys.stderr)
        return 1
    if shutil.which("pyinstaller") is None:
        print("error: pyinstaller not installed — run: pip install pyinstaller",
              file=sys.stderr)
        return 1

    command = [
        "pyinstaller",
        "--onefile",
        "--name", "godalgo-terminal",
        "--clean",
        "--noconfirm",
        # The mapping without which the packaged UI 404s on its own page.
        "--add-data", f"{STATIC}{SEPARATOR}godalgo/ui/static",
        "--paths", str(ROOT / "src"),
    ]
    for name in HIDDEN:
        command += ["--hidden-import", name]
    for name in EXCLUDE:
        command += ["--exclude-module", name]
    command.append(str(ROOT / "run-terminal.py"))

    print("building:", " ".join(command), "\n")
    result = subprocess.run(command, cwd=ROOT, check=False)
    if result.returncode != 0:
        return result.returncode

    binary = ROOT / "dist" / ("godalgo-terminal.exe" if sys.platform == "win32"
                              else "godalgo-terminal")
    if not binary.exists():
        print("error: build reported success but no binary was produced",
              file=sys.stderr)
        return 1

    size_mb = binary.stat().st_size / 1e6
    print(f"\nbuilt {binary} ({size_mb:.0f} MB)")
    print("run it with:")
    print(f"  {binary} --demo        # fabricated positions, nothing traded")
    print(f"  {binary}               # attached to a real session")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
