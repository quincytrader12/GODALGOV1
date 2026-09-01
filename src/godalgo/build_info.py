"""Which build this is.

Added because a whole debugging round was spent on a bug that had already been
fixed: a second copy could not bind its port, died, and the *older* copy kept
serving. The browser showed a terminal that looked perfectly healthy while
being several builds behind, and nothing on screen could have revealed that.

So the build stamp is visible in the interface and in ``/api/state``. It costs
nothing and turns "it still does the thing you fixed" into a checkable claim.

The stamp is written by ``build-exe.py`` into ``_build_stamp.py``. Running from
a source checkout there is no such file, so the commit is read from git
instead, and failing that it degrades to "dev" rather than raising -- a version
banner must never be the thing that stops the program starting.
"""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

from godalgo import __version__

__all__ = ["build_stamp", "describe"]


@lru_cache(maxsize=1)
def build_stamp() -> dict[str, str]:
    """``{version, commit, built_at, source}``. Never raises."""
    try:
        from godalgo import _build_stamp  # type: ignore[attr-defined]

        return {
            "version": __version__,
            "commit": getattr(_build_stamp, "COMMIT", "unknown"),
            "built_at": getattr(_build_stamp, "BUILT_AT", ""),
            "source": "packaged",
        }
    except ImportError:
        pass

    commit = "dev"
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=Path(__file__).resolve().parents[2], check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            commit = result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass

    return {
        "version": __version__,
        "commit": commit,
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": "checkout",
    }


def describe() -> str:
    """One line for a title bar or a bug report."""
    stamp = build_stamp()
    built = (stamp["built_at"] or "")[:10]
    return f"v{stamp['version']} · {stamp['commit']}" + (f" · {built}" if built else "")
