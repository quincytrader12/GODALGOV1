"""Exchange API credential storage for the local UI.

These are keys that can move money, so the handling rules are strict and worth
stating rather than implying:

* **Stored outside the repository**, under ``~/.godalgo/``, so they cannot be
  committed by accident. A credential in a repo is a credential on GitHub.
* **File mode 0600**, owner-only, set at creation rather than after -- a
  chmod-after-write leaves a window where the file is world-readable.
* **Never returned over HTTP.** The API serves a masked view only. The UI shows
  you which keys exist, never what they are; a browser tab is not a place to
  keep a secret, and neither is a browser's network log.
* **Never logged**, at any level.
* **Read-only by default.** A key added here is marked ``trade_enabled=False``
  unless explicitly set, so a misconfigured venue cannot place orders because
  someone pasted a key into a form.

The environment remains the preferred source for the keys the bot actually
trades with (``GODALGO_API_KEY`` / ``GODALGO_API_SECRET``). This store exists
because the UI needs to manage several venues, which a single env pair cannot.
"""

from __future__ import annotations

import json
import logging
import os
import stat
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

__all__ = ["CredentialStore", "ExchangeCredential"]

logger = logging.getLogger(__name__)

_DEFAULT_DIR = Path.home() / ".godalgo"


@dataclass(frozen=True, slots=True)
class ExchangeCredential:
    """One venue's credentials."""

    exchange_id: str
    api_key: str
    api_secret: str
    label: str = ""
    passphrase: str = ""
    """Required by some venues (Coinbase, OKX, KuCoin)."""

    testnet: bool = False
    trade_enabled: bool = False
    """Whether this credential may place orders. Off unless explicitly enabled."""

    added_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def masked(self) -> dict[str, Any]:
        """A view safe to serve over HTTP and render in a browser.

        Shows enough of the key to identify which one it is and nothing more.
        The secret is never included in any form, masked or otherwise.
        """
        return {
            "exchange_id": self.exchange_id,
            "label": self.label or self.exchange_id,
            "api_key_masked": _mask(self.api_key),
            "has_secret": bool(self.api_secret),
            "has_passphrase": bool(self.passphrase),
            "testnet": self.testnet,
            "trade_enabled": self.trade_enabled,
            "added_at": self.added_at,
        }


def _mask(value: str) -> str:
    """Show the first four and last four characters, nothing between."""
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}{'*' * 6}{value[-4:]}"


@dataclass
class CredentialStore:
    """Owner-readable JSON store of exchange credentials."""

    directory: Path = field(default_factory=lambda: _DEFAULT_DIR)
    _credentials: dict[str, ExchangeCredential] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.directory = Path(self.directory)
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.load()

    @property
    def path(self) -> Path:
        return self.directory / "credentials.json"

    def load(self) -> None:
        """Read the store, warning if the file is more permissive than 0600."""
        if not self.path.exists():
            return
        mode = stat.S_IMODE(self.path.stat().st_mode)
        if mode & 0o077:
            logger.warning(
                "%s is readable by others (mode %o); run chmod 600 on it",
                self.path, mode,
            )
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("could not read credential store: %s", type(exc).__name__)
            return
        self._credentials = {
            key: ExchangeCredential(**value) for key, value in raw.items()
        }

    def save(self) -> None:
        """Write atomically with owner-only permissions from creation.

        The temp file is created 0600 *before* any secret is written to it, so
        the data is never briefly world-readable on disk.
        """
        payload = {
            key: {
                "exchange_id": c.exchange_id, "api_key": c.api_key,
                "api_secret": c.api_secret, "label": c.label,
                "passphrase": c.passphrase, "testnet": c.testnet,
                "trade_enabled": c.trade_enabled, "added_at": c.added_at,
            }
            for key, c in self._credentials.items()
        }
        temp = self.path.with_suffix(".tmp")
        descriptor = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
        temp.replace(self.path)

    def add(self, credential: ExchangeCredential) -> None:
        key = credential.label or credential.exchange_id
        self._credentials[key] = credential
        self.save()
        # Deliberately no key material in the log line.
        logger.info(
            "stored credential for %s (trade_enabled=%s)",
            credential.exchange_id, credential.trade_enabled,
        )

    def remove(self, key: str) -> bool:
        if key not in self._credentials:
            return False
        del self._credentials[key]
        self.save()
        logger.info("removed credential %s", key)
        return True

    def get(self, key: str) -> ExchangeCredential | None:
        return self._credentials.get(key)

    def set_trade_enabled(self, key: str, enabled: bool) -> bool:
        current = self._credentials.get(key)
        if current is None:
            return False
        self._credentials[key] = ExchangeCredential(
            exchange_id=current.exchange_id, api_key=current.api_key,
            api_secret=current.api_secret, label=current.label,
            passphrase=current.passphrase, testnet=current.testnet,
            trade_enabled=enabled, added_at=current.added_at,
        )
        self.save()
        return True

    def listing(self) -> list[dict[str, Any]]:
        """Masked view of every stored credential. Safe to serve."""
        return [c.masked() for c in self._credentials.values()]

    def __len__(self) -> int:
        return len(self._credentials)
