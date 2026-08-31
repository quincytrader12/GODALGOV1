"""Telegram notifications.

The bot token is a credential and is handled like one: stored owner-only
alongside the exchange keys, **never logged**, and never returned over HTTP --
the status view reports only whether a token is present and the last four
digits of the chat id. A token in a log line is a token in someone else's
hands, and the send URL embeds it, so failures log a status code and never a
URL.

It can be set from the environment or from the terminal. Environment-only was
the original design and was wrong for the same reason it was wrong for live
arming: the terminal ships as a double-clicked executable, where "set an
environment variable first" is not a security control, it is a feature nobody
can reach.

Failures here are deliberately non-fatal. A notifier that can halt the trading
loop is a liability: losing a chat message is an inconvenience, while an
unhandled exception in a notification path taking down a bot holding a position
is not.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import httpx

__all__ = ["TelegramNotifier"]

logger = logging.getLogger(__name__)

_TOKEN_ENV = "GODALGO_TELEGRAM_TOKEN"
_CHAT_ENV = "GODALGO_TELEGRAM_CHAT_ID"
_API = "https://api.telegram.org"


@dataclass
class TelegramNotifier:
    """Sends messages to one chat.

    Args:
        timeout: Per-request timeout. Short on purpose -- the trading loop must
            never wait on a chat API.
    """

    timeout: float = 10.0
    store: Any = None
    """Optional owner-only store used to persist the token across restarts.

    Duck-typed rather than imported so this module keeps no dependency on the
    UI layer; anything with ``get_secret``/``set_secret`` will do.
    """

    _token: str | None = None
    _chat_id: str | None = None

    def __post_init__(self) -> None:
        # Environment first, so an operator who has deliberately exported a
        # token is not silently overridden by something saved months ago.
        self._token = os.environ.get(_TOKEN_ENV)
        self._chat_id = os.environ.get(_CHAT_ENV)
        if not self.configured:
            self._load_from_store()

    def _load_from_store(self) -> None:
        if self.store is None:
            return
        try:
            saved = self.store.get_secret("telegram") or {}
        except Exception:
            logger.debug("could not read the telegram config", exc_info=True)
            return
        self._token = self._token or saved.get("token") or None
        self._chat_id = self._chat_id or saved.get("chat_id") or None

    def configure(self, token: str, chat_id: str) -> None:
        """Set and persist the credentials. Never logs either value."""
        self._token = (token or "").strip() or None
        self._chat_id = (chat_id or "").strip() or None
        if self.store is not None and self.configured:
            try:
                self.store.set_secret(
                    "telegram", {"token": self._token, "chat_id": self._chat_id}
                )
            except OSError:
                # Non-fatal: the notifier still works this session, it just
                # will not survive a restart.
                logger.error("could not persist the telegram config")
        logger.info("telegram configured (token not logged)")

    def clear(self) -> None:
        self._token = None
        self._chat_id = None
        if self.store is not None:
            try:
                self.store.set_secret("telegram", {})
            except OSError:
                logger.error("could not clear the telegram config")

    @property
    def configured(self) -> bool:
        return bool(self._token and self._chat_id)

    @property
    def status(self) -> dict[str, object]:
        """Configuration state, with no secret in it.

        Reports only whether a token is present and a masked tail of the chat
        id, because this is rendered in the UI and served over HTTP.
        """
        return {
            "configured": self.configured,
            "token_present": bool(self._token),
            "chat_id_tail": self._chat_id[-4:] if self._chat_id else None,
            "token_env": _TOKEN_ENV,
            "chat_env": _CHAT_ENV,
        }

    async def send(self, text: str, *, markdown: bool = True) -> bool:
        """Send a message. Returns whether it was delivered.

        Never raises. Every failure path returns False and logs, so a notifier
        problem cannot propagate into the trading loop.
        """
        if not self.configured:
            logger.debug("telegram not configured; message dropped")
            return False

        payload = {
            "chat_id": self._chat_id,
            "text": text,
            "disable_web_page_preview": True,
        }
        if markdown:
            payload["parse_mode"] = "Markdown"

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{_API}/bot{self._token}/sendMessage", json=payload
                )
            if response.status_code == 200:
                return True
            # Log the status but never the URL -- it contains the token.
            logger.error("telegram send failed: HTTP %d", response.status_code)
            return False
        except httpx.HTTPError as exc:
            logger.error("telegram send failed: %s", type(exc).__name__)
            return False

    async def verify(self) -> tuple[bool, str]:
        """Check the token against getMe, for the settings panel."""
        if not self.configured:
            return False, "add a bot token and chat id first"
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{_API}/bot{self._token}/getMe")
            if response.status_code != 200:
                return False, f"telegram rejected the token (HTTP {response.status_code})"
            name = (response.json().get("result") or {}).get("username", "unknown")
            return True, f"connected as @{name}"
        except httpx.HTTPError as exc:
            return False, f"could not reach telegram: {type(exc).__name__}"
