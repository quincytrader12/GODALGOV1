"""Telegram notifications.

Credentials come from the environment only -- never from a config file, never
from the UI's saved settings, never logged. A bot token in a log line is a bot
token in someone else's hands.

Failures here are deliberately non-fatal. A notifier that can halt the trading
loop is a liability: losing a chat message is an inconvenience, while an
unhandled exception in a notification path taking down a bot holding a position
is not.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

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
    _token: str | None = None
    _chat_id: str | None = None

    def __post_init__(self) -> None:
        self._token = os.environ.get(_TOKEN_ENV)
        self._chat_id = os.environ.get(_CHAT_ENV)

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
            return False, f"set {_TOKEN_ENV} and {_CHAT_ENV} in the environment"
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{_API}/bot{self._token}/getMe")
            if response.status_code != 200:
                return False, f"telegram rejected the token (HTTP {response.status_code})"
            name = (response.json().get("result") or {}).get("username", "unknown")
            return True, f"connected as @{name}"
        except httpx.HTTPError as exc:
            return False, f"could not reach telegram: {type(exc).__name__}"
