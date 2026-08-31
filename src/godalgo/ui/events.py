"""What the terminal is doing, as a readable stream.

Every venue call, credential change, mode switch and failure lands here, and
the UI renders it. The motivation is concrete: a bot that is working and a bot
that is silently doing nothing look identical from the outside, and the second
one is the expensive case. Without a log the only way to tell them apart is to
fund an account and watch, which is exactly the wrong order to do things in.

Two rules hold everywhere in this module:

* **Failures are events, not exceptions.** A venue that rejects a key is
  information the operator needs on screen; raising it into the caller only
  loses it.
* **No secret ever enters an event.** Not the key, not the secret, not a bot
  token, not a URL that embeds one. Events are rendered in a browser, kept in
  memory, and read aloud in screenshots.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

__all__ = ["Event", "EventLog", "Level"]

logger = logging.getLogger(__name__)

Level = Literal["info", "good", "warn", "error"]

_MAX_DETAIL = 400


@dataclass(frozen=True, slots=True)
class Event:
    """One thing that happened."""

    at: datetime
    level: Level
    category: str
    """Coarse source: ``venue``, ``credentials``, ``mode``, ``telegram``,
    ``data``, ``engine``. Used by the UI to filter, so keep it stable."""

    message: str
    detail: str = ""
    sequence: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "at": self.at.isoformat(),
            "level": self.level,
            "category": self.category,
            "message": self.message,
            "detail": self.detail,
            "sequence": self.sequence,
        }


@dataclass
class EventLog:
    """A bounded, thread-safe ring of recent events.

    Bounded because this runs for weeks unattended and an unbounded list is a
    slow memory leak. Thread-safe because the writers are a websocket task, an
    engine loop, and a background poller, none of which coordinate.

    Args:
        capacity: How many events to retain. The UI shows far fewer; the extra
            headroom is for reading back after something goes wrong.
    """

    capacity: int = 500
    _events: deque[Event] = field(init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _sequence: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self._events = deque(maxlen=self.capacity)

    def record(
        self, level: Level, category: str, message: str, detail: str = ""
    ) -> Event:
        """Append an event and return it.

        The detail is truncated rather than rejected: a ccxt error body can run
        to kilobytes, and losing the event entirely because it was verbose is
        worse than losing its tail.
        """
        text = str(detail or "")
        if len(text) > _MAX_DETAIL:
            text = text[: _MAX_DETAIL - 1] + "…"

        with self._lock:
            self._sequence += 1
            event = Event(
                at=datetime.now(UTC),
                level=level,
                category=category,
                message=str(message),
                detail=text,
                sequence=self._sequence,
            )
            self._events.append(event)

        # Mirror to the standard logger so a headless run keeps the same
        # record. Errors are logged as warnings, not errors: these are
        # conditions the system handles, and a stack trace here would imply
        # something crashed.
        logger.log(
            logging.WARNING if level in ("warn", "error") else logging.INFO,
            "%s: %s%s", category, message, f" ({text})" if text else "",
        )
        return event

    def info(self, category: str, message: str, detail: str = "") -> Event:
        return self.record("info", category, message, detail)

    def good(self, category: str, message: str, detail: str = "") -> Event:
        return self.record("good", category, message, detail)

    def warn(self, category: str, message: str, detail: str = "") -> Event:
        return self.record("warn", category, message, detail)

    def error(self, category: str, message: str, detail: str = "") -> Event:
        return self.record("error", category, message, detail)

    def entries(self, limit: int = 60, since: int = 0) -> list[dict[str, Any]]:
        """Most recent first.

        Args:
            limit: Maximum returned.
            since: Return only events with a sequence above this. Lets the UI
                poll without re-rendering what it already has.
        """
        with self._lock:
            events = [e for e in self._events if e.sequence > since]
        return [e.to_dict() for e in reversed(events[-limit:])]

    @property
    def latest_sequence(self) -> int:
        with self._lock:
            return self._sequence

    def counts(self) -> dict[str, int]:
        """Totals by level, for the header indicators."""
        with self._lock:
            events = list(self._events)
        out = {"info": 0, "good": 0, "warn": 0, "error": 0}
        for event in events:
            out[event.level] = out.get(event.level, 0) + 1
        return out
