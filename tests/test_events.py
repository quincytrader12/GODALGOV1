"""The activity log.

Its reason for existing: a bot that is working and a bot that is silently doing
nothing look identical from the outside. Everything here protects that
usefulness -- and the rule that no secret may ever reach it, since it is
rendered in a browser and screenshotted.
"""

from __future__ import annotations

import json

from godalgo.ui.events import EventLog


def test_events_come_back_newest_first():
    """The UI shows the top of the list; the newest thing must be there."""
    events = EventLog()
    for i in range(5):
        events.info("venue", f"event {i}")
    assert [e["message"] for e in events.entries()] == [
        "event 4", "event 3", "event 2", "event 1", "event 0",
    ]


def test_the_ring_is_bounded():
    """This runs for weeks unattended; an unbounded list is a slow leak."""
    events = EventLog(capacity=10)
    for i in range(100):
        events.info("venue", f"event {i}")
    entries = events.entries(limit=1000)
    assert len(entries) == 10
    assert entries[0]["message"] == "event 99"


def test_since_returns_only_what_the_client_has_not_seen():
    events = EventLog()
    events.info("venue", "first")
    mark = events.latest_sequence
    events.info("venue", "second")

    fresh = events.entries(since=mark)
    assert [e["message"] for e in fresh] == ["second"]


def test_sequence_numbers_are_monotonic_across_levels():
    events = EventLog()
    events.info("a", "1"); events.warn("b", "2"); events.error("c", "3"); events.good("d", "4")
    sequences = [e["sequence"] for e in reversed(events.entries())]
    assert sequences == sorted(sequences)
    assert len(set(sequences)) == 4


def test_a_verbose_detail_is_truncated_rather_than_dropped():
    """A ccxt error body can run to kilobytes.

    Losing the event because it was verbose is worse than losing its tail.
    """
    events = EventLog()
    events.error("venue", "rejected", "x" * 5000)
    detail = events.entries()[0]["detail"]
    assert len(detail) < 500
    assert detail.endswith("…")


def test_counts_are_reported_per_level():
    events = EventLog()
    events.info("a", "i"); events.warn("a", "w"); events.warn("a", "w2"); events.error("a", "e")
    assert events.counts() == {"info": 1, "good": 0, "warn": 2, "error": 1}


def test_events_serialise_to_json():
    """They travel in the websocket snapshot; a non-serialisable field would
    take the whole stream down, not just the log."""
    events = EventLog()
    events.good("venue", "connected", "42 markets")
    blob = json.dumps(events.entries())
    assert "connected" in blob


def test_the_log_survives_concurrent_writers():
    """Writers are a websocket task, an engine loop and a poller, none of
    which coordinate."""
    import threading

    events = EventLog(capacity=1000)

    def spam(n):
        for i in range(100):
            events.info("venue", f"{n}-{i}")

    threads = [threading.Thread(target=spam, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    entries = events.entries(limit=1000)
    assert len(entries) == 800
    # Every sequence number is distinct: no two writers claimed the same slot.
    assert len({e["sequence"] for e in entries}) == 800
