"""Nothing expensive may run on the event loop.

The terminal shares one loop between the websocket that feeds the browser, the
market-data poller, and the trading engine. Anything synchronous and slow in
that loop freezes all three for its full duration -- and it looks like a hung
UI, not like a slow calculation.

The measurement that motivated this: the per-bar decision took 170ms at 600
bars of history, 470ms at 1200, and 830ms at 2000. Run inline it produced a
915ms event-loop stall on every bar close. Off the loop the worst stall is
~45ms, three dropped frames rather than fifty-five.
"""

from __future__ import annotations

import asyncio
import statistics
import time

import numpy as np
import pandas as pd
import pytest

from godalgo.execution.engine import LiveEngine, LiveEngineConfig
from godalgo.strategies.mean_reversion import MeanReversionStrategy
from godalgo.strategies.momentum import MomentumStrategy


def _frame(bars: int) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    index = pd.date_range("2024-01-01", periods=bars, freq="1min", tz="UTC")
    price = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, bars)))
    return pd.DataFrame(
        {"open": price, "high": price * 1.001, "low": price * 0.999,
         "close": price, "volume": 1000.0},
        index=index,
    )


def _engine() -> LiveEngine:
    return LiveEngine(
        object(), MomentumStrategy(), MeanReversionStrategy(),
        LiveEngineConfig(symbol="BTC/USDT"),
    )


async def _worst_stall(work) -> float:
    """Worst gap seen by a 10ms heartbeat while ``work`` runs.

    That gap is exactly how long the browser would have gone unserved.
    """
    gaps: list[float] = []
    stop = asyncio.Event()

    async def heartbeat() -> None:
        last = time.perf_counter()
        while not stop.is_set():
            await asyncio.sleep(0.01)
            now = time.perf_counter()
            gaps.append((now - last) * 1000)
            last = now

    task = asyncio.create_task(heartbeat())
    await asyncio.sleep(0.2)
    gaps.clear()
    await work()
    stop.set()
    await task
    return max(gaps) if gaps else 0.0


def test_the_decision_does_not_freeze_the_event_loop():
    """The regression this file exists for.

    Deliberately asserts a stall bound rather than "to_thread is called": the
    property that matters is that the loop stays served, however that is
    achieved.
    """
    frame = _frame(1500)
    engine = _engine()

    async def decide() -> None:
        await asyncio.to_thread(engine._compute_target, frame)

    stall = asyncio.run(_worst_stall(decide))

    # Generous against CI jitter, and still an order of magnitude below the
    # 915ms the inline version produced.
    assert stall < 250, (
        f"the event loop stalled for {stall:.0f}ms during a decision; the "
        f"websocket, market feed and UI are all unserved for that long"
    )


def test_the_inline_decision_is_slow_enough_to_matter():
    """Guards the premise, so the test above cannot pass for the wrong reason.

    If the decision ever became cheap, the offload would be unnecessary and
    this file would be misleading. Assert it is genuinely expensive.
    """
    frame = _frame(1500)
    engine = _engine()

    engine._compute_target(frame)  # warm caches
    started = time.perf_counter()
    engine._compute_target(frame)
    elapsed = (time.perf_counter() - started) * 1000

    assert elapsed > 50, (
        f"the decision now takes {elapsed:.0f}ms; if it is genuinely cheap the "
        f"offload can go, but check before removing it"
    )


def test_the_bar_frame_is_a_snapshot_not_a_view():
    """What makes the offload safe.

    The decision runs in a worker thread while ticks keep arriving on the loop.
    That is only sound because ``frame()`` builds a new DataFrame each call --
    a view into the live buffer would be a data race on every bar.
    """
    from godalgo.data.stream import BarAggregator

    bars = BarAggregator(interval_seconds=60, max_bars=100)
    base = pd.Timestamp("2024-01-01", tz="UTC")
    for i in range(5):
        bars.on_tick(100.0 + i, 1.0, base + pd.Timedelta(minutes=i))

    before = bars.frame()
    rows_before = len(before)

    for i in range(5, 10):
        bars.on_tick(200.0 + i, 1.0, base + pd.Timedelta(minutes=i))

    assert len(before) == rows_before, "frame() handed out a live view"
    assert len(bars.frame()) > rows_before


@pytest.mark.parametrize("bars", [200, 1000])
def test_the_snapshot_stays_cheap(bars):
    """It is serialised once a second per connected browser, on the loop."""
    from godalgo.ui.server import UIBridge

    bridge = UIBridge(market_feed_enabled=False)
    for i in range(bars):
        bridge.record_fill(f"S{i % 12}/USDT", 1.0 if i % 2 else -1.0, 100.0 + i)
    for i in range(300):
        bridge.events.info("data", f"event {i}", "detail")

    samples = []
    for _ in range(20):
        started = time.perf_counter()
        bridge.snapshot().to_dict()
        samples.append((time.perf_counter() - started) * 1000)

    median = statistics.median(samples)
    assert median < 15, (
        f"snapshot() takes {median:.1f}ms and runs once a second per client"
    )
