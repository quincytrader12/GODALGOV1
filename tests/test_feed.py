"""Data feed logic that does not require network access.

The network path cannot be exercised here -- this sandbox blocks exchange APIs
-- so these cover the frame-normalisation rules, which is where the subtle
correctness problems actually live.
"""

import pandas as pd
import pytest

from godalgo.data.feed import OHLCVFeed, bars_per_day, timeframe_to_minutes


@pytest.mark.parametrize(
    ("timeframe", "minutes"),
    [("1m", 1), ("5m", 5), ("15m", 15), ("1h", 60), ("4h", 240), ("1d", 1440), ("1w", 10080)],
)
def test_timeframe_conversion(timeframe, minutes):
    assert timeframe_to_minutes(timeframe) == minutes


def test_bars_per_day():
    assert bars_per_day("1h") == 24.0
    assert bars_per_day("1d") == 1.0
    assert bars_per_day("15m") == 96.0


@pytest.mark.parametrize("bad", ["1y", "h", "", "0h", "-1h", "abc"])
def test_invalid_timeframes_raise(bad):
    with pytest.raises(ValueError):
        timeframe_to_minutes(bad)


def _raw(timestamps):
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 10.0,
        }
    )


def test_duplicate_timestamps_are_dropped():
    """Paginated windows overlap at the seams.

    A duplicated index silently corrupts every rolling window downstream, and
    nothing later in the pipeline would flag it.
    """
    frame = OHLCVFeed._to_frame(_raw([0, 3_600_000, 3_600_000, 7_200_000]))
    assert len(frame) == 3
    assert frame.index.is_unique


def test_frames_are_sorted_and_utc():
    frame = OHLCVFeed._to_frame(_raw([7_200_000, 0, 3_600_000]))
    assert frame.index.is_monotonic_increasing
    assert str(frame.index.tz) == "UTC"


def test_partial_final_bar_is_dropped():
    """The newest candle is still forming; trading its close is lookahead."""
    frame = OHLCVFeed._to_frame(_raw([0, 3_600_000, 7_200_000]))
    kept = OHLCVFeed._finalise(frame, limit=None, drop_partial=True)
    assert len(kept) == 2
    assert kept.index[-1] == frame.index[-2]


def test_partial_bar_can_be_retained_for_live_use():
    frame = OHLCVFeed._to_frame(_raw([0, 3_600_000, 7_200_000]))
    assert len(OHLCVFeed._finalise(frame, limit=None, drop_partial=False)) == 3


def test_limit_keeps_the_most_recent_bars():
    frame = OHLCVFeed._to_frame(_raw([0, 3_600_000, 7_200_000, 10_800_000]))
    kept = OHLCVFeed._finalise(frame, limit=2, drop_partial=False)
    assert len(kept) == 2
    assert kept.index[-1] == frame.index[-1]


def test_unknown_exchange_rejected():
    with pytest.raises(ValueError, match="unknown ccxt exchange"):
        _ = OHLCVFeed(exchange_id="not_a_real_exchange").exchange
