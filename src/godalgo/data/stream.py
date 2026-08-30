"""Bar aggregation from a live tick stream.

The live path must produce bars that are *identical in construction* to the ones
the backtest ran on. If live bars close on a different boundary, or include the
still-forming bar, the strategy sees a different series than the one it was
validated against -- and the backtest stops being evidence.

Two rules, mirroring the engine's backtest conventions:

* Bars close on wall-clock boundaries, not on tick counts, so a quiet minute
  still produces a bar at the same instant it would have in history.
* A bar is only emitted once it is **complete**. The forming bar is never handed
  to a strategy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import pandas as pd

__all__ = ["Bar", "BarAggregator"]


@dataclass(slots=True)
class Bar:
    """One OHLCV bar under construction."""

    start: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    trades: int = 0

    def update(self, price: float, size: float = 0.0) -> None:
        self.high = max(self.high, price)
        self.low = min(self.low, price)
        self.close = price
        self.volume += size
        self.trades += 1

    def as_row(self) -> dict[str, float]:
        return {
            "open": self.open, "high": self.high, "low": self.low,
            "close": self.close, "volume": self.volume,
        }


@dataclass
class BarAggregator:
    """Aggregates ticks into completed bars on wall-clock boundaries.

    Args:
        interval_seconds: Bar length. Must divide evenly into an hour for
            boundaries to align with historical bars from an exchange.
        max_bars: Rolling history retained, in bars. Must exceed the longest
            warm-up any strategy needs, or signals never leave their warm-up.
    """

    interval_seconds: int = 60
    max_bars: int = 2000
    _current: Bar | None = field(default=None, init=False)
    _rows: list[dict[str, float]] = field(default_factory=list, init=False)
    _index: list[datetime] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if self.interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        if self.max_bars < 2:
            raise ValueError("max_bars must be at least 2")

    def _bucket(self, moment: datetime) -> datetime:
        epoch = int(moment.timestamp())
        return datetime.fromtimestamp(
            epoch - (epoch % self.interval_seconds), tz=UTC
        )

    def on_tick(self, price: float, size: float = 0.0, moment: datetime | None = None) -> Bar | None:
        """Feed a trade. Returns a bar only when one has just *completed*."""
        if price <= 0:
            return None
        moment = moment or datetime.now(UTC)
        bucket = self._bucket(moment)

        if self._current is None:
            self._current = Bar(bucket, price, price, price, price)
            self._current.update(price, size)
            return None

        if bucket > self._current.start:
            completed = self._current
            self._append(completed)
            self._current = Bar(bucket, price, price, price, price)
            self._current.update(price, size)
            return completed

        self._current.update(price, size)
        return None

    def _append(self, bar: Bar) -> None:
        self._rows.append(bar.as_row())
        self._index.append(bar.start)
        if len(self._rows) > self.max_bars:
            self._rows.pop(0)
            self._index.pop(0)

    def seed(self, bars: pd.DataFrame) -> None:
        """Prime with historical bars so the bot is not blind at startup.

        Without this, a restart means waiting out the full warm-up -- hundreds of
        bars -- before any signal is valid. Seeding from REST history closes that
        gap.
        """
        for timestamp, row in bars.iterrows():
            self._index.append(timestamp.to_pydatetime())
            self._rows.append(
                {c: float(row[c]) for c in ("open", "high", "low", "close", "volume")}
            )
        excess = len(self._rows) - self.max_bars
        if excess > 0:
            del self._rows[:excess]
            del self._index[:excess]

    def frame(self) -> pd.DataFrame:
        """Completed bars only, oldest first.

        Excludes the forming bar by construction -- handing that to a strategy
        would be a lookahead in the live path that the backtest never had.
        """
        if not self._rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        return pd.DataFrame(self._rows, index=pd.DatetimeIndex(self._index, tz=UTC))

    @property
    def n_complete(self) -> int:
        return len(self._rows)

    def last_bar_age(self, now: datetime | None = None) -> timedelta | None:
        """Time since the last completed bar, for staleness detection."""
        if not self._index:
            return None
        return (now or datetime.now(UTC)) - self._index[-1]
