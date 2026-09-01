"""Market data via ccxt.

Responsibilities kept narrow on purpose: fetch OHLCV, paginate correctly, cache
to disk, and hand back a clean frame. No indicators, no signals -- if this module
starts making decisions, the backtest and the live path stop sharing a single
definition of what the data is.

The correctness issues that matter here are unglamorous and account for most
real backtest/live divergence:

* **Pagination.** Exchanges cap a single OHLCV call (commonly 500-1500 bars).
  Requesting a year of hourly bars in one call silently returns the cap, and the
  backtest then runs on a fraction of the history the caller believes it has.
* **The partial final bar.** The most recent candle is still forming. Including
  it means training on a bar whose close does not exist yet -- a lookahead that
  is invisible in the output and flattering in the results.
* **Deduplication and ordering.** Paginated windows overlap at the seams, and
  duplicate timestamps quietly corrupt every rolling window downstream.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import ccxt
import pandas as pd

__all__ = ["OHLCVFeed", "bars_per_day", "timeframe_to_minutes"]

logger = logging.getLogger(__name__)

_OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def timeframe_to_minutes(timeframe: str) -> int:
    """Convert a ccxt timeframe string to minutes.

    Args:
        timeframe: e.g. ``"1m"``, ``"15m"``, ``"1h"``, ``"4h"``, ``"1d"``, ``"1w"``.

    Raises:
        ValueError: On an unrecognised unit or a non-numeric quantity.
    """
    units = {"m": 1, "h": 60, "d": 1440, "w": 10080}
    if len(timeframe) < 2:
        raise ValueError(f"unrecognised timeframe {timeframe!r}")
    unit = timeframe[-1].lower()
    if unit not in units:
        raise ValueError(f"unrecognised timeframe unit in {timeframe!r}")
    try:
        quantity = int(timeframe[:-1])
    except ValueError as exc:
        raise ValueError(f"unrecognised timeframe {timeframe!r}") from exc
    if quantity <= 0:
        raise ValueError(f"timeframe quantity must be positive, got {timeframe!r}")
    return quantity * units[unit]


def bars_per_day(timeframe: str) -> float:
    """Bars per calendar day, for the engine's annualisation."""
    return 1440.0 / timeframe_to_minutes(timeframe)


@dataclass(slots=True)
class OHLCVFeed:
    """Paginated, cached OHLCV loader for one exchange.

    Args:
        exchange_id: Any ccxt exchange id, e.g. ``"binance"``, ``"kraken"``.
        cache_dir: Directory for parquet caches. ``None`` disables caching.
        rate_limit: Whether to honour ccxt's built-in throttle. Leave True --
            the alternative is a ban.
        max_retries: Attempts per page on transient network or exchange errors.
    """

    exchange_id: str = "binance"
    cache_dir: Path | None = None
    rate_limit: bool = True
    max_retries: int = 3
    _exchange: ccxt.Exchange | None = None

    def __post_init__(self) -> None:
        if self.cache_dir is not None:
            self.cache_dir = Path(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def exchange(self) -> ccxt.Exchange:
        """Lazily constructed ccxt client.

        Read-only: no API keys are accepted here. Market data needs no
        credentials, and a data loader that cannot authenticate cannot place an
        order by accident.
        """
        if self._exchange is None:
            if self.exchange_id not in ccxt.exchanges:
                raise ValueError(f"unknown ccxt exchange {self.exchange_id!r}")
            klass = getattr(ccxt, self.exchange_id)
            # ccxt leaves aiohttp's trust_env off, so it ignores HTTPS_PROXY and the
            # system proxy entirely. On a machine behind a corporate proxy, a VPN
            # client, or antivirus that intercepts TLS, every request fails while the
            # browser beside it works -- which reads as "the exchange is down".
            self._exchange = klass({
                "enableRateLimit": self.rate_limit, "aiohttp_trust_env": True,
            })
        return self._exchange

    def supports_ohlcv(self) -> bool:
        """Whether this exchange implements ``fetchOHLCV``.

        Worth checking before building anything on it -- the unified API is
        unified in shape, not in coverage, and a handful of venues do not
        provide candles at all.
        """
        return bool(self.exchange.has.get("fetchOHLCV"))

    def fetch(
        self,
        symbol: str,
        timeframe: str = "1h",
        *,
        since: str | pd.Timestamp | None = None,
        limit: int | None = None,
        drop_partial: bool = True,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Fetch OHLCV bars, paginating until ``limit`` or the present.

        Args:
            symbol: Unified market symbol, e.g. ``"BTC/USDT"``.
            timeframe: ccxt timeframe string.
            since: Start time. Defaults to as far back as ``limit`` requires.
            limit: Maximum bars to return. ``None`` fetches to the present.
            drop_partial: Drop the final, still-forming candle. Keep this True
                for anything that feeds a backtest.
            use_cache: Read from and write to the parquet cache when configured.

        Returns:
            Frame indexed by UTC timestamp with ``open``, ``high``, ``low``,
            ``close``, ``volume``, sorted oldest-first and deduplicated.

        Raises:
            ValueError: If the exchange does not support OHLCV.
        """
        if not self.supports_ohlcv():
            raise ValueError(f"{self.exchange_id} does not support fetchOHLCV")

        cache_path = self._cache_path(symbol, timeframe) if use_cache else None
        if cache_path is not None and cache_path.exists():
            cached = pd.read_parquet(cache_path)
            if not cached.empty and (limit is None or len(cached) >= limit):
                logger.debug("cache hit for %s %s (%d bars)", symbol, timeframe, len(cached))
                return self._finalise(cached, limit, drop_partial)

        minutes = timeframe_to_minutes(timeframe)
        step_ms = minutes * 60_000

        since_ms = self._resolve_since(since, step_ms, limit)
        frames: list[pd.DataFrame] = []
        fetched = 0

        while True:
            batch = self._fetch_page(symbol, timeframe, since_ms)
            if not batch:
                break

            frames.append(pd.DataFrame(batch, columns=_OHLCV_COLUMNS))
            fetched += len(batch)

            last_ts = batch[-1][0]
            # Advance past the last bar received. Without the step the next call
            # returns the same page and the loop never terminates.
            next_since = last_ts + step_ms

            if limit is not None and fetched >= limit:
                break
            if next_since >= self.exchange.milliseconds():
                break
            if len(batch) == 1 and next_since <= since_ms:
                break
            since_ms = next_since

        if not frames:
            return pd.DataFrame(columns=_OHLCV_COLUMNS[1:])

        raw = pd.concat(frames, ignore_index=True)
        frame = self._to_frame(raw)

        if cache_path is not None:
            frame.to_parquet(cache_path)

        return self._finalise(frame, limit, drop_partial)

    def _fetch_page(self, symbol: str, timeframe: str, since_ms: int | None) -> list[list[float]]:
        """One OHLCV page, with bounded retries on transient failures."""
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                return self.exchange.fetch_ohlcv(symbol, timeframe, since=since_ms)
            except (ccxt.NetworkError, ccxt.ExchangeNotAvailable, ccxt.RequestTimeout) as exc:
                last_error = exc
                # Exponential backoff. Retrying immediately against a rate limit
                # is how a transient error becomes a ban.
                delay = 2.0**attempt
                logger.warning(
                    "fetch %s %s attempt %d/%d failed (%s); retrying in %.0fs",
                    symbol, timeframe, attempt + 1, self.max_retries, exc, delay,
                )
                time.sleep(delay)
            except ccxt.BadSymbol:
                raise
        if last_error is not None:
            raise last_error
        return []

    @staticmethod
    def _to_frame(raw: pd.DataFrame) -> pd.DataFrame:
        """Index by UTC timestamp, deduplicate, sort."""
        frame = raw.copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
        frame = frame.set_index("timestamp").sort_index()
        # Paginated windows overlap at the seams; duplicates silently corrupt
        # every rolling calculation downstream.
        frame = frame[~frame.index.duplicated(keep="first")]
        return frame.astype(float)

    @staticmethod
    def _finalise(frame: pd.DataFrame, limit: int | None, drop_partial: bool) -> pd.DataFrame:
        out = frame
        if drop_partial and len(out) > 1:
            out = out.iloc[:-1]
        if limit is not None and len(out) > limit:
            out = out.iloc[-limit:]
        return out

    def _resolve_since(
        self, since: str | pd.Timestamp | None, step_ms: int, limit: int | None
    ) -> int | None:
        if since is not None:
            return int(pd.Timestamp(since, tz="UTC").timestamp() * 1000)
        if limit is not None:
            # Over-fetch by 20% so that dropping the partial bar and any
            # exchange-side gaps still leaves `limit` usable bars.
            span = int(limit * step_ms * 1.2)
            return self.exchange.milliseconds() - span
        return None

    def _cache_path(self, symbol: str, timeframe: str) -> Path | None:
        if self.cache_dir is None:
            return None
        safe = symbol.replace("/", "-").replace(":", "_")
        return self.cache_dir / f"{self.exchange_id}_{safe}_{timeframe}.parquet"
