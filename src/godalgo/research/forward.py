"""The only unbiased evidence this system produces.

A search picks its winner after seeing how the data turned out. Every
correction in ``selection.py`` is an attempt to undo that, and none of them
fully can. The one immune form of evidence is a track record on days that had
not happened yet.

Which makes this file's real feature **persistence**. Its value is months of
accumulation, so it survives restarts, crashes and rebuilds by being appended
to disk as it goes rather than held in memory and summarised at the end. A
forward record that resets on restart is not a forward record.

What it is used for, in order of importance:

* **Sizing.** ``godalgo.portfolio.book`` scales a strategy's allocation by the
  length of its forward record, never by the in-sample Sharpe that got it
  adopted. This module is where that length comes from.
* **Retirement.** A live record that diverges from its backtest retires the
  strategy to zero weight rather than starting a debate. The test is defined in
  advance here, because after a bad month it will not be.
* **Attribution.** By strategy, symbol and regime, every day. A book that is up
  while three of its four strategies are losing is one strategy's bet, and that
  is worth knowing before the fourth turns.

Every number reported carries its uncertainty. A live Sharpe from forty days is
almost uninformative, and printing it bare invites a decision it cannot
support -- so the bootstrap interval travels with it, and when that interval
spans zero the record is described as consistent with having no edge.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "Attribution",
    "DivergenceRule",
    "ForwardRecord",
    "SharpeEstimate",
    "TrackEntry",
    "bootstrap_sharpe",
    "cost_breakeven_bps",
]

logger = logging.getLogger(__name__)

_ANNUAL = 252.0


@dataclass(frozen=True, slots=True)
class TrackEntry:
    """One strategy's result for one day.

    Deliberately a daily row rather than a per-trade one. Trades are what the
    system does; days are what the evidence is measured in, and a per-trade
    record makes a high-turnover strategy look like it has more history than a
    patient one with the same elapsed time.
    """

    day: date
    strategy: str
    symbol: str
    pnl: float
    ret: float
    """Return on equity for the day, as a fraction."""

    regime: str = "indeterminate"
    mode: str = "paper"
    """Paper and live records are kept apart. Merging them would let paper
    days inflate the track length that decides live sizing."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "day": self.day.isoformat(), "strategy": self.strategy,
            "symbol": self.symbol, "pnl": self.pnl, "ret": self.ret,
            "regime": self.regime, "mode": self.mode,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> TrackEntry:
        return cls(
            day=date.fromisoformat(raw["day"]),
            strategy=raw.get("strategy", "unknown"),
            symbol=raw.get("symbol", ""),
            pnl=float(raw.get("pnl", 0.0)),
            ret=float(raw.get("ret", 0.0)),
            regime=raw.get("regime", "indeterminate"),
            mode=raw.get("mode", "paper"),
        )


@dataclass(frozen=True, slots=True)
class SharpeEstimate:
    """A Sharpe with the uncertainty that makes it usable.

    A point estimate from a short record is not a small amount of evidence, it
    is a number that looks like evidence. The interval is what stops it being
    acted on.
    """

    sharpe: float
    low: float
    high: float
    days: int

    @property
    def spans_zero(self) -> bool:
        return self.low <= 0.0 <= self.high

    @property
    def summary(self) -> str:
        if self.days < 20:
            return (
                f"{self.days} days is too short to say anything: Sharpe "
                f"{self.sharpe:.2f} but the record cannot support a conclusion"
            )
        if self.spans_zero:
            return (
                f"Sharpe {self.sharpe:.2f} over {self.days} days, 95% interval "
                f"[{self.low:.2f}, {self.high:.2f}] — spans zero, so this "
                f"record is consistent with having no edge"
            )
        direction = "positive" if self.low > 0 else "negative"
        return (
            f"Sharpe {self.sharpe:.2f} over {self.days} days, 95% interval "
            f"[{self.low:.2f}, {self.high:.2f}] — {direction} at 95%"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sharpe": self.sharpe, "low": self.low, "high": self.high,
            "days": self.days, "spans_zero": self.spans_zero,
            "summary": self.summary,
        }


def bootstrap_sharpe(
    returns: Any, *, draws: int = 2000, seed: int = 0,
    periods_per_year: float = _ANNUAL,
) -> SharpeEstimate:
    """Sharpe with a bootstrapped 95% interval.

    Resampled with replacement rather than derived from a formula, because the
    closed forms assume normal returns and trading returns are not: they are
    skewed and fat-tailed, which is precisely the regime where an analytic
    interval is too narrow and too reassuring.
    """
    data = np.asarray(getattr(returns, "values", returns), dtype=float).ravel()
    data = data[np.isfinite(data)]
    n = data.size
    if n < 3:
        return SharpeEstimate(0.0, 0.0, 0.0, n)

    scale = np.sqrt(periods_per_year)

    def _sharpe(sample: np.ndarray) -> float:
        sd = float(np.std(sample, ddof=1))
        return float(np.mean(sample) / sd * scale) if sd > 0 else 0.0

    point = _sharpe(data)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(draws, n))
    samples = data[idx]
    sds = np.std(samples, axis=1, ddof=1)
    means = np.mean(samples, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        stats = np.where(sds > 0, means / sds * scale, 0.0)
    stats = stats[np.isfinite(stats)]
    if stats.size == 0:
        return SharpeEstimate(point, point, point, n)

    low, high = np.percentile(stats, [2.5, 97.5])
    return SharpeEstimate(point, float(low), float(high), n)


def cost_breakeven_bps(returns: Any, trades_per_day: float) -> float:
    """The round-trip cost at which this edge disappears.

    Reported because "profitable at 5bps" and "profitable at 50bps" are
    completely different claims, and a backtest states neither unless asked.
    A strategy whose edge dies just above the cost it was tested at has no
    margin for a widening spread.
    """
    data = np.asarray(getattr(returns, "values", returns), dtype=float).ravel()
    data = data[np.isfinite(data)]
    if data.size == 0 or trades_per_day <= 0:
        return 0.0
    mean_daily = float(np.mean(data))
    if mean_daily <= 0:
        return 0.0
    return float(mean_daily / trades_per_day * 10_000.0)


@dataclass(frozen=True, slots=True)
class Attribution:
    """Where the money came from, split every way that can mislead."""

    by_strategy: dict[str, float]
    by_symbol: dict[str, float]
    by_regime: dict[str, float]
    total: float

    @property
    def carried_by_one(self) -> str | None:
        """The strategy holding up a book whose others are losing.

        A blended number hides this completely, and it is the thing worth
        knowing before the one that is working turns.
        """
        if len(self.by_strategy) < 2 or self.total <= 0:
            return None
        winners = {k: v for k, v in self.by_strategy.items() if v > 0}
        if len(winners) != 1:
            return None
        name, value = next(iter(winners.items()))
        return name if value >= self.total else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "by_strategy": self.by_strategy, "by_symbol": self.by_symbol,
            "by_regime": self.by_regime, "total": self.total,
            "carried_by_one": self.carried_by_one,
        }


@dataclass(frozen=True, slots=True)
class DivergenceRule:
    """When a live record has diverged far enough from its backtest to retire.

    Defined in advance, because after a bad month it will not be. Retirement is
    to zero weight rather than to a discussion.
    """

    window_days: int = 60
    max_live_sharpe_gap: float = 0.0
    """Live rolling Sharpe below this while the backtest claimed positive."""

    drawdown_multiple: float = 1.0
    """Retire when live drawdown exceeds the backtest maximum by this factor."""

    def verdict(
        self, *, live_returns: Any, backtest_sharpe: float,
        backtest_max_drawdown: float,
    ) -> tuple[bool, str]:
        """Whether to retire, and the sentence explaining it."""
        data = np.asarray(getattr(live_returns, "values", live_returns), dtype=float)
        data = data[np.isfinite(data)][-self.window_days:]
        if data.size < 20:
            return False, (
                f"{data.size} live days is too short to judge divergence; "
                f"{self.window_days} is the window"
            )

        estimate = bootstrap_sharpe(data)
        if backtest_sharpe > 0 and estimate.sharpe <= self.max_live_sharpe_gap:
            return True, (
                f"retired: live Sharpe {estimate.sharpe:.2f} over "
                f"{data.size} days against a backtest that claimed "
                f"{backtest_sharpe:.2f}"
            )

        equity = np.cumprod(1.0 + data)
        drawdown = float(1.0 - (equity / np.maximum.accumulate(equity)).min())
        limit = backtest_max_drawdown * self.drawdown_multiple
        if backtest_max_drawdown > 0 and drawdown > limit:
            return True, (
                f"retired: live drawdown {drawdown:.1%} exceeds the backtest "
                f"maximum of {backtest_max_drawdown:.1%}"
            )

        return False, (
            f"live Sharpe {estimate.sharpe:.2f}, drawdown {drawdown:.1%} — "
            f"within what the backtest claimed"
        )


@dataclass
class ForwardRecord:
    """The persisted track record.

    Appended to disk on every entry rather than written at shutdown, because
    the failure that destroys a forward record is the one that stops the
    process from shutting down cleanly.
    """

    path: Path
    entries: list[TrackEntry] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.load()

    def load(self) -> int:
        """Read what is on disk. A corrupt line is skipped, never fatal.

        Months of accumulation must not be lost to one bad line -- a partial
        write from a hard kill is the likeliest cause, and it will be the last
        line, not the useful ones.
        """
        self.entries = []
        if not self.path.exists():
            return 0
        skipped = 0
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                self.entries.append(TrackEntry.from_dict(json.loads(line)))
            except (ValueError, KeyError, TypeError):
                skipped += 1
        if skipped:
            logger.warning(
                "skipped %d unreadable line(s) in the forward record; the rest "
                "loaded", skipped,
            )
        return len(self.entries)

    def append(self, entry: TrackEntry) -> None:
        """Record one day, on disk immediately."""
        self.entries.append(entry)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry.to_dict()) + "\n")

    def record_day(
        self, *, strategy: str, symbol: str, pnl: float, ret: float,
        regime: str = "indeterminate", mode: str = "paper",
        day: date | None = None,
    ) -> TrackEntry:
        entry = TrackEntry(
            day=day or datetime.now(UTC).date(), strategy=strategy,
            symbol=symbol, pnl=pnl, ret=ret, regime=regime, mode=mode,
        )
        self.append(entry)
        return entry

    # -- reading it back ---------------------------------------------------

    def _select(self, strategy: str | None, mode: str | None) -> list[TrackEntry]:
        rows = self.entries
        if strategy is not None:
            rows = [e for e in rows if e.strategy == strategy]
        if mode is not None:
            rows = [e for e in rows if e.mode == mode]
        return rows

    def days(self, strategy: str | None = None, mode: str | None = None) -> int:
        """Distinct days of record. What the allocator's ramp reads.

        Distinct days, not entries: a strategy trading four symbols does not
        have four times the evidence.
        """
        return len({e.day for e in self._select(strategy, mode)})

    def returns(
        self, strategy: str | None = None, mode: str | None = None
    ) -> np.ndarray:
        """Daily returns, summed across symbols within each day."""
        totals: dict[date, float] = defaultdict(float)
        for entry in self._select(strategy, mode):
            totals[entry.day] += entry.ret
        return np.array([totals[d] for d in sorted(totals)], dtype=float)

    def sharpe(
        self, strategy: str | None = None, mode: str | None = None
    ) -> SharpeEstimate:
        return bootstrap_sharpe(self.returns(strategy, mode))

    def attribution(self, day: date | None = None) -> Attribution:
        """Where the P&L came from, for one day or for everything."""
        rows = [e for e in self.entries if day is None or e.day == day]
        by_strategy: dict[str, float] = defaultdict(float)
        by_symbol: dict[str, float] = defaultdict(float)
        by_regime: dict[str, float] = defaultdict(float)
        for entry in rows:
            by_strategy[entry.strategy] += entry.pnl
            by_symbol[entry.symbol] += entry.pnl
            by_regime[entry.regime] += entry.pnl
        return Attribution(
            by_strategy=dict(by_strategy), by_symbol=dict(by_symbol),
            by_regime=dict(by_regime),
            total=float(sum(e.pnl for e in rows)),
        )

    def regime_breakdown(self, strategy: str | None = None) -> dict[str, SharpeEstimate]:
        """Performance split by regime.

        A strategy that makes everything in bull markets and gives it back
        sideways is a leveraged long in a costume, and the blended number hides
        that completely.
        """
        buckets: dict[str, dict[date, float]] = defaultdict(lambda: defaultdict(float))
        for entry in self._select(strategy, None):
            buckets[entry.regime][entry.day] += entry.ret
        return {
            regime: bootstrap_sharpe(
                np.array([days[d] for d in sorted(days)], dtype=float)
            )
            for regime, days in buckets.items()
        }

    def summary(self, strategy: str | None = None) -> dict[str, Any]:
        """Out-of-sample first. Yield is a footnote.

        This ordering is the opposite of what a backtest engine usually shows,
        and it is the whole design.
        """
        estimate = self.sharpe(strategy)
        returns = self.returns(strategy)
        total = float(np.sum(returns)) if returns.size else 0.0
        return {
            "out_of_sample": estimate.to_dict(),
            "days": self.days(strategy),
            "regimes": {k: v.to_dict() for k, v in
                        self.regime_breakdown(strategy).items()},
            "attribution": self.attribution().to_dict(),
            "total_return": total,
            "path": str(self.path),
        }
