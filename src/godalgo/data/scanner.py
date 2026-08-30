"""Market scanner: choosing what to trade.

A single-symbol bot trades whatever it was pointed at, whether or not that
instrument currently offers anything. Scanning inverts it -- rank the universe,
trade what actually looks tradeable, and be willing to trade nothing.

Four filters, applied in this order because each is cheaper than the next and
rejects more:

1. **Liquidity.** Volume and spread. An illiquid pair fails on execution
   whatever its signal says, and a backtest on thin data will never tell you so.
2. **Volatility band.** Too quiet and no move covers the spread; too violent
   and position sizing collapses to nothing useful anyway.
3. **Feasibility.** The decisive one, and the one most scanners omit: does the
   expected edge clear the round-trip cost at this timeframe? A symbol can look
   like the strongest trend in the universe and still be untradeable because
   its moves are smaller than its spread. Ranking on signal strength alone
   systematically selects exactly those instruments.
4. **Regime clarity.** Only then, how strong is the read.

Then a diversification pass, which in crypto is not optional. BTC, ETH and SOL
longs are not three positions -- they are one position in three wrappers, and a
scanner that ranks purely on score will hand you that every time, concentrating
risk precisely when it believes it is spreading it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from godalgo.core.types import Regime
from godalgo.execution.broker import FeeSchedule
from godalgo.feasibility import assess_frequency
from godalgo.features.regime import classify_regime

__all__ = ["Candidate", "MarketScanner", "ScanCriteria", "ScanResult"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ScanCriteria:
    """Filters and weights for universe selection."""

    quote_currency: str = "USDT"
    min_volume_24h: float = 5_000_000.0
    """Minimum 24h quote volume. Thin books fail on execution regardless of
    how good the signal looks."""

    max_spread_bps: float = 15.0
    min_annual_vol: float = 0.25
    """Below this, moves are too small to clear costs at any reasonable size."""

    max_annual_vol: float = 3.0
    """Above this, volatility targeting sizes the position down to nothing and
    stop distances become impractical."""

    min_headroom: float = 1.0
    """Required ratio of expected edge to the cost gate. Below 1.0 the symbol
    cannot pay for its own execution."""

    min_confidence: float = 0.15
    """Minimum regime confidence. An indeterminate read is not an opportunity."""

    max_candidates: int = 6
    max_correlation: float = 0.85
    """Ceiling on correlation with an already-selected symbol.

    Crypto majors routinely run above 0.9 against each other; without this the
    scanner returns the same trade several times over.
    """

    correlation_window: int = 500
    min_bars: int = 400
    conviction_assumption: float = 0.25
    """Conviction used for the feasibility estimate.

    Deliberately below the p90 the strategies actually produce: a scanner that
    assumes its best case will admit symbols that only work on their best bars.
    """

    def __post_init__(self) -> None:
        if self.min_annual_vol >= self.max_annual_vol:
            raise ValueError("min_annual_vol must be below max_annual_vol")
        if not 0 < self.max_correlation <= 1:
            raise ValueError("max_correlation must be in (0, 1]")
        if self.max_candidates < 1:
            raise ValueError("max_candidates must be at least 1")


@dataclass(frozen=True, slots=True)
class Candidate:
    """One scanned instrument."""

    symbol: str
    score: float
    regime: Regime
    confidence: float
    annual_vol: float
    spread_bps: float
    volume_24h: float
    headroom: float
    """Expected edge as a multiple of the cost gate. Below 1.0 is untradeable."""

    hurst: float
    half_life: float | None
    rejected: str | None = None
    """Why it was excluded, or None if it survived."""

    @property
    def tradeable(self) -> bool:
        return self.rejected is None

    def to_dict(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "score": round(self.score, 4),
            "regime": self.regime.value,
            "confidence": round(self.confidence, 3),
            "annual_vol": round(self.annual_vol, 4),
            "spread_bps": round(self.spread_bps, 2),
            "volume_24h": round(self.volume_24h, 0),
            "headroom": round(self.headroom, 2),
            "hurst": round(self.hurst, 3),
            "half_life": round(self.half_life, 1) if self.half_life else None,
            "rejected": self.rejected,
        }


@dataclass(frozen=True, slots=True)
class ScanResult:
    """A completed scan."""

    timestamp: datetime
    selected: tuple[Candidate, ...]
    rejected: tuple[Candidate, ...]
    scanned: int

    def summary(self) -> str:
        lines = [f"scanned {self.scanned}, selected {len(self.selected)}"]
        for c in self.selected:
            lines.append(
                f"  {c.symbol:<14s} {c.regime.value:<15s} conf {c.confidence:.2f}  "
                f"vol {c.annual_vol:.0%}  headroom {c.headroom:.1f}x  score {c.score:.3f}"
            )
        reasons: dict[str, int] = {}
        for c in self.rejected:
            key = (c.rejected or "unknown").split("(")[0].strip()
            reasons[key] = reasons.get(key, 0) + 1
        if reasons:
            lines.append("  rejected: " + ", ".join(
                f"{k} x{v}" for k, v in sorted(reasons.items(), key=lambda kv: -kv[1])
            ))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "scanned": self.scanned,
            "selected": [c.to_dict() for c in self.selected],
            "rejected": [c.to_dict() for c in self.rejected],
        }


@dataclass
class MarketScanner:
    """Ranks a universe and returns a diversified, tradeable shortlist."""

    criteria: ScanCriteria = field(default_factory=ScanCriteria)
    fees: FeeSchedule = field(default_factory=FeeSchedule)
    bars_per_day: float = 24.0
    holding_bars: float = 28.0
    """Expected holding period, from the strategies' own estimate. Feasibility
    is most sensitive to this term, so it must not be guessed."""

    def scan(
        self,
        history: dict[str, pd.DataFrame],
        tickers: dict[str, dict[str, float]] | None = None,
    ) -> ScanResult:
        """Rank a universe.

        Args:
            history: Bar history per symbol, oldest first.
            tickers: Optional per-symbol ``{"quoteVolume", "bid", "ask"}``. When
                absent, liquidity filters are skipped and only the statistical
                ones apply -- reported honestly rather than silently passing
                every symbol.

        Returns:
            A ``ScanResult``. ``selected`` is ordered best first and already
            diversified.
        """
        tickers = tickers or {}
        assessed: list[Candidate] = []

        for symbol, bars in history.items():
            candidate = self._assess(symbol, bars, tickers.get(symbol, {}))
            if candidate is not None:
                assessed.append(candidate)

        survivors = sorted(
            (c for c in assessed if c.tradeable), key=lambda c: -c.score
        )
        rejected = [c for c in assessed if not c.tradeable]

        selected = self._diversify(survivors, history)
        # Anything dropped for correlation is a rejection, and saying so is the
        # difference between "nothing qualified" and "everything qualified but
        # was the same trade".
        for candidate in survivors:
            if candidate not in selected:
                rejected.append(replace(candidate, rejected="correlated with a selection"))

        return ScanResult(
            timestamp=datetime.now(UTC),
            selected=tuple(selected),
            rejected=tuple(rejected),
            scanned=len(history),
        )

    # -- assessment --------------------------------------------------------

    def _assess(
        self, symbol: str, bars: pd.DataFrame, ticker: dict[str, float]
    ) -> Candidate | None:
        if bars is None or "close" not in bars or len(bars) < self.criteria.min_bars:
            return None

        close = bars["close"].astype(float)
        returns = np.log(close).diff().dropna()
        if returns.empty:
            return None

        annual_vol = float(returns.std(ddof=1) * np.sqrt(self.bars_per_day * 365.0))

        volume = float(ticker.get("quoteVolume") or 0.0)
        bid, ask = float(ticker.get("bid") or 0.0), float(ticker.get("ask") or 0.0)
        spread_bps = (
            1e4 * (ask - bid) / ((ask + bid) / 2) if bid > 0 and ask > bid else 0.0
        )

        state = classify_regime(close.tail(self.criteria.correlation_window), symbol)

        report = assess_frequency(
            bar_seconds=86_400.0 / self.bars_per_day,
            annual_vol=annual_vol,
            conviction=self.criteria.conviction_assumption,
            holding_bars=self.holding_bars,
            fees=self.fees,
            spread_bps=spread_bps or 2.0,
        )

        def reject(reason: str) -> Candidate:
            return self._build(symbol, 0.0, state, annual_vol, spread_bps, volume,
                               report.headroom, reason)

        # Cheapest and most decisive filters first.
        if ticker and volume < self.criteria.min_volume_24h:
            return reject(f"illiquid (24h volume {volume:,.0f})")
        if spread_bps > self.criteria.max_spread_bps:
            return reject(f"spread too wide ({spread_bps:.1f}bps)")
        if annual_vol < self.criteria.min_annual_vol:
            return reject(f"too quiet ({annual_vol:.0%} annual vol)")
        if annual_vol > self.criteria.max_annual_vol:
            return reject(f"too volatile ({annual_vol:.0%} annual vol)")
        if report.headroom < self.criteria.min_headroom:
            return reject(f"edge below costs (headroom {report.headroom:.2f}x)")
        if state.regime is Regime.INDETERMINATE:
            return reject("no regime")
        if state.confidence < self.criteria.min_confidence:
            return reject(f"weak regime (confidence {state.confidence:.2f})")

        return self._build(
            symbol,
            self._score(state.confidence, report.headroom, volume),
            state, annual_vol, spread_bps, volume, report.headroom, None,
        )

    @staticmethod
    def _build(symbol, score, state, annual_vol, spread_bps, volume, headroom, rejected):
        return Candidate(
            symbol=symbol, score=score, regime=state.regime,
            confidence=state.confidence, annual_vol=annual_vol,
            spread_bps=spread_bps, volume_24h=volume, headroom=headroom,
            hurst=state.hurst, half_life=state.half_life, rejected=rejected,
        )

    @staticmethod
    def _score(confidence: float, headroom: float, volume: float) -> float:
        """Combine regime clarity, economic headroom, and liquidity.

        Headroom is capped before it enters the score. Beyond a few multiples
        of cost, more headroom stops being an edge and starts being volatility
        -- which is risk, not opportunity, and ranking on it unbounded would
        steer the scanner toward the most violent instrument available.
        """
        clarity = min(1.0, confidence)
        economics = min(1.0, headroom / 4.0)
        liquidity = min(1.0, np.log10(max(volume, 1.0)) / 9.0) if volume > 0 else 0.5
        return float(0.45 * clarity + 0.40 * economics + 0.15 * liquidity)

    # -- diversification ---------------------------------------------------

    def _diversify(
        self, ranked: list[Candidate], history: dict[str, pd.DataFrame]
    ) -> list[Candidate]:
        """Greedily take the best candidate not already represented.

        Crypto majors move together, so a shortlist ranked purely on score is
        usually one bet held several times. Each pick must be sufficiently
        uncorrelated with everything already chosen.
        """
        selected: list[Candidate] = []
        returns_cache: dict[str, pd.Series] = {}

        def returns_for(symbol: str) -> pd.Series | None:
            if symbol not in returns_cache:
                bars = history.get(symbol)
                if bars is None or "close" not in bars:
                    return None
                series = np.log(bars["close"].astype(float)).diff().dropna()
                returns_cache[symbol] = series.tail(self.criteria.correlation_window)
            return returns_cache[symbol]

        for candidate in ranked:
            if len(selected) >= self.criteria.max_candidates:
                break

            mine = returns_for(candidate.symbol)
            clash = False
            for chosen in selected:
                theirs = returns_for(chosen.symbol)
                if mine is None or theirs is None:
                    continue
                rho = _correlation(mine, theirs)
                if rho is not None and abs(rho) > self.criteria.max_correlation:
                    logger.debug(
                        "%s dropped: correlation %.2f with %s",
                        candidate.symbol, rho, chosen.symbol,
                    )
                    clash = True
                    break
            if not clash:
                selected.append(candidate)

        return selected


def _correlation(a: pd.Series, b: pd.Series) -> float | None:
    """Pearson correlation on the overlapping index.

    Aligned on the index rather than positionally: two series with different
    histories would otherwise be compared across mismatched timestamps, which
    produces a number that looks like a correlation and is not one.
    """
    joined = pd.concat([a, b], axis=1, join="inner").dropna()
    if len(joined) < 30:
        return None
    values = joined.to_numpy()
    if values[:, 0].std() == 0 or values[:, 1].std() == 0:
        return None
    return float(np.corrcoef(values[:, 0], values[:, 1])[0, 1])
