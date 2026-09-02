"""Deciding whether a search result is real.

Searching many strategies and reporting the best is the most reliable way to
manufacture a beautiful backtest that loses money. With 150 no-edge strategies
on one history, the luckiest shows a Sharpe above 2 -- not because it found
anything, but because 150 draws from a zero-centred distribution have a
maximum, and the maximum is what a search reports.

``godalgo.backtest.metrics`` already corrects the *arithmetic* of that with the
deflated Sharpe ratio and PBO. This module carries the four things that
arithmetic cannot see, each of which has been observed to pass every
statistical test while being wrong:

* **Pooled trial counts.** Fifty symbols by 154 combinations is one search of
  7,700, not fifty small ones. Deflating per-symbol and reporting that number
  is the single easiest way to keep a confident-looking result: 99% per-symbol
  has been observed collapsing to 37% pooled. The gap is exactly how much of
  the confidence came from having looked at only one symbol.

* **Effective breadth.** Momentum on five correlated names is one bet held five
  times, presenting itself as five confirmations. Counted from the correlation
  of *return streams*, not of prices, because two strategies on the same
  instrument can be uncorrelated and two on different instruments can be the
  same trade.

* **Excess over buy-and-hold.** A watchlist is survivorship-biased by
  construction -- it is a list of names someone already liked. Six candidates
  have been observed surviving their holdouts with four of them trailing simply
  owning the stock. No statistical test catches that, because none is asking.

* **Holdout looks.** A holdout is only unbiased while it is untouched. Every
  look spends a little of it, so the looks are counted and reported rather than
  left to memory.

Nothing here reports a pass on its own. The verdict is a refusal by default,
and the expected outcome of a well-run search is that nothing survives.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np

__all__ = [
    "Holdout",
    "SelectionGate",
    "Trial",
    "TrialLedger",
    "Verdict",
    "buy_and_hold_sharpe",
    "effective_breadth",
    "luck_bar",
]

logger = logging.getLogger(__name__)

_ANNUAL = 252.0


@dataclass(frozen=True, slots=True)
class Trial:
    """One thing that was tried, whether or not it worked.

    Failures are kept because the *count of attempts* is the input every
    correction needs. Dropping losers understates it, which is arithmetically
    identical to overstating the winner -- and it is the natural thing to do,
    because nobody wants to keep a list of things that did not work.
    """

    strategy: str
    symbol: str
    params: dict[str, Any] = field(default_factory=dict)
    in_sample_sharpe: float = 0.0
    out_of_sample_sharpe: float | None = None
    returns: Any = None
    """The trial's return stream, for correlation. Optional and never sent to
    a UI: it is a full series per trial."""

    at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class TrialLedger:
    """Every attempt in a search, pooled.

    The pooling is the point. A search that reports "154 trials" while having
    run that over fifty symbols is understating its own multiplicity by a
    factor of fifty, and the deflation it applies is correspondingly too
    generous.
    """

    trials: list[Trial] = field(default_factory=list)

    def record(self, trial: Trial) -> None:
        self.trials.append(trial)

    @property
    def n_pooled(self) -> int:
        """Every trial across every symbol. The number that matters."""
        return len(self.trials)

    def n_for_symbol(self, symbol: str) -> int:
        return sum(1 for t in self.trials if t.symbol == symbol)

    @property
    def symbols(self) -> list[str]:
        return sorted({t.symbol for t in self.trials})

    def sharpe_variance(self) -> float:
        """Spread of trial Sharpes, which is what deflation is measured against.

        Note the honest weakness, and it is a real one: this is the spread of
        *what was tried*. A hundred and fifty variants of trend-following on
        one trending sample all agree with each other, so the spread is too
        narrow and the deflation too weak. DSR corrects multiplicity; it cannot
        correct a search that only looked in one direction. That is why the
        holdout and the buy-and-hold comparison exist below rather than the
        statistics being trusted alone.
        """
        values = [t.in_sample_sharpe for t in self.trials if np.isfinite(t.in_sample_sharpe)]
        if len(values) < 2:
            return 0.0
        return float(np.var(values, ddof=1))

    def best(self) -> Trial | None:
        finite = [t for t in self.trials if np.isfinite(t.in_sample_sharpe)]
        return max(finite, key=lambda t: t.in_sample_sharpe) if finite else None

    def summary(self) -> dict[str, Any]:
        """What to display, with the pooled and per-symbol figures side by side."""
        best = self.best()
        symbol = best.symbol if best else ""
        pooled_bar = luck_bar(self.n_pooled, self.sharpe_variance())
        symbol_bar = luck_bar(self.n_for_symbol(symbol), self.sharpe_variance())
        return {
            "trials_pooled": self.n_pooled,
            "trials_for_best_symbol": self.n_for_symbol(symbol),
            "symbols": len(self.symbols),
            "luck_bar_pooled": pooled_bar,
            "luck_bar_per_symbol": symbol_bar,
            "best_sharpe": best.in_sample_sharpe if best else None,
            "best_symbol": symbol,
        }


def luck_bar(n_trials: int, sharpe_variance: float) -> float:
    """The Sharpe a strategy with *no edge* would reach after this many tries.

    This is the number to compare against, not zero. Reporting it beside a
    result turns "Sharpe 2.25" from an achievement into a measurement against
    the right baseline -- which on a random walk with 154 combinations is
    roughly where the winner landed.
    """
    from godalgo.backtest.metrics import expected_max_sharpe

    if n_trials < 2 or sharpe_variance <= 0:
        return 0.0
    return float(expected_max_sharpe(n_trials, sharpe_variance))


def effective_breadth(streams: Any) -> float:
    """How many independent bets a set of return streams really represents.

    ``N_eff = N / (1 + (N-1) * mean_pairwise_correlation)``

    Ten holdings in correlated names is one holding with extra commission. This
    belongs beside the position count on every risk display, because the
    position count is the number that looks like diversification and this is
    the one that is.

    Args:
        streams: A 2-D array or DataFrame, one column per position or strategy.
    """
    matrix = np.asarray(getattr(streams, "values", streams), dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        return 0.0
    n = matrix.shape[1]
    if n == 1:
        return 1.0

    # Columns with no variation carry no information and would make the
    # correlation matrix undefined rather than merely uninformative.
    usable = matrix[:, np.nanstd(matrix, axis=0) > 0]
    n = usable.shape[1]
    if n == 0:
        return 0.0
    if n == 1:
        return 1.0

    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(usable, rowvar=False)
    off = corr[~np.eye(n, dtype=bool)]
    off = off[np.isfinite(off)]
    if off.size == 0:
        return float(n)

    rho = float(np.mean(off))
    # A negative average correlation would report more independent bets than
    # positions, which is not a claim worth making from an estimate this noisy.
    rho = max(0.0, rho)
    return float(n / (1.0 + (n - 1) * rho))


def buy_and_hold_sharpe(prices: Any, *, periods_per_year: float = _ANNUAL) -> float:
    """The bar every strategy on this symbol has to clear.

    Kept as a first-class comparison rather than an afterthought because a
    search that cannot see it will happily recommend something worse than
    owning the instrument, and be statistically correct in doing so.
    """
    series = np.asarray(getattr(prices, "values", prices), dtype=float).ravel()
    series = series[np.isfinite(series) & (series > 0)]
    if series.size < 3:
        return 0.0
    returns = np.diff(np.log(series))
    sd = float(np.std(returns, ddof=1))
    if sd <= 0:
        return 0.0
    return float(np.mean(returns) / sd * np.sqrt(periods_per_year))


@dataclass
class Holdout:
    """The last slice of history, and a count of how often it has been looked at.

    A holdout is unbiased exactly once. Every look spends a little of it, and
    after enough looks it is just another thing that was searched -- so the
    count is kept and reported rather than trusted to memory. It is not
    enforced as a hard limit here because a limit that silently blocks a look
    would be worse than a number that makes the cost visible.
    """

    fraction: float = 0.25
    looks: int = 0
    looked_at: list[str] = field(default_factory=list)

    def split(self, data: Any) -> tuple[Any, Any]:
        """Search set and holdout. The holdout is never returned to a search."""
        n = len(data)
        cut = int(n * (1.0 - self.fraction))
        return data[:cut], data[cut:]

    def look(self, label: str) -> None:
        self.looks += 1
        self.looked_at.append(label)
        if self.looks > 5:
            logger.warning(
                "the holdout has been looked at %d times; it is no longer a "
                "clean out-of-sample estimate", self.looks,
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "fraction": self.fraction,
            "looks": self.looks,
            "candidates": list(self.looked_at),
            "still_clean": self.looks <= 1,
        }


@dataclass(frozen=True, slots=True)
class Verdict:
    """Whether a candidate may trade, and precisely what it missed.

    Refusal is the default and the expected outcome. A high adoption rate means
    the gates are broken, not that the search is good.
    """

    adopted: bool
    reasons: tuple[str, ...] = ()
    """Every bar that was missed, not just the first. Fixing one at a time
    across separate runs is its own form of searching."""

    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def summary(self) -> str:
        if self.adopted:
            return "adopted: cleared every gate"
        return "refused: " + "; ".join(self.reasons)


@dataclass
class SelectionGate:
    """The gate into live trading. There is deliberately no override.

    A way around this check is a way to lose money with the system's blessing,
    so the thresholds are constructor arguments and there is no ``force``,
    no ``skip``, and no environment variable.
    """

    min_out_of_sample_sharpe: float = 0.5
    min_deflated_sharpe: float = 0.95
    """Evaluated against the POOLED trial count, never the per-symbol one."""

    min_excess_over_hold: float = 0.0
    """Sharpe points above buy-and-hold on the same symbol and period."""

    require_holdout: bool = True

    def evaluate(
        self,
        *,
        candidate: Trial,
        ledger: TrialLedger,
        holdout_sharpe: float | None,
        hold_sharpe: float,
        deflated: float | None = None,
    ) -> Verdict:
        """Judge one candidate against every bar at once."""
        from godalgo.backtest.metrics import deflated_sharpe_ratio

        reasons: list[str] = []
        n_pooled = ledger.n_pooled
        variance = ledger.sharpe_variance()
        bar = luck_bar(n_pooled, variance)

        if self.require_holdout and holdout_sharpe is None:
            reasons.append("never tested on the untouched holdout")

        oos = holdout_sharpe if holdout_sharpe is not None else float("-inf")
        if holdout_sharpe is not None and oos < self.min_out_of_sample_sharpe:
            reasons.append(
                f"out-of-sample Sharpe {oos:.2f} is below the "
                f"{self.min_out_of_sample_sharpe:.2f} floor"
            )

        if deflated is None and candidate.returns is not None:
            deflated = float(deflated_sharpe_ratio(
                candidate.returns, n_trials=n_pooled, sharpe_variance=variance,
            ))
        if deflated is not None and deflated < self.min_deflated_sharpe:
            reasons.append(
                f"pooled deflated Sharpe {deflated:.2%} is below the "
                f"{self.min_deflated_sharpe:.0%} floor across {n_pooled} trials"
            )

        excess = oos - hold_sharpe if holdout_sharpe is not None else float("-inf")
        if holdout_sharpe is not None and excess <= self.min_excess_over_hold:
            reasons.append(
                f"does not beat buy-and-hold on {candidate.symbol}: "
                f"{oos:.2f} against {hold_sharpe:.2f}"
            )

        if candidate.in_sample_sharpe <= bar:
            reasons.append(
                f"in-sample Sharpe {candidate.in_sample_sharpe:.2f} is inside "
                f"what {n_pooled} trials of a no-edge strategy would reach "
                f"({bar:.2f})"
            )

        return Verdict(
            adopted=not reasons,
            reasons=tuple(reasons),
            metrics={
                "trials_pooled": n_pooled,
                "luck_bar_pooled": bar,
                "out_of_sample_sharpe": holdout_sharpe,
                "buy_and_hold_sharpe": hold_sharpe,
                "excess_over_hold": excess if np.isfinite(excess) else None,
                "deflated_sharpe": deflated,
            },
        )
