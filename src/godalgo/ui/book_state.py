"""The allocator, actually deciding something.

``portfolio/book.py`` was a tested library that nothing consulted, which is the
same position the mode switch was in before it was wired: every part correct
and none of them connected. This is the connection.

What it does on each cycle:

* builds a candidate from every watched symbol, with volatility measured from
  the price history the feed already holds;
* reads the length of each strategy's forward record, so allocation is a
  function of out-of-sample track length rather than of the backtest that got
  it adopted;
* applies the drawdown ladder against the terminal's own peak equity;
* checks the resulting target against settlement and the day-trade budget,
  because a plan needing six round trips this week is impossible when it is
  written, not when its fourth order is refused;
* **caps the engine's target weight** at what the book permits for the symbol
  it trades.

That last point is the one that makes this real rather than decorative. The
cap can only ever reduce: an allocator that could enlarge a position the
strategy did not ask for would be taking a view of its own.

Everything it decides is published with the binding constraint named, because
an allocation nobody can attribute to a rule is one nobody will override when
it is wrong.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from godalgo.portfolio.book import BookLimits, BookResult, Candidate, allocate
from godalgo.risk.settlement import AccountState, plan_rebalance
from godalgo.venues.alpaca import is_crypto

if TYPE_CHECKING:
    from godalgo.research.forward import ForwardRecord

__all__ = ["BookManager"]

logger = logging.getLogger(__name__)

_MIN_HISTORY = 20
"""Observations needed before a volatility estimate is worth using.

Below this the estimate is noise, and inverse volatility on a noisy volatility
is just a random weighting with a respectable name. Such symbols get the
median volatility of the rest rather than an invented one.
"""

_DEFAULT_VOL = 0.35
"""Used only when nothing at all can be measured. Deliberately high: an
unknown instrument should be sized small, not large."""


@dataclass
class BookManager:
    """Keeps a current allocation, and caps what the engine may hold.

    Args:
        limits: Hard bounds. Never widened at runtime.
        forward: The persisted track record, for the sizing ramp.
    """

    limits: BookLimits = field(default_factory=BookLimits)
    forward: ForwardRecord | None = None
    interval_seconds: float = 5.0
    """Seconds between observations, matching the market feed's poll cadence.

    Explicit because the annualised volatility scales with its square root, so
    a wrong value here is not a small error: assuming minute bars against a
    five-second feed overstates volatility by more than three times.
    """

    result: BookResult | None = None
    plan: Any = None
    _history: dict[str, list[float]] = field(default_factory=dict, repr=False)
    _max_history: int = 400

    def observe(self, prices: dict[str, float]) -> None:
        """Fold a price tick into the volatility history.

        Kept here rather than recomputed from the venue on every cycle: the
        feed is already polling these, and asking again would be a round trip
        per symbol for numbers we have.
        """
        for symbol, price in prices.items():
            if price and price > 0:
                series = self._history.setdefault(symbol, [])
                series.append(float(price))
                if len(series) > self._max_history:
                    del series[: len(series) - self._max_history]

    def volatility(self, symbol: str) -> float | None:
        """Annualised volatility from observed prices, or None if too short.

        The annualisation is only as good as ``interval_seconds``, which is why
        that is a parameter rather than a constant: the first version assumed
        minute bars while the feed polls every five seconds, which overstated
        volatility twelvefold, drove the vol-target scalar to 0.02, and
        collapsed the whole book to cash. Nothing raised; the terminal simply
        held no positions and looked like it had decided not to.

        Crypto and equities get different year lengths on purpose. The same
        per-tick variation annualises very differently for something trading
        continuously than for something trading six and a half hours a day,
        and using one figure for both would systematically overweight
        whichever it flattered.
        """
        series = self._history.get(symbol) or []
        if len(series) < _MIN_HISTORY:
            return None
        returns = np.diff(np.log(np.asarray(series, dtype=float)))
        returns = returns[np.isfinite(returns)]
        if returns.size < _MIN_HISTORY - 1:
            return None

        seconds_per_year = (
            365.0 * 24 * 3600 if is_crypto(symbol) else 252.0 * 6.5 * 3600
        )
        periods = seconds_per_year / max(self.interval_seconds, 1e-6)
        return float(np.std(returns, ddof=1) * np.sqrt(periods))

    def candidates(
        self, symbols: list[str], *, convictions: dict[str, float] | None = None,
    ) -> list[Candidate]:
        """Build the candidate set from what is actually known."""
        convictions = convictions or {}
        measured = [v for v in (self.volatility(s) for s in symbols) if v]
        fallback = float(np.median(measured)) if measured else _DEFAULT_VOL

        days = self.forward.days() if self.forward is not None else 0
        return [
            Candidate(
                symbol=symbol,
                volatility=self.volatility(symbol) or fallback,
                sector="crypto" if is_crypto(symbol) else "us_equity",
                forward_days=days,
                conviction=convictions.get(symbol, 1.0),
            )
            for symbol in symbols
        ]

    def rebuild(
        self,
        symbols: list[str],
        *,
        equity: float,
        peak_equity: float,
        current: dict[str, float] | None = None,
        account: AccountState | None = None,
        convictions: dict[str, float] | None = None,
    ) -> BookResult:
        """Recompute the allocation. Never raises: this runs on a display path."""
        drawdown = 0.0
        if peak_equity > 0:
            drawdown = max(0.0, 1.0 - equity / peak_equity)

        try:
            self.result = allocate(
                self.candidates(symbols, convictions=convictions),
                limits=self.limits, equity=equity, drawdown=drawdown,
                current=current,
            )
        except Exception:  # noqa: BLE001 - an allocation must not kill the loop
            logger.exception("could not rebuild the book")
            return self.result or allocate([])

        if account is not None:
            targets = {a.symbol: a.weight for a in self.result.allocations}
            self.plan = plan_rebalance(
                targets, current or {}, account,
                crypto={s for s in targets if is_crypto(s)},
            )
        return self.result

    def permitted_weight(self, symbol: str, requested: float) -> tuple[float, str]:
        """What the book allows for this symbol, and why.

        Only ever reduces. Returns the requested weight untouched when no
        allocation has been computed yet -- a book that has not run must not
        silently flatten the engine.
        """
        if self.result is None:
            return requested, "no allocation computed yet; unconstrained"

        allowed = next(
            (a for a in self.result.allocations if a.symbol == symbol), None
        )
        if allowed is None:
            return 0.0, (
                f"{symbol} is not in the book — every candidate was cut by a "
                f"constraint, so the permitted weight is zero"
            )

        cap = abs(allowed.weight)
        if abs(requested) <= cap:
            return requested, f"within the book's {cap:.1%} for {symbol}"
        return float(np.sign(requested) * cap), allowed.explanation

    def to_dict(self) -> dict[str, Any]:
        """What the terminal renders. Small: this travels once a second."""
        if self.result is None:
            return {
                "state": "not_computed",
                "detail": "the book has not been built yet — this is 'not yet', "
                          "not 'nothing qualified'",
            }
        payload = self.result.to_dict()
        payload["state"] = "ready"
        if self.forward is not None:
            estimate = self.forward.sharpe()
            payload["forward"] = {
                "days": self.forward.days(),
                "sharpe": estimate.to_dict(),
            }
        if self.plan is not None:
            payload["settlement"] = self.plan.to_dict()
        return payload
