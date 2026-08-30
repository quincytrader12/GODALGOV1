"""Can this strategy trade at this frequency at all?

The question has an arithmetic answer, and it is worth computing before running
anything, because the answer is often no.

**Volatility scales with the square root of time. Costs do not scale at all.**

Expected move over a holding period is ``sigma_annual * sqrt(T_bar * hold /
T_year)``, while a round trip costs a fixed number of basis points -- the spread
plus two fees -- no matter how long the bar is. Shorten the bar and the edge
shrinks as sqrt(T) while the cost stays put. Below some bar length no signal,
however good, pays for itself.

That floor is the single most useful number to know about a trading system, and
it is why "make it trade faster" is usually a request to lose money faster. The
honest response to a bot that will not trade is to establish whether the
frequency is viable, not to lower the gate until orders appear.

Two levers move the floor, and only two:

* **Hold longer.** Edge grows as sqrt(hold), so a 25-bar hold clears a bar
  roughly 8x shorter than a 3-bar hold does.
* **Stop crossing the spread.** Maker and taker round trips differ by ~4x at
  typical fees, which is ~16x in bar length.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from godalgo.execution.broker import FeeSchedule

if TYPE_CHECKING:
    from godalgo.backtest.engine import BacktestResult
    from godalgo.strategies.base import Strategy

__all__ = [
    "FeasibilityReport",
    "assess_frequency",
    "assess_from_backtest",
    "minimum_viable_bar_seconds",
]

_SECONDS_PER_YEAR = 365.0 * 24.0 * 3600.0


def _round_trip_cost_bps(fees: FeeSchedule, spread_bps: float, *, maker: bool) -> float:
    """Cost of entering and exiting once, in bps.

    A maker round trip pays two maker fees and does not cross; a taker round
    trip pays two taker fees and crosses the spread twice.
    """
    if maker:
        return 2.0 * fees.maker * 1e4
    return 2.0 * fees.taker * 1e4 + 2.0 * spread_bps


def expected_edge_bps(
    annual_vol: float,
    bar_seconds: float,
    conviction: float,
    holding_bars: float,
) -> float:
    """Expected move over a holding period, in bps.

    Args:
        annual_vol: Annualised volatility as a fraction, e.g. 0.60 for 60%.
        bar_seconds: Bar length in seconds.
        conviction: Typical absolute signal strength in [0, 1]. Use a realistic
            average, not the maximum -- the gate is cleared on typical bars, not
            on the best one.
        holding_bars: Bars a position is held.
    """
    sigma_bar = annual_vol * np.sqrt(bar_seconds / _SECONDS_PER_YEAR)
    return float(conviction * sigma_bar * np.sqrt(max(holding_bars, 1.0)) * 1e4)


def minimum_viable_bar_seconds(
    annual_vol: float,
    conviction: float,
    holding_bars: float,
    *,
    fees: FeeSchedule | None = None,
    spread_bps: float = 2.0,
    min_edge_multiple: float = 1.5,
    maker: bool = True,
) -> float:
    """Shortest bar at which expected edge clears the cost gate.

    Solves ``conviction * sigma_annual * sqrt(T/T_year) * sqrt(hold) * 1e4 =
    multiple * cost`` for ``T``.

    Returns:
        Bar length in seconds. Below this, the configuration cannot trade
        profitably no matter how well the signal is tuned.
    """
    fees = fees or FeeSchedule()
    cost = _round_trip_cost_bps(fees, spread_bps, maker=maker)
    required = min_edge_multiple * cost

    denominator = conviction * annual_vol * np.sqrt(max(holding_bars, 1.0))
    if denominator <= 0:
        return float("inf")
    return float(_SECONDS_PER_YEAR * ((required / 1e4) / denominator) ** 2)


@dataclass(frozen=True, slots=True)
class FeasibilityReport:
    """Whether a configuration can trade, and what would make it able to."""

    bar_seconds: float
    annual_vol: float
    conviction: float
    holding_bars: float
    edge_bps: float
    maker_cost_bps: float
    taker_cost_bps: float
    min_edge_multiple: float
    min_bar_maker: float
    min_bar_taker: float

    @property
    def tradeable_as_maker(self) -> bool:
        return self.edge_bps >= self.min_edge_multiple * self.maker_cost_bps

    @property
    def tradeable_as_taker(self) -> bool:
        return self.edge_bps >= self.min_edge_multiple * self.taker_cost_bps

    @property
    def headroom(self) -> float:
        """Edge as a multiple of the maker requirement. Below 1.0 will not trade."""
        required = self.min_edge_multiple * self.maker_cost_bps
        return self.edge_bps / required if required > 0 else float("inf")

    def describe(self) -> str:
        def fmt(seconds: float) -> str:
            if seconds < 120:
                return f"{seconds:.0f}s"
            if seconds < 7200:
                return f"{seconds / 60:.1f}m"
            if seconds < 172800:
                return f"{seconds / 3600:.1f}h"
            return f"{seconds / 86400:.1f}d"

        verdict = (
            "TRADEABLE (maker)" if self.tradeable_as_maker
            else "NOT TRADEABLE at this frequency"
        )
        need_maker = self.min_edge_multiple * self.maker_cost_bps
        need_taker = self.min_edge_multiple * self.taker_cost_bps
        hold_time = fmt(self.holding_bars * self.bar_seconds)

        lines = [
            verdict,
            f"  bar length          {fmt(self.bar_seconds)}",
            f"  annual vol          {self.annual_vol:.0%}",
            f"  typical conviction  {self.conviction:.2f}",
            (f"  holding period      {self.holding_bars:.0f} bars ({hold_time})"),
            f"  expected edge       {self.edge_bps:.1f} bps",
            (f"  maker round trip    {self.maker_cost_bps:.1f} bps (need {need_maker:.1f})"),
            (f"  taker round trip    {self.taker_cost_bps:.1f} bps (need {need_taker:.1f})"),
            f"  headroom            {self.headroom:.2f}x",
            (
                f"  min viable bar      {fmt(self.min_bar_maker)} maker / "
                f"{fmt(self.min_bar_taker)} taker"
            ),
        ]
        if not self.tradeable_as_maker:
            shortfall = self.min_bar_maker / self.bar_seconds
            needed_hold = self.holding_bars * shortfall**2
            lines += [
                "",
                "  To trade at this frequency, one of:",
                f"    - use bars >= {fmt(self.min_bar_maker)} ({shortfall:.1f}x longer)",
                (
                    f"    - hold longer: edge grows as sqrt(hold), so "
                    f"{needed_hold:.0f} bars would clear it"
                ),
                "    - reduce costs (maker rebates, tighter venue)",
                (
                    "  Lowering min_edge_multiple is NOT on this list: it does "
                    "not create edge, it only stops measuring it."
                ),
            ]
        return "\n".join(lines)


def assess_frequency(
    bar_seconds: float,
    annual_vol: float,
    conviction: float,
    holding_bars: float,
    *,
    fees: FeeSchedule | None = None,
    spread_bps: float = 2.0,
    min_edge_multiple: float = 1.5,
) -> FeasibilityReport:
    """Assess whether a configuration can clear its costs.

    Args:
        bar_seconds: Bar length in seconds.
        annual_vol: Annualised volatility of the asset, as a fraction.
        conviction: Typical absolute signal strength in [0, 1].
        holding_bars: Bars a position is typically held. Take this from
            ``Strategy.expected_holding_bars`` rather than guessing -- it is the
            term the answer is most sensitive to.
        fees: Venue fee schedule.
        spread_bps: Typical spread in bps, for the taker cost.
        min_edge_multiple: Required ratio of edge to cost.
    """
    fees = fees or FeeSchedule()
    return FeasibilityReport(
        bar_seconds=bar_seconds,
        annual_vol=annual_vol,
        conviction=conviction,
        holding_bars=holding_bars,
        edge_bps=expected_edge_bps(annual_vol, bar_seconds, conviction, holding_bars),
        maker_cost_bps=_round_trip_cost_bps(fees, spread_bps, maker=True),
        taker_cost_bps=_round_trip_cost_bps(fees, spread_bps, maker=False),
        min_edge_multiple=min_edge_multiple,
        min_bar_maker=minimum_viable_bar_seconds(
            annual_vol, conviction, holding_bars, fees=fees,
            spread_bps=spread_bps, min_edge_multiple=min_edge_multiple, maker=True,
        ),
        min_bar_taker=minimum_viable_bar_seconds(
            annual_vol, conviction, holding_bars, fees=fees,
            spread_bps=spread_bps, min_edge_multiple=min_edge_multiple, maker=False,
        ),
    )


def assess_from_backtest(
    result: BacktestResult,
    bar_seconds: float,
    momentum: Strategy,
    reversion: Strategy,
    *,
    fees: FeeSchedule | None = None,
    spread_bps: float = 2.0,
    min_edge_multiple: float = 1.5,
    conviction_quantile: float = 0.90,
    warmup: int = 250,
) -> FeasibilityReport:
    """Assess feasibility using values measured from a backtest, not assumed.

    Every input to ``assess_frequency`` is a guess unless it comes from the
    system actually running: realised volatility, the conviction the strategies
    genuinely produce, and their blended holding period. Guessing conviction is
    the easiest way to get a confident wrong answer -- an assumed 0.30 against a
    realised 0.08 overstates edge nearly fourfold.

    Conviction is taken at a **high quantile rather than the mean**, and
    deliberately so. The gate is cleared on the bars the strategy actually wants
    to trade, not on the many bars where it has no view; averaging those in
    would understate what the system does when it acts.

    Args:
        result: A completed backtest over representative data.
        bar_seconds: Bar length of that backtest, in seconds.
        momentum: The momentum strategy instance used.
        reversion: The mean-reversion strategy instance used.
        fees: Venue fee schedule.
        spread_bps: Typical spread in bps.
        min_edge_multiple: Required ratio of edge to cost.
        conviction_quantile: Which quantile of absolute conviction to use.
        warmup: Leading bars to exclude, where signals are zero by construction
            and would drag the quantile down.

    Returns:
        A ``FeasibilityReport`` built entirely from observed quantities.
    """
    import numpy as np

    frame = result.frame.iloc[warmup:]
    conviction = float(frame["combined"].abs().quantile(conviction_quantile))

    returns = frame["close"].pct_change().dropna()
    bars_per_year = _SECONDS_PER_YEAR / bar_seconds
    annual_vol = float(returns.std(ddof=1) * np.sqrt(bars_per_year))

    # Blend the two holding periods by how much each strategy actually drove
    # the signal over this sample.
    mom_weight = float(frame["contrib_momentum"].abs().sum())
    rev_weight = float(frame["contrib_reversion"].abs().sum())
    total = mom_weight + rev_weight
    if total > 0:
        holding = (
            mom_weight * momentum.expected_holding_bars
            + rev_weight * reversion.expected_holding_bars
        ) / total
    else:
        holding = momentum.expected_holding_bars

    return assess_frequency(
        bar_seconds=bar_seconds,
        annual_vol=annual_vol,
        conviction=conviction,
        holding_bars=holding,
        fees=fees,
        spread_bps=spread_bps,
        min_edge_multiple=min_edge_multiple,
    )
