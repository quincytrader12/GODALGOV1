"""How much of the book each position gets.

Everything in ``godalgo.research.selection`` decides *whether* a strategy is
real. This decides *how much*, which is where most of the realised risk
actually comes from. A portfolio of good strategies sized badly loses money; a
portfolio of mediocre strategies sized well often does not.

**The governing fact: estimation error dominates.** Mean-variance optimisers
are error maximisers -- they place the largest weights exactly where the inputs
are most overstated, because an overstated expected return and an understated
covariance both attract capital. DeMiguel, Garlappi & Uppal (2009) found naive
1/N beat sample-based mean-variance out of sample across essentially every
dataset they tested, and estimated a 25-asset portfolio would need roughly
3,000 months of data before optimisation reliably won. Nobody has 250 years.

So the allocator is built to depend on the inputs it knows best. Ranked by how
badly they are known: expected returns (terrible), covariances (poor),
volatilities (acceptable), recent correlations (acceptable). Inverse volatility
needs only the third. That ordering, and not any cleverness, is why
inverse-volatility and risk-parity schemes survive out of sample while
return-forecast optimisation usually does not.

The output is meant to be **boring**. Most weights should sit at a constraint
rather than at an optimum. Interesting, concentrated, high-conviction weights
are what fitting estimation error looks like from the inside.

Every allocation is explainable in one sentence naming the binding constraint
-- "AAPL at 4.2%, capped by the 1% ADV limit, not by the signal" -- because an
allocation nobody can attribute to a rule is one nobody will override when it
is wrong, and it will be wrong.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "Allocation",
    "BookLimits",
    "BookResult",
    "Candidate",
    "allocate",
    "drawdown_multiplier",
    "no_trade_band",
    "shrunk_covariance",
    "stress_gross",
]

logger = logging.getLogger(__name__)

_ANNUAL = 252.0


@dataclass(frozen=True, slots=True)
class BookLimits:
    """Hard bounds. Limits, not preferences -- none of these is advisory.

    Like ``RiskLimits``, none of this is in any strategy's search space. An
    allocator that could widen its own position cap does not have one.
    """

    max_position_weight: float = 0.08
    """Ceiling per position. 5-10% for a retail book."""

    max_sector_weight: float = 0.25
    max_gross: float = 1.0
    """1.0 is unlevered. Stated explicitly because a gross cap above 1 is a
    decision to borrow, and it should never be reached by accident."""

    max_net: float = 1.0
    max_adv_share: float = 0.01
    """Share of average daily volume. You cannot exit what you cannot trade,
    and the position that is too big to exit is the one you most want out of."""

    min_position_weight: float = 0.005
    """Below the point where the spread exceeds the expected edge the correct
    size is zero, not small. A dust position pays costs to express nothing."""

    target_volatility: float = 0.12
    """Annualised. 10-15% is sane for an equity book."""

    max_leverage: float = 1.0
    """Cap on the vol-targeting scalar. A vol-targeting rule with no cap levers
    enormously into a quiet market, which is immediately before it stops being
    one. This is the single most dangerous omission in the whole module."""

    new_strategy_fraction: float = 0.25
    """New entrants start here and scale on forward evidence, never on the
    backtest that got them adopted."""

    forward_days_for_full_size: int = 60
    band_relative: float = 0.20
    """No-trade band, as a fraction of target weight."""

    band_absolute: float = 0.01
    drawdown_ladder: tuple[tuple[float, float], ...] = (
        (0.05, 0.75), (0.10, 0.50), (0.15, 0.0),
    )
    """(peak-to-trough drawdown, permitted fraction of gross). Decided in
    advance and encoded, because a drawdown is exactly when discretion is
    worst."""


@dataclass(frozen=True, slots=True)
class Candidate:
    """One instrument the book may hold, and what is known about it."""

    symbol: str
    volatility: float
    """Annualised. The one input this allocator trusts."""

    sector: str = "unknown"
    adv_usd: float = 0.0
    """Average daily dollar volume, in dollars.

    Zero means unknown, and unknown is treated as unconstrained rather than as
    zero: refusing to size anything without volume data would silently empty
    the book, which is a worse failure than an uncapped position.
    """

    forward_days: int = 0
    """Length of the out-of-sample forward record. Allocation is a function of
    this, not of in-sample Sharpe."""

    conviction: float = 1.0
    """Sign and strength from the strategy, in [-1, 1]. Used for direction and
    to zero a position, never to enlarge one beyond its risk budget."""


@dataclass(frozen=True, slots=True)
class Allocation:
    """One line of the book, and why it is that size."""

    symbol: str
    weight: float
    binding: str
    """Which rule set this weight. ``"inverse_vol"`` means nothing bound it."""

    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol, "weight": self.weight,
            "binding": self.binding, "explanation": self.explanation,
        }


@dataclass(frozen=True, slots=True)
class BookResult:
    """The whole allocation, with the numbers that say whether it is sane."""

    allocations: tuple[Allocation, ...]
    cash_weight: float
    gross: float
    net: float
    effective_breadth: float
    """Independent bets, not positions. If this is under 3 on a ten-name book,
    the book is a concentrated bet that looks diversified."""

    vol_scalar: float = 1.0
    drawdown_scalar: float = 1.0
    stressed_loss: float = 0.0
    """Loss if every correlation goes to 0.8, which is what happens in the
    conditions the diversification was for."""

    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "allocations": [a.to_dict() for a in self.allocations],
            "cash_weight": self.cash_weight,
            "gross": self.gross,
            "net": self.net,
            "effective_breadth": self.effective_breadth,
            "positions": len(self.allocations),
            "vol_scalar": self.vol_scalar,
            "drawdown_scalar": self.drawdown_scalar,
            "stressed_loss": self.stressed_loss,
            "notes": list(self.notes),
        }


def shrunk_covariance(returns: Any, shrinkage: float | None = None) -> np.ndarray:
    """Covariance shrunk toward a constant-correlation target.

    Sample covariance is unusable when the number of assets approaches the
    number of observations: it is near-singular, and any optimiser fed it will
    happily lever into the noise directions, which are precisely the directions
    the estimate is least sure about.

    Shrinks toward the Ledoit-Wolf constant-correlation target -- each pair's
    correlation replaced by the average -- with the intensity estimated from
    the dispersion of the sample correlations rather than assumed. This is a
    practical form of the estimator, not the full analytic one; where the
    intensity cannot be estimated it falls back to heavy shrinkage, because
    being wrong toward the target is much cheaper than being wrong toward the
    sample.
    """
    data = np.asarray(getattr(returns, "values", returns), dtype=float)
    if data.ndim != 2 or data.shape[1] == 0:
        return np.zeros((0, 0))

    n_obs, n_assets = data.shape
    sample = np.cov(data, rowvar=False, ddof=1)
    sample = np.atleast_2d(sample)
    if n_assets == 1:
        return sample

    sd = np.sqrt(np.clip(np.diag(sample), 1e-18, None))
    corr = sample / np.outer(sd, sd)
    off = corr[~np.eye(n_assets, dtype=bool)]
    mean_corr = float(np.mean(off)) if off.size else 0.0

    target_corr = np.full((n_assets, n_assets), mean_corr)
    np.fill_diagonal(target_corr, 1.0)
    target = target_corr * np.outer(sd, sd)

    if shrinkage is None:
        # More observations relative to assets means the sample is worth more.
        # With few observations the target dominates, which is the correct
        # response to an estimate that cannot support its own dimensionality.
        ratio = n_assets / max(n_obs, 1)
        shrinkage = float(np.clip(ratio, 0.1, 0.9))
    shrinkage = float(np.clip(shrinkage, 0.0, 1.0))
    return shrinkage * target + (1.0 - shrinkage) * sample


def drawdown_multiplier(drawdown: float, ladder: tuple[tuple[float, float], ...]) -> float:
    """Permitted fraction of gross at this drawdown.

    Args:
        drawdown: Peak-to-trough, as a positive fraction.
    """
    permitted = 1.0
    for threshold, fraction in sorted(ladder):
        if drawdown >= threshold:
            permitted = fraction
    return permitted


def stress_gross(weights: np.ndarray, vols: np.ndarray, rho: float = 0.8) -> float:
    """Portfolio volatility if every correlation goes to ``rho``.

    Every diversification estimate made in calm conditions overstates the
    protection available in the conditions it was for. This is the number to
    check survivability against, and when it is not survivable the position
    count is not the problem -- the gross is.
    """
    if weights.size == 0:
        return 0.0
    contribution = np.abs(weights) * vols
    total = float(np.sum(contribution))
    sum_squares = float(np.sum(contribution ** 2))
    variance = sum_squares + rho * (total ** 2 - sum_squares)
    return float(np.sqrt(max(variance, 0.0)))


def no_trade_band(
    target: float, current: float, limits: BookLimits
) -> tuple[float, str]:
    """Where to actually move, given a band around the target.

    Rebalancing to exact target weights on every bar burns the edge in costs.
    Inside the band, do nothing; outside it, trade back to the *band edge*
    rather than the centre -- moving to the centre pays the full distance for a
    position that will drift straight back out.
    """
    width = max(abs(target) * limits.band_relative, limits.band_absolute)
    drift = current - target
    if abs(drift) <= width:
        return current, "inside the no-trade band"
    edge = target + np.sign(drift) * width
    return float(edge), "rebalanced to the band edge"


def allocate(
    candidates: list[Candidate],
    *,
    limits: BookLimits | None = None,
    equity: float = 0.0,
    returns: Any = None,
    drawdown: float = 0.0,
    current: dict[str, float] | None = None,
) -> BookResult:
    """Turn a set of candidates into a book.

    The order is deliberate: start from the input that is best known, then let
    every constraint reduce. Nothing in here can enlarge a position -- a
    constraint that could increase a weight is not a constraint.

    Args:
        candidates: What may be held, with volatility and liquidity.
        limits: Hard bounds. Defaults are a retail book.
        equity: Account equity, in the same currency as ``adv_usd``. Without
            it the liquidity cap cannot be computed and is skipped, with a
            note -- a cap applied to the wrong units is worse than none.
        returns: Optional matrix of position return streams, one column each,
            used for effective breadth and the covariance-based vol forecast.
        drawdown: Current peak-to-trough drawdown, positive fraction.
        current: Present weights, for the no-trade band.
    """
    limits = limits or BookLimits()
    notes: list[str] = []

    if equity <= 0 and any(c.adv_usd > 0 for c in candidates):
        notes.append(
            "no equity supplied, so the ADV liquidity cap was not applied — "
            "positions are unconstrained by how much of them could be sold"
        )

    live = [c for c in candidates if c.volatility > 0 and abs(c.conviction) > 0]
    if not live:
        return BookResult(
            allocations=(), cash_weight=1.0, gross=0.0, net=0.0,
            effective_breadth=0.0,
            notes=("nothing to allocate: no candidate had a usable volatility "
                   "and a non-zero conviction — fully in cash, which is a "
                   "decision, not a failure",),
        )

    vols = np.array([c.volatility for c in live])
    signs = np.array([np.sign(c.conviction) or 1.0 for c in live])

    # 1. Inverse volatility. No return forecast anywhere in this line, which is
    #    the whole reason it survives out of sample.
    raw = 1.0 / vols
    raw = raw / raw.sum()
    binding = ["inverse_vol"] * len(live)

    # 2. Volatility targeting, then a hard leverage cap.
    forecast = (
        _portfolio_vol(raw, returns) if returns is not None
        else stress_gross(raw, vols, rho=0.3)
    )
    vol_scalar = 1.0
    if forecast > 0:
        vol_scalar = limits.target_volatility / forecast
    capped = min(vol_scalar, limits.max_leverage)
    if capped < vol_scalar:
        notes.append(
            f"volatility targeting wanted {vol_scalar:.2f}x gross to reach "
            f"{limits.target_volatility:.0%}; held at the {limits.max_leverage:.2f}x "
            f"leverage cap. Forecast volatility is {forecast:.1%}."
        )
    vol_scalar = capped
    weights = raw * vol_scalar

    # The drawdown ladder is computed here and applied LAST. Applying it now
    # would be undone by the per-position caps: halving a weight that is then
    # allowed back up to its 8% ceiling is not a 50% de-risk. Measured before
    # the fix, "-10% -> 50% gross" delivered 76%.
    dd_scalar = drawdown_multiplier(drawdown, limits.drawdown_ladder)

    # 4. Per-position ceilings. Each can only reduce.
    for i, candidate in enumerate(live):
        if weights[i] > limits.max_position_weight:
            weights[i] = limits.max_position_weight
            binding[i] = "max_position_weight"

        if candidate.adv_usd > 0 and equity > 0:
            # ADV is in dollars and a weight is a fraction of equity, so the
            # cap only exists once both are known. Without equity there is no
            # arithmetic here, only a plausible-looking number -- so the cap is
            # skipped and said out loud rather than applied to the wrong units.
            adv_cap = (candidate.adv_usd * limits.max_adv_share) / equity
            if adv_cap < weights[i]:
                weights[i] = adv_cap
                binding[i] = "max_adv_share"

        if candidate.forward_days < limits.forward_days_for_full_size:
            ramp = limits.new_strategy_fraction + (
                (1.0 - limits.new_strategy_fraction)
                * candidate.forward_days / max(limits.forward_days_for_full_size, 1)
            )
            if ramp < 1.0 and weights[i] * ramp < weights[i]:
                weights[i] *= ramp
                binding[i] = "forward_record_ramp"

    # 5. Sector ceilings, applied after per-position so the binding rule is the
    #    tighter of the two rather than whichever ran last.
    weights, binding = _cap_sectors(live, weights, binding, limits)

    # 6. Anything below the minimum is zero, not small.
    tiny = weights < limits.min_position_weight
    if tiny.any():
        for i in np.flatnonzero(tiny):
            binding[i] = "min_position_weight"
        weights = np.where(tiny, 0.0, weights)

    # 7. Gross and net ceilings.
    gross = float(np.sum(np.abs(weights)))
    if gross > limits.max_gross and gross > 0:
        weights = weights * (limits.max_gross / gross)
        binding = [b if b != "inverse_vol" else "max_gross" for b in binding]
        notes.append(f"scaled to the {limits.max_gross:.2f} gross cap")

    signed = weights * signs
    net = float(np.sum(signed))
    if abs(net) > limits.max_net and abs(net) > 0:
        signed = signed * (limits.max_net / abs(net))
        notes.append(f"scaled to the {limits.max_net:.2f} net cap")

    # 8. The no-trade band, against what is actually held.
    if current and dd_scalar >= 1.0:
        # Skipped while de-risking. The band exists to avoid paying costs for
        # noise; a drawdown response is not noise, and letting the band veto it
        # would be the band quietly overriding the risk ladder.
        for i, candidate in enumerate(live):
            held = current.get(candidate.symbol, 0.0)
            moved, why = no_trade_band(float(signed[i]), held, limits)
            if why == "inside the no-trade band" and moved != signed[i]:
                signed[i] = moved
                binding[i] = "no_trade_band"

    # 9. De-risking, last, so the ladder's number is the number that lands.
    if dd_scalar < 1.0:
        signed = signed * dd_scalar
        binding = ["drawdown_ladder"] * len(binding)
        notes.append(
            f"drawdown {drawdown:.1%} — gross cut to {dd_scalar:.0%} by the "
            f"de-risking ladder, applied after every other constraint so the "
            f"reduction is not given back by a position cap"
            + (". The book is flat pending review." if dd_scalar == 0 else "")
        )

    gross = float(np.sum(np.abs(signed)))
    net = float(np.sum(signed))
    breadth = _breadth(returns, live, signed)
    stressed = stress_gross(signed, vols, rho=0.8)

    if breadth and breadth < 3.0 and int(np.sum(np.abs(signed) > 0)) >= 10:
        notes.append(
            f"effective breadth is {breadth:.1f} across "
            f"{int(np.sum(np.abs(signed) > 0))} positions — this is a "
            f"concentrated bet that looks diversified"
        )
    notes.append(
        f"at 0.8 correlation across everything the book's volatility is "
        f"{stressed:.1%}; check that loss is survivable, because the position "
        f"count will not help in that state"
    )

    allocations = tuple(
        Allocation(
            symbol=c.symbol,
            weight=float(w),
            binding=b,
            explanation=_explain(c, float(w), b, limits),
        )
        for c, w, b in zip(live, signed, binding, strict=True)
        if abs(w) > 0
    )

    cash = max(0.0, 1.0 - gross)
    if cash > 0.3:
        notes.append(
            f"{cash:.0%} in cash. An allocator that must be fully invested "
            f"will always find something to buy; this one does not have to."
        )

    return BookResult(
        allocations=allocations, cash_weight=cash, gross=gross, net=net,
        effective_breadth=breadth, vol_scalar=vol_scalar,
        drawdown_scalar=dd_scalar, stressed_loss=stressed, notes=tuple(notes),
    )


def _cap_sectors(
    live: list[Candidate], weights: np.ndarray, binding: list[str],
    limits: BookLimits,
) -> tuple[np.ndarray, list[str]]:
    sectors: dict[str, list[int]] = {}
    for i, candidate in enumerate(live):
        sectors.setdefault(candidate.sector, []).append(i)

    for sector, idx in sectors.items():
        if sector == "unknown":
            continue
        total = float(np.sum(np.abs(weights[idx])))
        if total > limits.max_sector_weight and total > 0:
            scale = limits.max_sector_weight / total
            for i in idx:
                weights[i] *= scale
                binding[i] = "max_sector_weight"
    return weights, binding


def _portfolio_vol(weights: np.ndarray, returns: Any) -> float:
    """Forecast volatility from a shrunk covariance, annualised."""
    cov = shrunk_covariance(returns)
    if cov.shape[0] != weights.size:
        return 0.0
    variance = float(weights @ cov @ weights)
    return float(np.sqrt(max(variance, 0.0)) * np.sqrt(_ANNUAL))


def _breadth(returns: Any, live: list[Candidate], weights: np.ndarray) -> float:
    from godalgo.research.selection import effective_breadth

    if returns is None:
        return float(np.sum(np.abs(weights) > 0))
    data = np.asarray(getattr(returns, "values", returns), dtype=float)
    if data.ndim != 2 or data.shape[1] != len(live):
        return float(np.sum(np.abs(weights) > 0))
    held = np.flatnonzero(np.abs(weights) > 0)
    if held.size == 0:
        return 0.0
    return effective_breadth(data[:, held])


def _explain(
    candidate: Candidate, weight: float, binding: str, limits: BookLimits
) -> str:
    """One sentence naming the binding constraint.

    An allocation whose output cannot be attributed to a specific rule is one
    nobody will override when it is wrong.
    """
    pct = f"{abs(weight):.1%}"
    side = "long" if weight > 0 else "short"
    reasons = {
        "inverse_vol": (
            f"sized by inverse volatility ({candidate.volatility:.0%} "
            f"annualised); no constraint was binding"
        ),
        "max_position_weight": (
            f"capped by the {limits.max_position_weight:.0%} per-position "
            f"limit, not by the signal"
        ),
        "max_sector_weight": (
            f"capped by the {limits.max_sector_weight:.0%} limit on "
            f"{candidate.sector}, not by the signal"
        ),
        "max_adv_share": (
            f"capped by the {limits.max_adv_share:.0%} ADV limit — a larger "
            f"position could not be exited"
        ),
        "forward_record_ramp": (
            f"held to a fraction of target: {candidate.forward_days} days of "
            f"forward record against the {limits.forward_days_for_full_size} "
            f"needed for full size"
        ),
        "max_gross": f"scaled down to fit the {limits.max_gross:.2f} gross cap",
        "no_trade_band": "left where it was; the drift is inside the no-trade band",
        "drawdown_ladder": (
            f"scaled to {limits.drawdown_ladder!r} by the de-risking ladder; "
            f"the signal did not change, the book's drawdown did"
        ),
        "min_position_weight": (
            f"below the {limits.min_position_weight:.1%} minimum, so the "
            f"correct size is zero rather than small"
        ),
    }
    return f"{candidate.symbol} {side} at {pct} — " + reasons.get(binding, binding)
