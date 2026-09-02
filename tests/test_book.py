"""Sizing the book.

A portfolio of good strategies sized badly loses money; a portfolio of mediocre
strategies sized well often does not. Everything here is about the second half.

The recurring theme: constraints may only ever reduce. A rule that can increase
a weight is not a constraint, and the ordering bug this suite caught -- the
drawdown ladder firing before the position caps, so a "50% de-risk" delivered
76% -- is exactly what that looks like in practice.
"""

from __future__ import annotations

import numpy as np
import pytest

from godalgo.portfolio.book import (
    BookLimits,
    Candidate,
    allocate,
    drawdown_multiplier,
    no_trade_band,
    shrunk_covariance,
    stress_gross,
)


def _book(n: int = 4, **kw) -> list[Candidate]:
    vols = [0.15, 0.25, 0.40, 0.60, 0.90, 0.20][:n]
    return [
        Candidate(f"S{i}", v, sector=kw.pop("sector", "tech"),
                  forward_days=kw.get("forward_days", 200),
                  adv_usd=kw.get("adv_usd", 5e9))
        for i, v in enumerate(vols)
    ]


# --------------------------------------------------------------------------
# the default allocator
# --------------------------------------------------------------------------

def test_a_quieter_instrument_gets_more_weight():
    """Inverse volatility, and no return forecast anywhere in the line -- the
    reason this survives out of sample when optimisation does not.

    The position cap is lifted here on purpose: with it in place both names
    sit at the ceiling and the test would pass or fail on the cap rather than
    on the sizing rule it is about.
    """
    limits = BookLimits(max_position_weight=1.0, max_sector_weight=1.0)
    result = allocate(
        [Candidate("CALM", 0.10, forward_days=200),
         Candidate("WILD", 0.80, forward_days=200)],
        limits=limits,
    )
    weights = {a.symbol: a.weight for a in result.allocations}
    assert weights["CALM"] > weights["WILD"] * 5


def test_no_expected_return_is_consulted():
    """Expected returns are the worst-known input. Two candidates differing
    only in conviction magnitude must size the same."""
    a = allocate([Candidate("A", 0.2, conviction=1.0), Candidate("B", 0.2, conviction=0.1)])
    weights = [round(x.weight, 9) for x in a.allocations]
    assert weights[0] == weights[1]


def test_conviction_sets_direction_not_size():
    result = allocate([Candidate("A", 0.2, conviction=-1.0)])
    assert result.allocations[0].weight < 0


# --------------------------------------------------------------------------
# constraints may only reduce
# --------------------------------------------------------------------------

def test_no_position_exceeds_its_cap():
    limits = BookLimits(max_position_weight=0.05)
    result = allocate(_book(2), limits=limits, equity=100_000)
    assert all(abs(a.weight) <= limits.max_position_weight + 1e-9
               for a in result.allocations)


def test_a_sector_cap_binds_across_its_members():
    limits = BookLimits(max_sector_weight=0.10, max_position_weight=0.50)
    result = allocate(_book(4), limits=limits, equity=1e9)
    total = sum(abs(a.weight) for a in result.allocations)
    assert total <= 0.10 + 1e-9


def test_an_illiquid_name_is_capped_by_what_could_be_exited():
    """You cannot exit what you cannot trade, and the position too big to exit
    is the one you most want out of."""
    thin = Candidate("THIN", 0.30, adv_usd=5_000_000, forward_days=200)
    deep = Candidate("DEEP", 0.30, adv_usd=1e10, forward_days=200)
    result = allocate([thin, deep], equity=1_000_000)
    weights = {a.symbol: a.weight for a in result.allocations}
    binding = {a.symbol: a.binding for a in result.allocations}
    assert weights["THIN"] < weights["DEEP"]
    assert binding["THIN"] == "max_adv_share"


def test_a_name_too_thin_to_hold_usefully_is_dropped_entirely():
    """The two liquidity rules meeting: if the most that could be exited is
    below the minimum worth holding, the answer is no position rather than a
    token one that pays a spread to express nothing."""
    thin = Candidate("THIN", 0.30, adv_usd=100_000, forward_days=200)
    deep = Candidate("DEEP", 0.30, adv_usd=1e10, forward_days=200)
    result = allocate([thin, deep], equity=1_000_000)
    assert [a.symbol for a in result.allocations] == ["DEEP"]


def test_without_equity_the_liquidity_cap_is_skipped_and_said_so():
    """A cap applied to the wrong units is worse than no cap. Dollars are not
    a weight until equity is known."""
    result = allocate([Candidate("X", 0.3, adv_usd=1e6)], equity=0.0)
    assert any("ADV" in n for n in result.notes)


def test_a_dust_position_is_zero_not_small():
    """Below the point where the spread exceeds the edge, the correct size is
    zero. A dust position pays costs to express nothing."""
    limits = BookLimits(min_position_weight=0.10)
    result = allocate(_book(4), limits=limits, equity=1e9)
    assert all(abs(a.weight) >= 0.10 for a in result.allocations)


def test_gross_never_exceeds_its_cap():
    limits = BookLimits(max_gross=0.5, max_position_weight=1.0,
                        max_sector_weight=1.0, min_position_weight=0.0)
    result = allocate(_book(4), limits=limits, equity=1e9)
    assert result.gross <= 0.5 + 1e-9


# --------------------------------------------------------------------------
# leverage and volatility targeting
# --------------------------------------------------------------------------

def test_volatility_targeting_is_capped():
    """A vol-targeting rule with no cap levers enormously into a quiet market,
    which is immediately before it stops being one."""
    calm = [Candidate(f"C{i}", 0.02) for i in range(4)]
    limits = BookLimits(target_volatility=0.15, max_leverage=1.0)
    result = allocate(calm, limits=limits, equity=1e9)
    assert result.vol_scalar <= 1.0
    assert any("leverage cap" in n for n in result.notes)


def test_the_cap_is_reported_not_silent():
    calm = [Candidate(f"C{i}", 0.02) for i in range(4)]
    result = allocate(calm, limits=BookLimits(max_leverage=1.0), equity=1e9)
    assert any("volatility targeting wanted" in n for n in result.notes)


# --------------------------------------------------------------------------
# the drawdown ladder — the ordering bug this suite caught
# --------------------------------------------------------------------------

def test_the_ladder_delivers_exactly_the_gross_it_states():
    """The bug: the ladder fired before the position caps, so halved weights
    were allowed straight back up to their ceiling. "-10% -> 50% gross"
    measured 76%. It is applied last now, and this asserts the number."""
    book = _book(3)
    base = allocate(book, equity=1e9).gross
    for drawdown, expected in ((0.06, 0.75), (0.11, 0.50), (0.16, 0.0)):
        result = allocate(book, equity=1e9, drawdown=drawdown)
        assert result.gross == pytest.approx(base * expected, rel=1e-6), drawdown


def test_the_ladder_thresholds():
    ladder = BookLimits().drawdown_ladder
    assert drawdown_multiplier(0.0, ladder) == 1.0
    assert drawdown_multiplier(0.05, ladder) == 0.75
    assert drawdown_multiplier(0.14, ladder) == 0.50
    assert drawdown_multiplier(0.20, ladder) == 0.0


def test_de_risking_is_not_vetoed_by_the_no_trade_band():
    """The band exists to avoid paying costs for noise. A drawdown response is
    not noise, and letting the band override the ladder would be the cost
    optimisation quietly disabling the risk control."""
    book = _book(3)
    held = {c.symbol: 0.30 for c in book}
    result = allocate(book, equity=1e9, drawdown=0.16, current=held)
    assert result.gross == 0.0


# --------------------------------------------------------------------------
# forward evidence, not backtest evidence
# --------------------------------------------------------------------------

def test_a_new_strategy_enters_small():
    """Allocation is a function of out-of-sample track length, never of the
    in-sample Sharpe that got it adopted."""
    fresh = Candidate("NEW", 0.25, forward_days=0)
    proven = Candidate("OLD", 0.25, forward_days=200)
    result = allocate([fresh, proven], equity=1e9)
    weights = {a.symbol: a.weight for a in result.allocations}
    assert weights["NEW"] < weights["OLD"] * 0.5


def test_the_ramp_scales_with_the_record():
    weights = []
    for days in (0, 30, 60):
        result = allocate(
            [Candidate("X", 0.25, forward_days=days)], equity=1e9,
        )
        weights.append(result.allocations[0].weight)
    assert weights[0] < weights[1] < weights[2]


# --------------------------------------------------------------------------
# correlation
# --------------------------------------------------------------------------

def test_effective_breadth_is_reported():
    rng = np.random.default_rng(4)
    base = rng.normal(0, 0.01, 300)
    streams = np.column_stack([base + rng.normal(0, 0.0005, 300) for _ in range(4)])
    result = allocate(_book(4), equity=1e9, returns=streams)
    assert result.effective_breadth < 2.0


def test_the_book_is_stress_tested_at_high_correlation():
    """Every diversification estimate made in calm conditions overstates the
    protection available in the conditions it was for."""
    result = allocate(_book(4), equity=1e9)
    assert result.stressed_loss > 0
    assert any("0.8 correlation" in n for n in result.notes)


def test_stress_exceeds_the_calm_estimate():
    weights = np.array([0.1, 0.1, 0.1, 0.1])
    vols = np.array([0.3, 0.3, 0.3, 0.3])
    assert stress_gross(weights, vols, rho=0.8) > stress_gross(weights, vols, rho=0.1)


def test_covariance_is_shrunk_toward_a_target():
    """Sample covariance is unusable when assets approach observations: it is
    near-singular and an optimiser will lever into the noise directions."""
    rng = np.random.default_rng(6)
    thin = rng.normal(0, 0.01, (12, 10))       # 10 assets, 12 observations
    sample = np.cov(thin, rowvar=False, ddof=1)
    shrunk = shrunk_covariance(thin)
    # Shrinkage pulls the extreme off-diagonals in, which is the whole point.
    off = ~np.eye(10, dtype=bool)
    assert np.abs(shrunk[off]).max() < np.abs(sample[off]).max()
    assert np.all(np.linalg.eigvalsh(shrunk) > -1e-12)


# --------------------------------------------------------------------------
# cash, bands and explainability
# --------------------------------------------------------------------------

def test_cash_is_a_position():
    """An allocator that must be fully invested will always find something to
    buy. Being 40% cash is a decision, and frequently the right one."""
    result = allocate(_book(2), limits=BookLimits(max_position_weight=0.05),
                      equity=1e9)
    assert result.cash_weight > 0.85
    assert result.cash_weight + result.gross == pytest.approx(1.0)


def test_nothing_to_allocate_explains_itself():
    result = allocate([])
    assert result.cash_weight == 1.0
    assert "decision, not a failure" in result.notes[0]


def test_the_band_leaves_small_drifts_alone():
    target, current = 0.10, 0.105
    moved, why = no_trade_band(target, current, BookLimits())
    assert moved == current
    assert "inside" in why


def test_the_band_trades_back_to_its_edge_not_the_centre():
    """Moving to the centre pays the full distance for a position that drifts
    straight back out."""
    limits = BookLimits(band_relative=0.20, band_absolute=0.0)
    moved, why = no_trade_band(0.10, 0.20, limits)
    assert moved == pytest.approx(0.12)
    assert "band edge" in why


def test_every_allocation_names_its_binding_constraint():
    """An allocation nobody can attribute to a rule is one nobody will
    override when it is wrong, and it will be wrong."""
    result = allocate(_book(4), equity=100_000)
    for a in result.allocations:
        assert a.symbol in a.explanation
        assert a.binding
        assert len(a.explanation) > 30
        assert "—" in a.explanation


def test_the_output_is_boring():
    """Most weights should sit at a constraint rather than at an optimum.
    Interesting, concentrated weights are what fitting estimation error looks
    like from the inside."""
    result = allocate(_book(4), equity=100_000)
    at_a_limit = [a for a in result.allocations if a.binding != "inverse_vol"]
    assert len(at_a_limit) >= len(result.allocations) // 2
