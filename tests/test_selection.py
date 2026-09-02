"""The gates that say no.

Every test here exists because the failure it describes produces a
BETTER-LOOKING result while being wrong. In this domain optimism is the
failure mode.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from godalgo.research.selection import (
    Holdout,
    SelectionGate,
    Trial,
    TrialLedger,
    buy_and_hold_sharpe,
    effective_breadth,
    luck_bar,
)


# --------------------------------------------------------------------------
# the selection problem
# --------------------------------------------------------------------------

def test_a_search_over_noise_produces_an_impressive_winner():
    """The whole reason this module exists, demonstrated rather than asserted.

    150 strategies with no edge on one history: the best of them looks good,
    because 150 draws from a zero-centred distribution have a maximum and the
    maximum is what a search reports.
    """
    rng = np.random.default_rng(11)
    ledger = TrialLedger()
    for i in range(150):
        # Pure noise. No edge exists anywhere in this loop.
        returns = rng.normal(0, 0.01, 500)
        sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(252))
        ledger.record(Trial(strategy=f"s{i}", symbol="X", in_sample_sharpe=sharpe))

    best = ledger.best()
    assert best.in_sample_sharpe > 1.0, "the luckiest of 150 should look good"
    # And the bar says so.
    assert luck_bar(ledger.n_pooled, ledger.sharpe_variance()) > 1.0


def test_the_luck_bar_rises_with_the_number_of_attempts():
    """More searching is a higher bar, not a better result."""
    assert luck_bar(1000, 0.25) > luck_bar(100, 0.25) > luck_bar(10, 0.25)


def test_trials_pool_across_symbols():
    """Fifty symbols by 154 combinations is ONE search of 7,700. Deflating
    per-symbol is the easiest way to keep a confident-looking result."""
    ledger = TrialLedger()
    for symbol in ("AAPL", "MSFT"):
        for i in range(77):
            ledger.record(Trial("m", symbol, in_sample_sharpe=0.1 * (i % 7)))

    assert ledger.n_pooled == 154
    assert ledger.n_for_symbol("AAPL") == 77
    summary = ledger.summary()
    # The pooled bar is the higher one; the gap is how much confidence came
    # from having looked at only one symbol.
    assert summary["luck_bar_pooled"] > summary["luck_bar_per_symbol"]


def test_failures_are_kept():
    """Dropping losers understates the attempt count, which is arithmetically
    identical to overstating the winner."""
    ledger = TrialLedger()
    ledger.record(Trial("good", "X", in_sample_sharpe=2.0))
    for _ in range(50):
        ledger.record(Trial("bad", "X", in_sample_sharpe=-1.0))
    assert ledger.n_pooled == 51


# --------------------------------------------------------------------------
# effective breadth
# --------------------------------------------------------------------------

def test_correlated_positions_count_as_one_bet():
    """Five holdings in correlated names is one holding with extra commission,
    presenting itself as five confirmations."""
    rng = np.random.default_rng(3)
    base = rng.normal(0, 0.01, 400)
    streams = np.column_stack([base + rng.normal(0, 0.0005, 400) for _ in range(5)])
    assert effective_breadth(streams) < 1.5


def test_independent_positions_count_as_many():
    rng = np.random.default_rng(3)
    streams = rng.normal(0, 0.01, (400, 5))
    assert effective_breadth(streams) > 4.0


def test_breadth_handles_degenerate_input():
    assert effective_breadth(np.zeros((10, 3))) == 0.0
    assert effective_breadth(np.random.default_rng(0).normal(0, 1, (10, 1))) == 1.0
    assert effective_breadth(np.empty((10, 0))) == 0.0


def test_breadth_never_claims_more_bets_than_positions():
    """A negative average correlation would report more independence than
    positions, which is not a claim worth making from a noisy estimate."""
    rng = np.random.default_rng(5)
    a = rng.normal(0, 0.01, 300)
    streams = np.column_stack([a, -a])
    assert effective_breadth(streams) <= 2.0


# --------------------------------------------------------------------------
# buy-and-hold, the bar nobody asks about
# --------------------------------------------------------------------------

def test_buy_and_hold_is_measurable():
    rng = np.random.default_rng(2)
    prices = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0006, 0.01, 600))))
    assert buy_and_hold_sharpe(prices) > 0


def test_buy_and_hold_of_a_flat_series_is_zero():
    assert buy_and_hold_sharpe(pd.Series([100.0] * 50)) == 0.0
    assert buy_and_hold_sharpe(pd.Series([100.0, 101.0])) == 0.0


# --------------------------------------------------------------------------
# the holdout
# --------------------------------------------------------------------------

def test_the_holdout_is_the_last_quarter_and_is_not_returned_to_a_search():
    holdout = Holdout(fraction=0.25)
    data = list(range(100))
    search, kept = holdout.split(data)
    assert len(search) == 75
    assert kept == list(range(75, 100))


def test_looks_at_the_holdout_are_counted():
    """A holdout is unbiased exactly once. After enough looks it is just
    another thing that was searched."""
    holdout = Holdout()
    assert holdout.to_dict()["still_clean"] is True
    holdout.look("candidate-a")
    assert holdout.to_dict()["still_clean"] is True
    holdout.look("candidate-b")
    assert holdout.looks == 2
    assert holdout.to_dict()["still_clean"] is False
    assert "candidate-a" in holdout.to_dict()["candidates"]


# --------------------------------------------------------------------------
# the gate
# --------------------------------------------------------------------------

def _ledger(n: int = 150, spread: float = 0.5) -> TrialLedger:
    rng = np.random.default_rng(1)
    ledger = TrialLedger()
    for i in range(n):
        ledger.record(Trial("s", "X", in_sample_sharpe=float(rng.normal(0, spread))))
    return ledger


def test_a_strong_candidate_that_trails_buy_and_hold_is_refused():
    """Observed: six candidates survived their holdouts and four trailed simply
    owning the stock. No statistical test caught it, because none was asking."""
    gate = SelectionGate()
    verdict = gate.evaluate(
        candidate=Trial("s", "AAPL", in_sample_sharpe=3.0),
        ledger=_ledger(),
        holdout_sharpe=0.9,
        hold_sharpe=1.4,
        deflated=0.99,
    )
    assert not verdict.adopted
    assert any("buy-and-hold" in r for r in verdict.reasons)


def test_a_winner_inside_the_luck_bar_is_refused():
    gate = SelectionGate()
    ledger = _ledger()
    bar = luck_bar(ledger.n_pooled, ledger.sharpe_variance())
    verdict = gate.evaluate(
        candidate=Trial("s", "X", in_sample_sharpe=bar - 0.01),
        ledger=ledger, holdout_sharpe=1.0, hold_sharpe=0.0, deflated=0.99,
    )
    assert not verdict.adopted
    assert any("no-edge" in r for r in verdict.reasons)


def test_an_untested_candidate_is_refused():
    verdict = SelectionGate().evaluate(
        candidate=Trial("s", "X", in_sample_sharpe=5.0),
        ledger=_ledger(), holdout_sharpe=None, hold_sharpe=0.0, deflated=0.99,
    )
    assert not verdict.adopted
    assert any("holdout" in r for r in verdict.reasons)


def test_every_missed_bar_is_reported_not_just_the_first():
    """Fixing one bar at a time across separate runs is its own form of
    searching."""
    verdict = SelectionGate().evaluate(
        candidate=Trial("s", "X", in_sample_sharpe=0.1),
        ledger=_ledger(), holdout_sharpe=-0.5, hold_sharpe=1.0, deflated=0.10,
    )
    assert len(verdict.reasons) >= 3


def test_a_genuinely_good_candidate_is_adopted():
    """The gate must be passable, or it is not a gate but a wall."""
    verdict = SelectionGate().evaluate(
        candidate=Trial("s", "X", in_sample_sharpe=4.0),
        ledger=_ledger(n=20, spread=0.3),
        holdout_sharpe=1.6, hold_sharpe=0.4, deflated=0.99,
    )
    assert verdict.adopted, verdict.reasons
    assert verdict.summary.startswith("adopted")


def test_the_gate_has_no_override():
    """A way around this check is a way to lose money with the system's
    blessing.

    Checks the interface rather than the prose: the first version of this test
    failed on the docstring that promised there was no override.
    """
    import dataclasses
    import inspect

    names = {f.name for f in dataclasses.fields(SelectionGate)}
    names |= set(inspect.signature(SelectionGate.evaluate).parameters)
    escapes = {"force", "override", "skip", "bypass", "ignore", "allow_anyway"}
    assert not (names & escapes), f"found an escape hatch: {names & escapes}"

    # And no environment variable can reach it either. Parsed rather than
    # grepped: the second version of this test failed on the sentence in the
    # docstring saying there was no environment variable.
    import ast

    tree = ast.parse(inspect.getsource(SelectionGate).lstrip())
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) \
                and isinstance(node.value.value, str):
            node.value.value = ""
    code = ast.unparse(tree)
    assert "environ" not in code and "getenv" not in code


def test_the_verdict_carries_its_numbers():
    verdict = SelectionGate().evaluate(
        candidate=Trial("s", "X", in_sample_sharpe=4.0),
        ledger=_ledger(), holdout_sharpe=1.2, hold_sharpe=0.3, deflated=0.99,
    )
    assert verdict.metrics["trials_pooled"] == 150
    assert verdict.metrics["buy_and_hold_sharpe"] == 0.3
    assert verdict.metrics["excess_over_hold"] == pytest.approx(0.9)
