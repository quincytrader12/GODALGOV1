"""The forward record, settlement, and the allocator actually deciding.

The forward record's only feature that matters is that it survives a restart:
its value is months of accumulation, and one that resets is not a forward
record. Everything else here is about a number being reported with the
uncertainty that makes it usable, or a plan being checked before it strands
half-executed.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from godalgo.research.forward import (
    DivergenceRule,
    ForwardRecord,
    bootstrap_sharpe,
    cost_breakeven_bps,
)
from godalgo.risk.settlement import AccountState, plan_rebalance


@pytest.fixture
def record(tmp_path) -> ForwardRecord:
    return ForwardRecord(tmp_path / "forward.jsonl")


def _fill(record: ForwardRecord, n: int, mean: float = 0.001, seed: int = 1,
          strategy: str = "mom", regime: str = "trending") -> None:
    rng = np.random.default_rng(seed)
    for i in range(n):
        record.record_day(
            strategy=strategy, symbol="SPY", pnl=float(rng.normal(20, 100)),
            ret=float(rng.normal(mean, 0.01)), regime=regime,
            day=dt.date(2026, 1, 1) + dt.timedelta(days=i),
        )


# --------------------------------------------------------------------------
# persistence — the only feature that matters
# --------------------------------------------------------------------------

def test_the_record_survives_a_restart(record, tmp_path):
    """Months of accumulation is the value. One that resets is not a record."""
    _fill(record, 40)
    assert record.days() == 40

    reopened = ForwardRecord(tmp_path / "forward.jsonl")
    assert reopened.days() == 40
    assert len(reopened.entries) == 40


def test_it_is_written_as_it_goes_not_at_shutdown(record):
    """The failure that destroys a forward record is the one that stops the
    process shutting down cleanly."""
    record.record_day(strategy="m", symbol="X", pnl=1.0, ret=0.001)
    assert record.path.exists()
    assert record.path.read_text().strip()


def test_a_corrupt_line_does_not_lose_the_rest(record, tmp_path):
    """A partial write from a hard kill is the last line, not the useful ones."""
    _fill(record, 10)
    with record.path.open("a") as handle:
        handle.write('{"day": "not-a-date"\n')

    reopened = ForwardRecord(tmp_path / "forward.jsonl")
    assert reopened.days() == 10


def test_days_counts_days_not_entries(record):
    """A strategy trading four symbols does not have four times the evidence."""
    for symbol in ("A", "B", "C", "D"):
        record.record_day(strategy="m", symbol=symbol, pnl=1.0, ret=0.001,
                          day=dt.date(2026, 1, 1))
    assert record.days() == 1


def test_paper_and_live_records_are_kept_apart(record):
    """Merging them would let paper days inflate the track length that decides
    live sizing."""
    _fill(record, 30)
    record.record_day(strategy="mom", symbol="SPY", pnl=1.0, ret=0.01,
                      mode="live", day=dt.date(2027, 1, 1))
    assert record.days(mode="paper") == 30
    assert record.days(mode="live") == 1


# --------------------------------------------------------------------------
# uncertainty travels with the number
# --------------------------------------------------------------------------

def test_a_short_record_says_it_cannot_support_a_conclusion():
    """A point estimate from a short record is not a small amount of evidence,
    it is a number that looks like evidence."""
    estimate = bootstrap_sharpe(np.random.default_rng(0).normal(0.002, 0.01, 15))
    assert "too short" in estimate.summary


def test_an_interval_spanning_zero_is_reported_as_no_edge():
    estimate = bootstrap_sharpe(np.random.default_rng(0).normal(0.0, 0.01, 60))
    assert estimate.spans_zero
    assert "consistent with having no edge" in estimate.summary


def test_a_clear_edge_is_reported_as_positive():
    estimate = bootstrap_sharpe(np.random.default_rng(0).normal(0.004, 0.004, 400))
    assert not estimate.spans_zero
    assert "positive at 95%" in estimate.summary


def test_the_interval_narrows_with_more_days():
    rng = np.random.default_rng(2)
    short = bootstrap_sharpe(rng.normal(0.001, 0.01, 40))
    long = bootstrap_sharpe(rng.normal(0.001, 0.01, 800))
    assert (long.high - long.low) < (short.high - short.low)


def test_the_cost_level_at_which_the_edge_dies_is_reported():
    """'Profitable at 5bps' and 'profitable at 50bps' are different claims."""
    returns = np.full(100, 0.001)
    assert cost_breakeven_bps(returns, trades_per_day=1.0) == pytest.approx(10.0)
    assert cost_breakeven_bps(returns, trades_per_day=10.0) == pytest.approx(1.0)
    assert cost_breakeven_bps(np.full(100, -0.001), 1.0) == 0.0


# --------------------------------------------------------------------------
# attribution
# --------------------------------------------------------------------------

def test_a_book_carried_by_one_strategy_is_named(record):
    """A book up while three of four strategies lose is one strategy's bet,
    and that is worth knowing before the fourth turns."""
    day = dt.date(2026, 3, 1)
    record.record_day(strategy="winner", symbol="A", pnl=500.0, ret=0.02, day=day)
    for name in ("a", "b", "c"):
        record.record_day(strategy=name, symbol="B", pnl=-100.0, ret=-0.004, day=day)

    attribution = record.attribution(day)
    assert attribution.total == pytest.approx(200.0)
    assert attribution.carried_by_one == "winner"


def test_a_broadly_profitable_book_is_not_flagged(record):
    day = dt.date(2026, 3, 1)
    for name in ("a", "b"):
        record.record_day(strategy=name, symbol="X", pnl=100.0, ret=0.01, day=day)
    assert record.attribution(day).carried_by_one is None


def test_performance_splits_by_regime(record):
    """A strategy that makes everything in bull markets and gives it back
    sideways is a leveraged long in a costume."""
    _fill(record, 60, mean=0.004, regime="trending", seed=3)
    _fill(record, 60, mean=-0.003, regime="mean_reverting", seed=4)
    breakdown = record.regime_breakdown()
    assert set(breakdown) == {"trending", "mean_reverting"}
    assert breakdown["trending"].sharpe > breakdown["mean_reverting"].sharpe


def test_the_summary_leads_with_out_of_sample(record):
    """The opposite of what a backtest engine usually shows, and the design."""
    _fill(record, 40)
    summary = record.summary()
    assert list(summary)[0] == "out_of_sample"


# --------------------------------------------------------------------------
# retirement on divergence
# --------------------------------------------------------------------------

def test_a_strategy_that_diverges_from_its_backtest_is_retired():
    rule = DivergenceRule()
    retire, why = rule.verdict(
        live_returns=np.random.default_rng(0).normal(-0.003, 0.01, 70),
        backtest_sharpe=1.5, backtest_max_drawdown=0.10,
    )
    assert retire
    assert "retired" in why and "1.50" in why


def test_a_strategy_matching_its_backtest_is_kept():
    retire, why = DivergenceRule().verdict(
        live_returns=np.random.default_rng(0).normal(0.002, 0.006, 70),
        backtest_sharpe=1.5, backtest_max_drawdown=0.30,
    )
    assert not retire
    assert "within what the backtest claimed" in why


def test_too_short_a_record_does_not_retire_anything():
    """Retiring on ten days would be reacting to noise."""
    retire, why = DivergenceRule().verdict(
        live_returns=np.random.default_rng(0).normal(-0.01, 0.01, 10),
        backtest_sharpe=1.5, backtest_max_drawdown=0.10,
    )
    assert not retire
    assert "too short" in why


# --------------------------------------------------------------------------
# settlement and the day-trade budget at portfolio level
# --------------------------------------------------------------------------

def test_a_rebalance_needing_more_day_trades_than_remain_is_trimmed():
    """Discovering this on the fourth order means the first three were spent on
    a rotation that cannot complete."""
    account = AccountState(equity=10_000, cash=10_000, settled_cash=10_000,
                           day_trades_left=1, pdt_constrained=True)
    targets = {"AAPL": 0.0, "MSFT": 0.0, "NVDA": 0.0}
    current = {"AAPL": 0.2, "MSFT": 0.2, "NVDA": 0.2}
    plan = plan_rebalance(targets, current, account,
                          same_day_exits={"AAPL", "MSFT", "NVDA"})
    assert not plan.complete
    assert len(plan.deferred) == 2
    assert any("Holding overnight is not a day trade" in r for r in plan.reasons)


def test_crypto_legs_are_never_deferred_for_the_day_trade_rule():
    account = AccountState(equity=10_000, cash=10_000, settled_cash=10_000,
                           day_trades_left=0, pdt_constrained=True)
    targets = {"BTC/USD": 0.0, "AAPL": 0.0}
    current = {"BTC/USD": 0.2, "AAPL": 0.2}
    plan = plan_rebalance(targets, current, account, crypto={"BTC/USD"},
                          same_day_exits={"BTC/USD", "AAPL"})
    assert "BTC/USD" in plan.reachable
    assert "AAPL" in plan.deferred


def test_unsettled_cash_cannot_be_spent():
    """A cash account that treats a sale as immediate buying power produces a
    target the broker will reject halfway through."""
    account = AccountState(equity=10_000, cash=10_000, settled_cash=2_000,
                           is_margin=False)
    plan = plan_rebalance({"AAPL": 0.8}, {}, account)
    assert plan.reachable["AAPL"] < 0.8
    assert any("not spendable until it settles" in r for r in plan.reasons)


def test_a_margin_account_may_use_its_full_cash():
    account = AccountState(equity=10_000, cash=10_000, settled_cash=2_000,
                           is_margin=True)
    plan = plan_rebalance({"AAPL": 0.8}, {}, account)
    assert plan.reachable["AAPL"] == pytest.approx(0.8)


def test_a_reachable_plan_is_left_alone():
    account = AccountState(equity=10_000, cash=10_000, settled_cash=10_000)
    plan = plan_rebalance({"AAPL": 0.05}, {}, account)
    assert plan.complete
    assert plan.reasons == ()
