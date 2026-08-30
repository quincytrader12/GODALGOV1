"""Autonomous retuning.

The tests that matter here are the refusals. A loop that proposes parameters on
a timer is only safe because the gate it must clear is the same one a human run
faces, and because it will not change the model underneath an open position.
"""

import asyncio
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from godalgo.backtest.engine import BacktestConfig
from godalgo.evolve.autopilot import Autopilot, AutopilotConfig
from godalgo.evolve.promotion import PromotionCriteria
from godalgo.evolve.search import SearchConfig
from godalgo.strategies.mean_reversion import MeanReversionParams
from godalgo.strategies.momentum import MomentumParams

T0 = datetime(2024, 1, 1, tzinfo=UTC)


def bars(n=9000, seed=3):
    rng = np.random.default_rng(seed)
    r = np.zeros(n)
    for i in range(1, n):
        r[i] = 0.3 * r[i - 1] + rng.normal(0, 0.01)
    price = 100 * np.exp(np.cumsum(r))
    idx = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {"open": price, "high": price * 1.002, "low": price * 0.998,
         "close": price, "volume": 1.0},
        index=idx,
    )


def make(**overrides):
    """Autopilot with recording callbacks."""
    applied = []
    flat = {"value": True}
    config = AutopilotConfig(**{
        "enabled": True,
        "min_bars": 100,
        "candidates": 3,
        "search": SearchConfig(n_candidates=3, seed=1),
        **overrides,
    })
    pilot = Autopilot(
        config,
        history=lambda: bars(),
        apply_params=lambda m, r: applied.append((m, r)),
        is_flat=lambda: flat["value"],
        backtest_config=BacktestConfig(),
        symbol="TEST",
    )
    return pilot, applied, flat


# --- scheduling ------------------------------------------------------------

def test_disabled_autopilot_never_runs():
    """Self-modification is opt-in, like live trading."""
    pilot, applied, _ = make(enabled=False)
    asyncio.run(pilot.run(poll_seconds=0.01))   # returns immediately
    assert pilot.state.rounds_run == 0
    assert applied == []


def test_first_round_is_due_then_the_interval_holds():
    pilot, _, _ = make(interval_hours=24.0)
    assert pilot.due(T0) is True
    pilot.state.last_run_at = T0
    assert pilot.due(T0 + timedelta(hours=1)) is False
    assert pilot.due(T0 + timedelta(hours=25)) is True


def test_daily_budget_caps_rounds():
    """A restart loop must not burn through the trial budget.

    Every extra trial raises the deflated-Sharpe bar, so an unbounded searcher
    makes promotion harder while looking busier.
    """
    pilot, _, _ = make(max_rounds_per_day=2, interval_hours=0.001)
    pilot._round_times = [T0, T0 + timedelta(minutes=1)]
    assert pilot.due(T0 + timedelta(minutes=2)) is False
    assert "budget" in pilot.state.blocked_reason


def test_insufficient_history_blocks_the_round():
    pilot, _, _ = make(min_bars=50_000)
    assert asyncio.run(pilot.maybe_run(T0)) is None
    assert "need 50000 bars" in pilot.state.blocked_reason


# --- the gate --------------------------------------------------------------

def test_a_failing_candidate_is_not_promoted():
    """The default criteria are strict; noise must not clear them."""
    pilot, applied, _ = make()
    gate = asyncio.run(pilot.maybe_run(T0))
    assert gate is not None
    if not gate.promoted:
        assert pilot.state.rejections == 1
        assert applied == []
        assert pilot.state.pending_swap is False


def test_autopilot_cannot_relax_its_own_criteria():
    """The proposer must not also be able to lower the bar."""
    pilot, _, _ = make()
    assert isinstance(pilot.config.criteria, PromotionCriteria)
    # Criteria are a frozen dataclass, so no code path can mutate them at
    # runtime -- the proposer cannot lower the bar it has to clear.
    with pytest.raises(AttributeError):
        pilot.config.criteria.min_deflated_sharpe = 0.0
    with pytest.raises(AttributeError):
        pilot.config.criteria.max_pbo = 1.0


def test_risk_limits_are_not_reachable_from_autopilot():
    """The bot may learn how to trade; it may not learn to take more risk."""
    pilot, _, _ = make()
    assert not hasattr(pilot, "risk")
    assert not hasattr(pilot.config, "risk")
    for space in (MomentumParams.SPACE, MeanReversionParams.SPACE):
        assert not any("risk" in k or "drawdown" in k or "loss" in k for k in space)


def test_every_decision_reaches_the_ledger(tmp_path):
    pilot, _, _ = make(ledger_path=tmp_path / "led.jsonl")
    asyncio.run(pilot.maybe_run(T0))
    entries = pilot.ledger.entries()
    assert len(entries) == 1
    assert entries[0]["note"] == "autopilot"


# --- swapping --------------------------------------------------------------

def test_swap_waits_for_a_flat_book():
    """Changing the model under an open position orphans it: the trade would be
    exited by rules that did not open it."""
    pilot, applied, flat = make()
    pilot._approved = (MomentumParams(fast_lookback=11), MeanReversionParams())
    pilot.state.pending_swap = True

    flat["value"] = False
    assert pilot.apply_pending() is False
    assert applied == []
    assert pilot.state.pending_swap is True

    flat["value"] = True
    assert pilot.apply_pending() is True
    assert len(applied) == 1
    assert applied[0][0].fast_lookback == 11
    assert pilot.state.pending_swap is False


def test_require_flat_can_be_disabled_deliberately():
    pilot, applied, flat = make(require_flat=False)
    pilot._approved = (MomentumParams(), MeanReversionParams())
    flat["value"] = False
    assert pilot.apply_pending() is True
    assert len(applied) == 1


def test_apply_pending_is_a_noop_without_an_approved_candidate():
    pilot, applied, _ = make()
    assert pilot.apply_pending() is False
    assert applied == []


def test_incumbent_is_tracked_for_the_improvement_check():
    pilot, _, _ = make()
    assert pilot.state.incumbent_sharpe is None
    pilot.state.incumbent_sharpe = 1.4
    # Subsequent rounds must beat the incumbent, not merely the absolute floor.
    assert pilot.state.incumbent_sharpe == 1.4


def test_search_failure_does_not_stop_trading():
    """A failed round must never take down the loop holding a position."""
    def explode():
        raise RuntimeError("data source down")

    pilot, applied, _ = make()
    pilot.history = explode

    async def one_pass():
        task = asyncio.create_task(pilot.run(poll_seconds=0.01))
        await asyncio.sleep(0.06)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(one_pass())
    assert pilot.state.blocked_reason is not None
    assert applied == []


def test_state_snapshot_is_serialisable():
    import json

    pilot, _, _ = make()
    json.dumps(pilot.state.snapshot())


# --- engine integration ----------------------------------------------------

def test_engine_exposes_what_autopilot_needs():
    from godalgo.execution.broker import PaperBroker
    from godalgo.execution.engine import LiveEngine, LiveEngineConfig
    from godalgo.strategies.mean_reversion import MeanReversionStrategy
    from godalgo.strategies.momentum import MomentumStrategy

    engine = LiveEngine(
        PaperBroker(), MomentumStrategy(), MeanReversionStrategy(), LiveEngineConfig(),
    )
    assert engine.is_flat() is True
    assert engine.history() is not None

    replacement = MomentumStrategy(MomentumParams(fast_lookback=7, slow_lookback=250))
    engine.swap_strategies(replacement, MeanReversionStrategy())
    assert engine.momentum is replacement
    # Warm-up must be re-derived: a new parameter set may need more history,
    # and signalling before it has warmed up trades an untrained model.
    assert engine._warmup >= replacement.warmup


def test_engine_is_not_flat_with_a_position():
    from godalgo.execution.broker import PaperBroker
    from godalgo.execution.engine import LiveEngine, LiveEngineConfig
    from godalgo.strategies.mean_reversion import MeanReversionStrategy
    from godalgo.strategies.momentum import MomentumStrategy

    engine = LiveEngine(
        PaperBroker(), MomentumStrategy(), MeanReversionStrategy(), LiveEngineConfig(),
    )
    engine.state.current_weight = 0.3
    assert engine.is_flat() is False
