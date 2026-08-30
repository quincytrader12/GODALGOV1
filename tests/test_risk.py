"""Risk limits must bind regardless of what the strategy layer asks for."""

from datetime import UTC, datetime, timedelta

import pytest

from godalgo.risk.limits import RiskLimits, RiskManager


def ts(hours=0):
    return datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=hours)


def test_gross_weight_is_capped():
    rm = RiskManager(limits=RiskLimits(max_gross_weight=0.5, max_position_change=1.0))
    rm.update_equity(1.0, ts())
    decision = rm.apply(3.0)
    assert decision.weight == 0.5
    assert "max_gross_weight" in decision.binding


def test_position_change_is_rate_limited():
    rm = RiskManager(limits=RiskLimits(max_position_change=0.10))
    rm.update_equity(1.0, ts())
    assert rm.apply(1.0).weight == pytest.approx(0.10)
    assert rm.apply(1.0).weight == pytest.approx(0.20)


def test_drawdown_kill_switch_flattens_and_latches():
    rm = RiskManager(limits=RiskLimits(max_drawdown_limit=0.10))
    rm.update_equity(1.0, ts(0))
    rm.update_equity(0.85, ts(1))            # -15% from peak
    assert rm.halted
    assert rm.apply(1.0).weight == 0.0

    # Recovery must not silently re-arm a drawdown halt.
    rm.update_equity(1.05, ts(2))
    assert rm.halted
    assert rm.apply(1.0).weight == 0.0


def test_daily_loss_halt_can_be_cleared_but_drawdown_cannot():
    rm = RiskManager(limits=RiskLimits(daily_loss_limit=0.03, max_drawdown_limit=0.90))
    rm.update_equity(1.0, ts(0))
    rm.update_equity(0.95, ts(1))
    assert rm.halted and rm.halt_reason.startswith("daily loss")
    rm.reset_day()
    assert not rm.halted


def test_consecutive_losses_derisk():
    limits = RiskLimits(max_consecutive_losses=3, consecutive_loss_derisk=0.5,
                        max_position_change=2.0, max_drawdown_limit=0.90,
                        daily_loss_limit=0.90)
    rm = RiskManager(limits=limits)
    equity = 1.0
    for i in range(5):
        equity *= 0.99
        rm.update_equity(equity, ts(i))
    decision = rm.apply(1.0)
    assert "consecutive_losses" in decision.binding
    assert decision.weight == pytest.approx(0.5)


def test_nonfinite_equity_halts():
    rm = RiskManager()
    rm.update_equity(float("nan"), ts())
    assert rm.halted


def test_limits_reject_incoherent_configuration():
    with pytest.raises(ValueError):
        RiskLimits(max_gross_weight=-1.0)
    with pytest.raises(ValueError):
        RiskLimits(daily_loss_limit=1.5)
