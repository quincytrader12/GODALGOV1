"""Market scanner and portfolio supervision.

Two properties carry most of the value here, and both are refusals:

* a symbol whose expected edge cannot clear its round-trip cost is rejected,
  however strong its signal looks;
* correlated symbols are not selected together, because in crypto that is one
  bet held several times while believing it is diversification.
"""

import asyncio
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from godalgo.core.types import Regime
from godalgo.data.scanner import MarketScanner, ScanCriteria, ScanResult
from godalgo.portfolio.supervisor import PortfolioSupervisor, SupervisorConfig

N = 800
IDX = pd.date_range("2024-01-01", periods=N, freq="1h", tz="UTC")
T0 = datetime(2024, 6, 1, tzinfo=UTC)


def frame(returns, p0=100.0):
    px = p0 * np.exp(np.cumsum(returns))
    return pd.DataFrame(
        {"open": px, "high": px * 1.001, "low": px * 0.999, "close": px, "volume": 1.0},
        index=IDX,
    )


def trending(seed=1, phi=0.35, sigma=0.010):
    rng = np.random.default_rng(seed)
    r = np.zeros(N)
    for i in range(1, N):
        r[i] = phi * r[i - 1] + rng.normal(0, sigma)
    return r


def liquid(symbols):
    return {s: {"quoteVolume": 5e7, "bid": 99.99, "ask": 100.01} for s in symbols}


# --- filters ---------------------------------------------------------------

def test_illiquid_symbols_are_rejected():
    """A thin book fails on execution whatever its signal says."""
    hist = {"THIN/USDT": frame(trending())}
    result = MarketScanner().scan(hist, {"THIN/USDT": {"quoteVolume": 1e3, "bid": 99.9, "ask": 100.1}})
    assert not result.selected
    assert "illiquid" in result.rejected[0].rejected


def test_wide_spreads_are_rejected():
    hist = {"WIDE/USDT": frame(trending())}
    result = MarketScanner().scan(hist, {"WIDE/USDT": {"quoteVolume": 5e7, "bid": 99.0, "ask": 101.0}})
    assert not result.selected
    assert "spread" in result.rejected[0].rejected


def test_quiet_symbols_cannot_clear_costs():
    rng = np.random.default_rng(2)
    hist = {"QUIET/USDT": frame(rng.normal(0, 0.0006, N))}
    result = MarketScanner().scan(hist, liquid(hist))
    assert not result.selected
    assert result.rejected[0].rejected is not None


def test_wildly_volatile_symbols_are_rejected():
    rng = np.random.default_rng(3)
    hist = {"WILD/USDT": frame(rng.normal(0, 0.09, N))}
    result = MarketScanner().scan(hist, liquid(hist))
    assert not result.selected
    assert "volatile" in result.rejected[0].rejected


def test_a_symbol_with_no_regime_is_not_an_opportunity():
    rng = np.random.default_rng(4)
    hist = {"NOISE/USDT": frame(rng.normal(0, 0.01, N))}
    result = MarketScanner().scan(hist, liquid(hist))
    assert not result.selected


def test_a_clean_trend_survives():
    hist = {"BTC/USDT": frame(trending(seed=7))}
    result = MarketScanner().scan(hist, liquid(hist))
    assert [c.symbol for c in result.selected] == ["BTC/USDT"]
    assert result.selected[0].regime is Regime.TRENDING
    assert result.selected[0].headroom > 1.0


# --- the correlation trap --------------------------------------------------

def test_correlated_symbols_are_not_selected_together():
    """The failure this exists to prevent.

    BTC, ETH and SOL longs are one position in three wrappers. A scanner
    ranking purely on score concentrates risk precisely when it believes it is
    spreading it.
    """
    rng = np.random.default_rng(11)
    base = trending(seed=7)   # a seed that reliably classifies as trending
    hist = {
        "BTC/USDT": frame(base),
        "ETH/USDT": frame(base * 0.97 + rng.normal(0, 0.001, N)),
        "SOL/USDT": frame(base * 0.95 + rng.normal(0, 0.001, N)),
    }
    result = MarketScanner(ScanCriteria(max_candidates=3)).scan(hist, liquid(hist))
    assert len(result.selected) == 1, "clones were selected as if independent"
    assert any("correlated" in (c.rejected or "") for c in result.rejected)


def test_uncorrelated_symbols_can_coexist():
    hist = {
        "AAA/USDT": frame(trending(seed=21)),
        "BBB/USDT": frame(trending(seed=99)),
    }
    result = MarketScanner(ScanCriteria(max_candidates=3, max_correlation=0.85)).scan(
        hist, liquid(hist)
    )
    assert len(result.selected) >= 1


def test_max_candidates_is_respected():
    hist = {f"S{i}/USDT": frame(trending(seed=100 + i)) for i in range(8)}
    result = MarketScanner(ScanCriteria(max_candidates=2)).scan(hist, liquid(hist))
    assert len(result.selected) <= 2


def test_short_history_is_skipped_not_guessed():
    short = frame(trending())[:50]
    result = MarketScanner().scan({"NEW/USDT": short}, liquid(["NEW/USDT"]))
    assert not result.selected and not result.rejected


def test_result_is_serialisable():
    import json

    hist = {"BTC/USDT": frame(trending(seed=7))}
    json.dumps(MarketScanner().scan(hist, liquid(hist)).to_dict())


def test_criteria_validation():
    with pytest.raises(ValueError):
        ScanCriteria(min_annual_vol=2.0, max_annual_vol=1.0)
    with pytest.raises(ValueError):
        ScanCriteria(max_correlation=1.5)


# --- supervisor ------------------------------------------------------------

class FakeEngine:
    def __init__(self, symbol):
        self.config = type("C", (), {"symbol": symbol})()
        self.state = type("S", (), {"current_weight": 0.0})()
        self.flattened = False

    def is_flat(self):
        return abs(self.state.current_weight) < 1e-9

    async def flatten(self):
        self.flattened = True
        self.state.current_weight = 0.0


def supervisor(**kw):
    hist = {"BTC/USDT": frame(trending(seed=7))}
    config = SupervisorConfig(**kw)
    return PortfolioSupervisor(
        config, MarketScanner(),
        history=lambda: hist,
        tickers=lambda: liquid(hist),
        make_engine=FakeEngine,
        equity=lambda: 10_000.0,
    )


def test_per_symbol_budget_divides_by_the_limit_not_the_count():
    """Dividing by the live count lets the first engine claim the whole book."""
    s = supervisor(max_gross_exposure=1.0, max_concurrent=4)
    assert s.per_symbol_budget() == pytest.approx(0.25)


def test_gross_exposure_is_bounded_across_symbols():
    """The limit a per-symbol cap cannot express."""
    s = supervisor(max_gross_exposure=0.6, max_concurrent=4)
    s.engines = {"A": FakeEngine("A"), "B": FakeEngine("B")}
    s.engines["A"].state.current_weight = 0.5
    assert s.gross_exposure == pytest.approx(0.5)
    assert s.clamp_target("B", 0.5) == pytest.approx(0.1)


def test_reductions_are_never_blocked_by_exposure():
    """A limit that prevents risk being reduced is not a risk limit."""
    s = supervisor(max_gross_exposure=0.2, max_concurrent=2)
    s.engines = {"A": FakeEngine("A")}
    s.engines["A"].state.current_weight = 0.9
    assert s.clamp_target("A", 0.0) == pytest.approx(0.0)


def test_buying_power_holds_back_a_reserve():
    """An account at 100% deployed cannot act on anything, including an exit."""
    s = supervisor(max_concurrent=2, reserve_fraction=0.2)
    assert s.buying_power_for("BTC/USDT") == pytest.approx(10_000 * 0.8 / 2)


def test_retiring_a_symbol_flattens_its_position_first():
    """Dropping the engine first leaves the position with nothing managing it."""
    s = supervisor()
    engine = FakeEngine("BTC/USDT")
    engine.state.current_weight = 0.3
    s.engines["BTC/USDT"] = engine
    s.state.active["BTC/USDT"] = T0

    asyncio.run(s._retire("BTC/USDT"))
    assert engine.flattened is True
    assert "BTC/USDT" not in s.engines
    assert s.state.retired_open == 1


def test_a_failed_flatten_keeps_the_engine():
    """The engine still owns the position and its stop."""
    class Stubborn(FakeEngine):
        async def flatten(self):
            raise RuntimeError("venue down")

    s = supervisor()
    engine = Stubborn("BTC/USDT")
    engine.state.current_weight = 0.3
    s.engines["BTC/USDT"] = engine
    s.state.active["BTC/USDT"] = T0

    asyncio.run(s._retire("BTC/USDT"))
    assert "BTC/USDT" in s.engines, "engine dropped while still holding a position"


def test_minimum_hold_prevents_threshold_churn():
    """A symbol at the selection boundary would otherwise be churned at cost."""
    s = supervisor(min_hold_hours=6.0)
    s.state.active["OLD/USDT"] = T0
    empty = ScanResult(timestamp=T0, selected=(), rejected=(), scanned=0)
    assert "OLD/USDT" in s.target_universe(empty, T0 + timedelta(hours=1))
    assert "OLD/USDT" not in s.target_universe(empty, T0 + timedelta(hours=7))


def test_scan_cadence():
    s = supervisor(rescan_hours=6.0)
    assert s.due_for_scan(T0) is True
    s.state.last_scan_at = T0
    assert s.due_for_scan(T0 + timedelta(hours=1)) is False
    assert s.due_for_scan(T0 + timedelta(hours=7)) is True


def test_rotation_admits_and_records():
    s = supervisor(max_concurrent=2)
    admitted, retired = asyncio.run(s.rotate(T0))
    assert admitted == ["BTC/USDT"]
    assert retired == []
    assert "BTC/USDT" in s.engines
    assert s.state.rotations == 1


def test_supervisor_state_is_serialisable():
    import json

    s = supervisor()
    asyncio.run(s.rotate(T0))
    json.dumps(s.state.snapshot())


def test_config_validation():
    with pytest.raises(ValueError):
        SupervisorConfig(max_concurrent=0)
    with pytest.raises(ValueError):
        SupervisorConfig(reserve_fraction=1.0)
