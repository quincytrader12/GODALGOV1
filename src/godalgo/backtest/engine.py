"""Backtest engine.

Vectorised where it is safe to be, explicit where it is not. The parts that must
not be vectorised are the ones with path dependence -- risk state, the kill
switch, the turnover buffer -- because each depends on what the previous bar
actually did rather than on what it wanted to do.

Three conventions keep the results honest, and every one of them exists because
its absence is a standard way to produce a backtest that cannot be traded:

**Signals are lagged one bar.** A signal computed from bar ``t``'s close can only
be traded at bar ``t+1``. Skipping this is the most common lookahead bug there
is, and it is usually worth an enormous, entirely fictional Sharpe.

**Costs are charged on turnover, not on trades.** Every change in target weight
pays fees plus modelled slippage. Costs scale with how much the position moved,
which is what actually happens.

**Warm-up is discarded.** Bars before every component has enough history are cut
from the reported statistics, rather than being included as flat.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from godalgo.backtest.metrics import PerformanceStats, compute_stats
from godalgo.core.types import Regime
from godalgo.features.regime import classify_regime
from godalgo.portfolio.allocator import AllocationConfig, blend_signals
from godalgo.portfolio.sizing import apply_turnover_buffer, volatility_target_scalar
from godalgo.risk.limits import RiskLimits, RiskManager
from godalgo.strategies.base import Strategy

__all__ = ["BacktestConfig", "BacktestResult", "CostModel", "run_backtest"]


@dataclass(frozen=True, slots=True)
class CostModel:
    """Transaction costs as fractions of notional traded.

    Defaults are deliberately pessimistic. A backtest that only survives at
    optimistic costs is telling you it does not survive.
    """

    fee_rate: float = 0.0006
    """Per-side taker fee. ~6bp is a realistic retail crypto taker tier."""

    slippage_rate: float = 0.0005
    """Modelled slippage per unit turnover, on top of fees."""

    funding_rate: float = 0.0
    """Per-bar financing on gross exposure, for perpetual futures.

    Left at zero by default because it is venue- and symbol-specific. It is not
    optional for a perp book -- funding regularly dominates fees for a strategy
    that holds directional exposure for days.
    """

    def cost_of(self, turnover: float, gross_exposure: float) -> float:
        """Total cost for one bar, as a fraction of equity."""
        trade_cost = abs(turnover) * (self.fee_rate + self.slippage_rate)
        carry_cost = abs(gross_exposure) * self.funding_rate
        return trade_cost + carry_cost


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    """Everything the engine needs beyond the strategies and the data."""

    bars_per_day: float = 24.0
    """Bars per calendar day. 24 for 1h bars, 1 for daily, 96 for 15m."""

    target_annual_vol: float = 0.20
    regime_window: int = 250
    """Trailing bars used for each regime classification."""

    regime_refit_every: int = 24
    """Bars between regime refits. Daily on 1h bars by default."""

    turnover_buffer: float = 0.05
    costs: CostModel = field(default_factory=CostModel)
    allocation: AllocationConfig = field(default_factory=AllocationConfig)
    risk: RiskLimits = field(default_factory=RiskLimits)
    trading_days: float = 365.0

    @property
    def periods_per_year(self) -> float:
        return self.bars_per_day * self.trading_days


@dataclass(slots=True)
class BacktestResult:
    """Full output of a backtest run."""

    equity: pd.Series
    returns: pd.Series
    weights: pd.Series
    frame: pd.DataFrame
    """Per-bar diagnostics: signals, regime, weights, costs, risk decisions."""

    stats: PerformanceStats
    halted_at: pd.Timestamp | None = None
    halt_reason: str | None = None

    def summary(self) -> str:
        s = self.stats
        lines = [
            f"observations   {s.n_observations}",
            f"total return   {s.total_return:>8.2%}",
            f"annual return  {s.annual_return:>8.2%}",
            f"annual vol     {s.annual_volatility:>8.2%}",
            f"sharpe         {s.sharpe:>8.2f}",
            f"sortino        {s.sortino:>8.2f}",
            f"max drawdown   {s.max_drawdown:>8.2%}",
            f"calmar         {s.calmar:>8.2f}",
            f"hit rate       {s.hit_rate:>8.2%}  (of active bars)",
            f"time in mkt    {s.time_in_market:>8.2%}",
            f"turnover/yr    {s.turnover:>8.1f}x",
        ]
        if self.halted_at is not None:
            lines.append(f"HALTED at {self.halted_at}: {self.halt_reason}")
        return "\n".join(lines)


def compute_regime_series(
    bars: pd.DataFrame,
    symbol: str,
    window: int,
    refit_every: int,
) -> pd.DataFrame:
    """Classify the regime on a rolling basis.

    Refits every ``refit_every`` bars and holds the call in between. Refitting
    every bar is both expensive and pointless: the tests read a long trailing
    window that barely changes bar to bar, and the extra churn would show up as
    allocation noise rather than as information.

    Returns:
        Frame with ``regime``, ``confidence``, ``hurst``, ``variance_ratio``,
        and ``half_life``. Bars before the first fit are INDETERMINATE with zero
        confidence, so the allocator de-risks rather than guessing.
    """
    close = bars["close"].astype(float)
    n = len(close)

    regimes: list[Regime] = []
    confidences = np.zeros(n)
    hursts = np.full(n, 0.5)
    vrs = np.ones(n)
    half_lives: list[float | None] = []

    current = Regime.INDETERMINATE
    conf = 0.0
    hurst = 0.5
    vr = 1.0
    hl: float | None = None

    for i in range(n):
        if i >= window and (i - window) % refit_every == 0:
            state = classify_regime(close.iloc[i - window : i + 1], symbol)
            current, conf, hurst, vr, hl = (
                state.regime,
                state.confidence,
                state.hurst,
                state.variance_ratio,
                state.half_life,
            )
        regimes.append(current)
        confidences[i] = conf
        hursts[i] = hurst
        vrs[i] = vr
        half_lives.append(hl)

    return pd.DataFrame(
        {
            "regime": regimes,
            "confidence": confidences,
            "hurst": hursts,
            "variance_ratio": vrs,
            "half_life": half_lives,
        },
        index=close.index,
    )


def run_backtest(
    bars: pd.DataFrame,
    momentum: Strategy,
    reversion: Strategy,
    config: BacktestConfig | None = None,
    symbol: str = "UNKNOWN",
) -> BacktestResult:
    """Run the full pipeline over a bar series.

    Pipeline order, which is also the dependency order:
    signals -> regime -> blend -> vol target -> turnover buffer -> lag -> risk
    -> costs -> equity.

    Args:
        bars: OHLCV frame indexed by timestamp, oldest first. Needs ``close``.
        momentum: A trend strategy instance.
        reversion: A reversion strategy instance.
        config: Engine configuration; defaults used if omitted.
        symbol: Label for the regime classifier and diagnostics.

    Returns:
        A ``BacktestResult`` with the equity curve, per-bar diagnostics, and
        summary statistics computed only over the post-warm-up window.

    Raises:
        ValueError: If ``bars`` lacks a ``close`` column, is not sorted
            ascending, or is too short to clear warm-up.
    """
    cfg = config or BacktestConfig()

    if "close" not in bars:
        raise ValueError("bars must contain a 'close' column")
    if not bars.index.is_monotonic_increasing:
        raise ValueError("bars must be sorted oldest-first")

    close = bars["close"].astype(float)
    warmup = max(momentum.warmup, reversion.warmup, cfg.regime_window)
    if len(close) <= warmup + 10:
        raise ValueError(
            f"need more than {warmup + 10} bars to backtest, got {len(close)}"
        )

    # --- signals -----------------------------------------------------------
    mom_signal = momentum.generate(bars)
    rev_signal = reversion.generate(bars)

    # --- regime and blending ----------------------------------------------
    regime_frame = compute_regime_series(
        bars, symbol, cfg.regime_window, cfg.regime_refit_every
    )
    blended = blend_signals(
        mom_signal,
        rev_signal,
        regime_frame["regime"],
        regime_frame["confidence"],
        cfg.allocation,
    )

    # --- sizing ------------------------------------------------------------
    asset_returns = close.pct_change()
    vol_scalar = volatility_target_scalar(
        np.log(close).diff(),
        cfg.target_annual_vol,
        cfg.bars_per_day,
        max_leverage=cfg.risk.max_gross_weight,
        trading_days=cfg.trading_days,
    )
    raw_target = (blended["combined"] * vol_scalar).clip(
        -cfg.risk.max_gross_weight, cfg.risk.max_gross_weight
    )
    buffered = apply_turnover_buffer(raw_target, cfg.turnover_buffer)

    # Trade at t+1 on a signal formed from t's close. Everything above this
    # line may look at bar t; nothing below it may.
    desired = buffered.shift(1).fillna(0.0)

    # --- risk, costs, equity ----------------------------------------------
    risk = RiskManager(limits=cfg.risk)
    n = len(close)

    weights = np.zeros(n)
    net_returns = np.zeros(n)
    costs = np.zeros(n)
    binding: list[str] = []
    equity_curve = np.ones(n)

    equity = 1.0
    prev_weight = 0.0
    halted_at: pd.Timestamp | None = None
    halt_reason: str | None = None

    desired_values = desired.to_numpy(dtype=float)
    asset_ret_values = asset_returns.fillna(0.0).to_numpy(dtype=float)
    timestamps = close.index

    for i in range(n):
        risk.update_equity(equity, _as_datetime(timestamps[i]))
        decision = risk.apply(desired_values[i])
        weight = decision.weight

        turnover = weight - prev_weight
        cost = cfg.costs.cost_of(turnover, weight)

        # The position set at bar i earns bar i's asset return. `desired` was
        # already shifted, so this is not a second lookahead.
        gross = weight * asset_ret_values[i]
        net = gross - cost

        equity *= 1.0 + net

        weights[i] = weight
        costs[i] = cost
        net_returns[i] = net
        equity_curve[i] = equity
        binding.append(",".join(decision.binding))

        if decision.halted and halted_at is None:
            halted_at = timestamps[i]
            halt_reason = risk.halt_reason

        prev_weight = weight

    frame = pd.DataFrame(
        {
            "close": close,
            "signal_momentum": mom_signal,
            "signal_reversion": rev_signal,
            "regime": regime_frame["regime"],
            "confidence": regime_frame["confidence"],
            "hurst": regime_frame["hurst"],
            "variance_ratio": regime_frame["variance_ratio"],
            "w_momentum": blended["w_momentum"],
            "w_reversion": blended["w_reversion"],
            "combined": blended["combined"],
            "vol_scalar": vol_scalar,
            "target_weight": desired,
            "weight": weights,
            "cost": costs,
            "net_return": net_returns,
            "equity": equity_curve,
            "risk_binding": binding,
        },
        index=timestamps,
    )

    # Report only on bars the system could actually have traded.
    live = frame.iloc[warmup:]
    live_returns = live["net_return"]
    # Re-base equity so the reported curve starts at 1.0 at the first live bar.
    live_equity = (1.0 + live_returns).cumprod()

    stats = compute_stats(live_returns, cfg.periods_per_year, live["weight"])

    return BacktestResult(
        equity=live_equity,
        returns=live_returns,
        weights=live["weight"],
        frame=frame,
        stats=stats,
        halted_at=halted_at,
        halt_reason=halt_reason,
    )


def _as_datetime(value) -> pd.Timestamp:
    """Coerce an index entry to something with ``.date()``, for the risk manager."""
    if isinstance(value, pd.Timestamp):
        return value
    return pd.Timestamp(value)
