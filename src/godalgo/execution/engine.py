"""The autonomous live trading loop.

Runs the same decision pipeline the backtest runs -- signals, regime, blend,
sizing, risk -- and then routes the resulting target weight into orders. The
strategy code does not know whether it is being backtested or traded live, which
is the property that makes the backtest meaningful.

What the live path adds over the backtest, all of it about failure rather than
alpha:

* **A staleness watchdog.** If market data stops arriving, the bot is holding a
  position it can no longer reason about. It flattens. A silent feed is more
  dangerous than a bad signal, because nothing about it looks like an error.
* **Reconciliation on a schedule.** Believed position is checked against the
  venue. Drift beyond tolerance halts trading rather than being papered over.
* **An economic gate.** The router refuses trades whose expected edge does not
  clear their round-trip cost.
* **A kill switch that flattens and cancels.** Halting while orders rest is not
  halting -- the resting orders re-enter the market on the bot's behalf.

Autonomy is bounded by construction. The engine decides *when and whether* to
trade within limits it cannot alter: ``RiskLimits`` is passed in, is never part
of any search space, and is re-checked on every bar.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from godalgo.backtest.engine import compute_regime_series
from godalgo.data.stream import BarAggregator
from godalgo.execution.broker import BookSnapshot, Broker, FeeSchedule
from godalgo.execution.reconcile import Reconciler
from godalgo.execution.router import OrderRouter, RoutingConfig, RoutingDecision
from godalgo.execution.types import (
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
    TradingMode,
)
from godalgo.features.indicators import ewma_volatility, log_returns
from godalgo.features.session import SessionConfig, SessionProfile, fit_session_profile
from godalgo.portfolio.allocator import AllocationConfig, blend_signals
from godalgo.risk.limits import RiskLimits, RiskManager
from godalgo.strategies.base import Strategy

__all__ = ["EngineState", "LiveEngine", "LiveEngineConfig"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LiveEngineConfig:
    """Live loop configuration."""

    symbol: str = "BTC/USDT"
    bar_seconds: int = 60
    mode: TradingMode = TradingMode.DRY_RUN
    """Default is DRY_RUN. Escalation is always explicit."""

    target_annual_vol: float = 0.20
    regime_window: int = 250
    regime_refit_every: int = 24
    max_history_bars: int = 2000

    max_book_age: float = 10.0
    """Seconds before a top-of-book snapshot is too old to price against.

    A trade stream can keep flowing while the book stream dies. Pricing a
    passive order off a stale book posts into a market that has moved, which is
    adverse selection paid for voluntarily.
    """

    max_data_staleness: float = 180.0
    """Seconds without a completed bar before the bot flattens.

    Sized as a small multiple of the bar interval. Too tight and a quiet market
    triggers spurious exits; too loose and the bot holds risk into a feed
    outage it cannot see.
    """

    reconcile_every: float = 60.0
    """Seconds between position reconciliations."""

    edge_horizon_bars: float | None = None
    """Bars the expected-edge estimate is projected over.

    ``None`` -- the default -- derives it from the strategies' own
    ``expected_holding_bars``, blend-weighted by which one is driving. A
    hardcoded horizon shorter than the true holding period understates edge by
    its square root and refuses trades that would clear their costs.

    Inflating it has the opposite failure -- it defeats the cost gate -- so this
    is derived rather than tuned.
    """

    allocation: AllocationConfig = field(default_factory=AllocationConfig)

    session: SessionConfig | None = None
    """Overnight/session drift overlay. ``None`` disables it."""

    session_window: int = 2000
    session_refit_every_bars: int = 500

    @property
    def bars_per_day(self) -> float:
        return 86_400.0 / self.bar_seconds


@dataclass
class EngineState:
    """Live state, for logging and inspection."""

    target_weight: float = 0.0
    current_weight: float = 0.0
    equity: float = 0.0
    bars_seen: int = 0
    orders_sent: int = 0
    orders_refused: int = 0
    halted: bool = False
    halt_reason: str | None = None
    last_decision: RoutingDecision | None = None
    last_bar_at: datetime | None = None
    last_reconcile_at: datetime | None = None

    def snapshot(self) -> dict[str, object]:
        return {
            "target_weight": round(self.target_weight, 6),
            "current_weight": round(self.current_weight, 6),
            "equity": round(self.equity, 2),
            "bars_seen": self.bars_seen,
            "orders_sent": self.orders_sent,
            "orders_refused": self.orders_refused,
            "halted": self.halted,
            "halt_reason": self.halt_reason,
        }


class LiveEngine:
    """Autonomous execution loop.

    Args:
        broker: Where orders go. Determines what ``mode`` actually means.
        momentum: Trend strategy instance.
        reversion: Mean-reversion strategy instance.
        config: Live loop configuration.
        router: Order router; a default is built if omitted.
        risk_limits: Hard limits. Constructed from configuration, never from
            anything the strategy or search layer produced.
        fees: Fee schedule used by the router's cost model.
    """

    def __init__(
        self,
        broker: Broker,
        momentum: Strategy,
        reversion: Strategy,
        config: LiveEngineConfig | None = None,
        *,
        router: OrderRouter | None = None,
        risk_limits: RiskLimits | None = None,
        fees: FeeSchedule | None = None,
    ) -> None:
        self.config = config or LiveEngineConfig()
        self.broker = broker
        self.momentum = momentum
        self.reversion = reversion
        self.fees = fees or FeeSchedule()
        self.router = router or OrderRouter(RoutingConfig(), self.fees)
        self.risk = RiskManager(limits=risk_limits or RiskLimits())
        self.reconciler = Reconciler(broker)
        self.state = EngineState()

        self.bars = BarAggregator(
            interval_seconds=self.config.bar_seconds,
            max_bars=self.config.max_history_bars,
        )
        self._book: BookSnapshot | None = None
        self._warmup = max(momentum.warmup, reversion.warmup, self.config.regime_window)
        self._session: SessionProfile | None = None
        self._session_fitted_at_bar = -1

        self.target_clamp: Callable[[str, float], float] | None = None
        """Optional portfolio-level clamp, set by a supervisor.

        Applied *after* the risk layer, and only ever to reduce. The engine
        knows nothing about other symbols, so anything portfolio-wide -- gross
        exposure across the book, buying power shared between engines -- has to
        arrive through this hook rather than be inferred here.
        """

        self.buying_power: Callable[[str], float] | None = None
        """Optional buying-power source. Without it the engine sizes against
        full equity, which is correct alone and wrong in a fleet."""

        if self.config.mode is TradingMode.LIVE:
            logger.warning(
                "LiveEngine starting in LIVE mode on %s -- real orders", self.config.symbol
            )

    # -- market data ------------------------------------------------------

    def seed_history(self, bars: pd.DataFrame) -> None:
        """Prime with historical bars so the bot is not blind after a restart."""
        self.bars.seed(bars)
        logger.info("seeded %d historical bars (warm-up needs %d)", self.bars.n_complete, self._warmup)

    def on_book(self, book: BookSnapshot) -> None:
        """Record the latest top of book. Required before any order is priced."""
        self._book = book

    def ingest_tick(
        self, price: float, size: float = 0.0, moment: datetime | None = None
    ) -> bool:
        """Aggregate a trade. Returns True when a bar has just completed.

        Synchronous and cheap by design. A network read loop calls this on every
        trade, so it must never block: the expensive part (regime fitting, order
        routing) is a separate step a driver can run on its own task. Doing the
        decision inline here would stall the socket read and drop ticks under
        load -- exactly when the market is moving.
        """
        completed = self.bars.on_tick(price, size, moment)
        if completed is None:
            return False
        self.state.bars_seen += 1
        self.state.last_bar_at = datetime.now(UTC)
        return True

    async def on_tick(
        self, price: float, size: float = 0.0, moment: datetime | None = None
    ) -> RoutingDecision | None:
        """Feed a trade and decide inline.

        Convenience for tests and single-threaded use. Live drivers should call
        ``ingest_tick`` and ``on_bar_close`` separately so a slow decision cannot
        stall market-data ingestion.

        Returns a decision only on a completed bar. The forming bar never
        triggers a decision -- acting on it would mean trading a close that does
        not exist yet, which is precisely the lookahead the backtest forbids.
        """
        if not self.ingest_tick(price, size, moment):
            return None
        return await self.on_bar_close()

    # -- the decision -----------------------------------------------------

    async def on_bar_close(self) -> RoutingDecision | None:
        """Run one full decision cycle. Called once per completed bar."""
        if self.state.halted:
            return None

        frame = self.bars.frame()
        if len(frame) <= self._warmup:
            logger.debug("warming up: %d/%d bars", len(frame), self._warmup)
            return None

        equity = await self.broker.equity()
        now = datetime.now(UTC)
        self.state.equity = equity

        # Risk state first: a halt must take effect before any new target is
        # computed, not after.
        self.risk.update_equity(equity, now)
        if self.risk.halted:
            await self.halt(self.risk.halt_reason or "risk limit")
            return None

        if await self._maybe_reconcile(now):
            return None

        target, edge_bps = self._compute_target(frame)
        decision = self.risk.apply(target)
        weight = decision.weight

        # Portfolio clamp last, so a fleet-wide limit can only tighten what the
        # per-symbol risk layer already allowed, never loosen it.
        if self.target_clamp is not None:
            clamped = self.target_clamp(self.config.symbol, weight)
            if abs(clamped) < abs(weight):
                logger.info(
                    "portfolio clamped %s target %.4f -> %.4f",
                    self.config.symbol, weight, clamped,
                )
            weight = clamped

        self.state.target_weight = weight

        if decision.was_reduced:
            logger.info("risk reduced target %.4f -> %.4f (%s)",
                        decision.requested, decision.weight, ",".join(decision.binding))

        return await self._route(weight, equity, edge_bps)

    def _compute_target(self, frame: pd.DataFrame) -> tuple[float, float]:
        """Target weight and expected edge, from the same pipeline as backtest.

        Returns:
            ``(target_weight, expected_edge_bps)``. The edge estimate converts
            conviction into an expected move by scaling trailing volatility over
            ``edge_horizon_bars`` -- deliberately conservative, because it is
            what the router's cost gate is tested against.
        """
        mom_signal = self.momentum.generate(frame)
        rev_signal = self.reversion.generate(frame)

        regimes = compute_regime_series(
            frame, self.config.symbol, self.config.regime_window, self.config.regime_refit_every
        )

        session_tilt = self._session_tilt(frame)
        blended = blend_signals(
            mom_signal, rev_signal, regimes["regime"], regimes["confidence"],
            self.config.allocation,
            session_tilt=session_tilt,
            tilt_weight=self.config.session.tilt_weight if self.config.session else 0.0,
        )
        conviction = float(blended["combined"].iloc[-1])

        # Horizon from whichever strategy is actually driving this bar.
        mom_contrib = abs(float(blended["momentum"].iloc[-1]))
        rev_contrib = abs(float(blended["reversion"].iloc[-1]))
        if self.config.edge_horizon_bars is not None:
            horizon = float(self.config.edge_horizon_bars)
        elif mom_contrib + rev_contrib > 0:
            horizon = (
                mom_contrib * self.momentum.expected_holding_bars
                + rev_contrib * self.reversion.expected_holding_bars
            ) / (mom_contrib + rev_contrib)
        else:
            horizon = self.momentum.expected_holding_bars
        horizon = max(1.0, horizon)

        returns = log_returns(frame["close"])
        bar_vol = ewma_volatility(returns, halflife=30.0).iloc[-1]
        if not np.isfinite(bar_vol) or bar_vol <= 0:
            return 0.0, 0.0

        annual_vol = float(bar_vol * np.sqrt(self.config.bars_per_day * 365.0))
        vol_scalar = min(
            self.config.target_annual_vol / annual_vol if annual_vol > 0 else 0.0,
            self.risk.limits.max_gross_weight,
        )
        target = float(np.clip(conviction * vol_scalar,
                               -self.risk.limits.max_gross_weight,
                               self.risk.limits.max_gross_weight))

        # Expected move over the holding horizon, in bps.
        edge_bps = abs(conviction) * float(bar_vol) * np.sqrt(horizon) * 1e4

        # A session drift is an expected move in its own right, so it belongs in
        # the number the cost gate tests -- but only when it points the same way
        # as the position. A tilt opposing the trade is not extra edge.
        if self._session is not None and target != 0.0:
            drift_bps = self._session.expected_drift_bps(frame.index[-1], horizon)
            if np.sign(drift_bps) == np.sign(target):
                edge_bps += abs(drift_bps)

        return target, edge_bps

    def _session_tilt(self, frame: pd.DataFrame) -> pd.Series | None:
        """Session tilt for the current history, refit on a bar cadence.

        Refit by bar count rather than wall clock so that a restart mid-session,
        or a gap in the feed, cannot change how often the profile is estimated
        -- which would make live behaviour depend on uptime rather than on data.
        """
        if self.config.session is None:
            return None

        n = len(frame)
        if n < self.config.session_window:
            return None

        due = self._session_fitted_at_bar < 0 or (
            n - self._session_fitted_at_bar >= self.config.session_refit_every_bars
        )
        if due:
            returns = log_returns(frame["close"]).iloc[-self.config.session_window :]
            self._session = fit_session_profile(returns, self.config.session)
            self._session_fitted_at_bar = n
            logger.info("refit session profile at bar %d\n%s", n, self._session.summary(3))

        if self._session is None:
            return None
        return self._session.tilt_series(frame.index)

    async def _route(self, target: float, equity: float, edge_bps: float) -> RoutingDecision | None:
        if self._book is None:
            logger.warning("no book snapshot; cannot price an order")
            return None

        book_age = (datetime.now(UTC) - self._book.timestamp).total_seconds()
        if book_age > self.config.max_book_age:
            logger.warning(
                "book is %.1fs old (max %.1fs); standing down rather than "
                "pricing against stale liquidity", book_age, self.config.max_book_age
            )
            self.state.orders_refused += 1
            return None

        position = await self.broker.position(self.config.symbol)
        mark = self._book.mid
        self.state.current_weight = (position.quantity * mark / equity) if equity > 0 else 0.0

        # In a fleet each engine gets a share, not the whole account. Sizing
        # against full equity would have every engine believe it owns all of it.
        sizing_equity = equity
        if self.buying_power is not None:
            allocated = self.buying_power(self.config.symbol)
            if allocated > 0:
                sizing_equity = min(equity, allocated)

        decision = self.router.decide(
            symbol=self.config.symbol,
            target_weight=target,
            current_weight=self.state.current_weight,
            equity=sizing_equity,
            book=self._book,
            expected_edge_bps=edge_bps,
        )
        self.state.last_decision = decision

        if not decision.should_trade:
            self.state.orders_refused += 1
            logger.debug("no order: %s", decision.reason)
            return decision

        if self.config.mode is TradingMode.DRY_RUN:
            logger.info(
                "DRY RUN %s %s %.8f @ %s (edge %.1fbps vs cost %.1fbps)",
                decision.order.side.value, decision.order.symbol, decision.order.amount,
                decision.order.price, decision.expected_edge_bps, decision.round_trip_cost_bps,
            )

        assert decision.order is not None
        self.router.mark_submitted(decision.order)
        try:
            result = await self.broker.submit(decision.order)
        finally:
            # Always clear the in-flight marker. Leaking it on an exception
            # blocks the symbol permanently and the bot silently stops trading.
            self.router.mark_settled(self.config.symbol)

        self.state.orders_sent += 1

        if result.status is OrderStatus.UNKNOWN:
            # The order may be live. Do not send anything else until the venue
            # has been asked what actually happened.
            await self.halt(f"ambiguous order outcome: {result.error}")
        elif result.status is OrderStatus.REJECTED:
            logger.warning("order rejected: %s", result.error)

        return decision

    # -- safety -----------------------------------------------------------

    async def _maybe_reconcile(self, now: datetime) -> bool:
        """Reconcile if due. Returns True if a drift halt was triggered."""
        last = self.state.last_reconcile_at
        if last is not None and (now - last).total_seconds() < self.config.reconcile_every:
            return False

        self.state.last_reconcile_at = now
        equity = self.state.equity
        mark = self._book.mid if self._book else None
        if mark is None or equity <= 0:
            return False

        believed_quantity = self.state.current_weight * equity / mark
        result = await self.reconciler.check(self.config.symbol, believed_quantity)
        if not result.within_tolerance:
            await self.halt(f"position drift: {result.describe()}")
            return True
        return False

    def swap_strategies(self, momentum: Strategy, reversion: Strategy) -> None:
        """Replace the running strategies.

        Called by Autopilot after a candidate clears the promotion gate. The
        engine owns the swap so it can re-derive warm-up: a new parameter set
        may need more history than the old one, and signalling on a window the
        new configuration has not warmed up on would be trading a strategy that
        has not yet seen enough data to have an opinion.
        """
        self.momentum = momentum
        self.reversion = reversion
        self._warmup = max(momentum.warmup, reversion.warmup, self.config.regime_window)
        logger.warning(
            "strategies swapped: %s | %s (warm-up now %d bars)",
            momentum, reversion, self._warmup,
        )

    def is_flat(self) -> bool:
        """Whether the engine believes it holds no position.

        Read from the last routed weight rather than from the venue, because
        this gates a parameter swap rather than an order -- and a swap must not
        block on a network round trip inside the trading loop.
        """
        return abs(self.state.current_weight) < 1e-9

    def history(self) -> pd.DataFrame:
        """Completed bars, for the autopilot's search."""
        return self.bars.frame()

    async def check_staleness(self, now: datetime | None = None) -> bool:
        """Flatten if market data has stopped. Returns True if it halted.

        Must be driven by a timer, not by incoming data -- a watchdog that only
        runs when data arrives cannot detect data not arriving.
        """
        now = now or datetime.now(UTC)
        age = self.bars.last_bar_age(now)
        if age is None:
            return False
        if age > timedelta(seconds=self.config.max_data_staleness):
            await self.halt(f"market data stale for {age.total_seconds():.0f}s")
            return True
        return False

    async def halt(self, reason: str) -> None:
        """Stop trading: cancel resting orders, then flatten.

        Cancel-before-flatten is the required order. Flattening while orders
        rest lets those orders re-open a position on a halted bot.
        """
        if self.state.halted:
            return
        self.state.halted = True
        self.state.halt_reason = reason
        logger.error("HALT: %s", reason)

        try:
            cancelled = await self.broker.cancel_all(self.config.symbol)
            logger.info("cancelled %d resting orders", cancelled)
        except Exception:
            logger.exception("cancel_all failed during halt")

        try:
            await self.flatten()
        except Exception:
            logger.exception("flatten failed during halt -- POSITION MAY BE OPEN")

    async def flatten(self) -> None:
        """Close any open position with a marketable reduce-only order."""
        position = await self.broker.position(self.config.symbol)
        if position.is_flat:
            logger.info("already flat")
            return
        if self._book is None:
            logger.error("cannot flatten without a book snapshot")
            return

        side = OrderSide.SELL if position.quantity > 0 else OrderSide.BUY
        order = Order(
            symbol=self.config.symbol,
            side=side,
            amount=abs(position.quantity),
            order_type=OrderType.MARKET,
            # Crossing is correct here. An exit that does not fill is not an
            # exit, and the fee saved by posting passively is irrelevant next
            # to the risk of remaining exposed.
            time_in_force=TimeInForce.IOC,
            reduce_only=True,
        )
        result = await self.broker.submit(order)
        logger.warning(
            "flatten %s %.8f -> %s", side.value, order.amount, result.status.value
        )

    async def run_watchdog(self, interval: float = 10.0) -> None:
        """Timer loop for staleness detection. Run as a background task."""
        while not self.state.halted:
            await asyncio.sleep(interval)
            await self.check_staleness()
