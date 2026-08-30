"""Autonomous retuning: learning while trading.

The search machinery in this package is complete but, on its own, inert -- it
only runs when a human types ``godalgo evolve``. A bot that trades autonomously
but is retuned by hand is not self-improving; it is a manual system with an
automated trigger.

``Autopilot`` closes that loop. On a schedule it refits parameters against
recent history, puts the winner through the same promotion gate a human run
would, and swaps the live configuration only if it passes.

Three rules make that safe rather than merely automatic:

**The gate is unchanged.** Nothing here relaxes deflated Sharpe, PBO, drawdown,
parameter stability, or improvement-over-incumbent. An automated proposer that
could also lower its own bar would be a machine for promoting noise on a timer.

**Swaps happen flat, or not at all.** Changing strategy parameters underneath an
open position means the exit is governed by a different model than the entry --
the position becomes an orphan no rule accounts for. Autopilot waits for flat.

**Search never touches risk.** ``RiskLimits`` is not in any search space, and
Autopilot holds no reference to it. The bot can learn how to trade; it cannot
learn to take more risk than it was configured to take.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from godalgo.backtest.engine import BacktestConfig
from godalgo.evolve.promotion import (
    GateResult,
    PromotionCriteria,
    PromotionLedger,
    evaluate_candidate,
)
from godalgo.evolve.search import SearchConfig, propose_candidates
from godalgo.evolve.walkforward import walk_forward_evaluate
from godalgo.strategies.base import StrategyParams
from godalgo.strategies.mean_reversion import MeanReversionParams, MeanReversionStrategy
from godalgo.strategies.momentum import MomentumParams, MomentumStrategy

__all__ = ["Autopilot", "AutopilotConfig", "AutopilotState"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AutopilotConfig:
    """When and how the bot retunes itself."""

    enabled: bool = False
    """Off by default. Self-modification is opt-in, like live trading."""

    interval_hours: float = 24.0
    """Hours between search rounds.

    Deliberately long. Retuning hourly would mean fitting to a handful of new
    bars, and every round adds to the trial count the deflated Sharpe corrects
    against -- searching more often makes the bar higher, not the bot smarter.
    """

    min_bars: int = 6000
    """History required before a round runs at all."""

    candidates: int = 16
    train_size: int = 2500
    test_size: int = 700
    max_rounds_per_day: int = 2
    """Ceiling on rounds regardless of interval, so a restart loop cannot
    burn through the trial budget."""

    require_flat: bool = True
    """Only swap parameters while flat.

    Swapping under an open position leaves it governed on exit by a model that
    did not open it.
    """

    criteria: PromotionCriteria = field(default_factory=PromotionCriteria)
    search: SearchConfig = field(default_factory=SearchConfig)
    ledger_path: Path = field(default_factory=lambda: Path("data/promotion_ledger.jsonl"))

    def __post_init__(self) -> None:
        if self.interval_hours <= 0:
            raise ValueError("interval_hours must be positive")
        if self.max_rounds_per_day < 1:
            raise ValueError("max_rounds_per_day must be at least 1")


@dataclass
class AutopilotState:
    """What the loop has done, for the UI and for diagnosis."""

    rounds_run: int = 0
    promotions: int = 0
    rejections: int = 0
    last_run_at: datetime | None = None
    last_result: GateResult | None = None
    pending_swap: bool = False
    """A candidate passed but is waiting for a flat book."""

    incumbent_sharpe: float | None = None
    blocked_reason: str | None = None

    def snapshot(self) -> dict[str, object]:
        return {
            "rounds_run": self.rounds_run,
            "promotions": self.promotions,
            "rejections": self.rejections,
            "last_run_at": self.last_run_at.isoformat() if self.last_run_at else None,
            "pending_swap": self.pending_swap,
            "incumbent_sharpe": self.incumbent_sharpe,
            "blocked_reason": self.blocked_reason,
            "last_verdict": (
                "promoted" if self.last_result and self.last_result.promoted
                else "rejected" if self.last_result else None
            ),
            "last_failures": list(self.last_result.failures) if self.last_result else [],
        }


class Autopilot:
    """Periodically refits parameters and promotes only what clears the gate.

    Args:
        config: Scheduling and gate configuration.
        history: Callable returning the bar history to search over. Supplied as
            a callable rather than a frame so each round sees current data
            without Autopilot owning the feed.
        apply_params: Callable invoked with an approved ``(momentum, reversion)``
            pair. The engine owns the swap; Autopilot only decides.
        is_flat: Callable reporting whether the book is currently flat.
        backtest_config: Engine configuration for the search.
        symbol: Label for the ledger.
    """

    def __init__(
        self,
        config: AutopilotConfig,
        history: Callable[[], pd.DataFrame],
        apply_params: Callable[[StrategyParams, StrategyParams], None],
        is_flat: Callable[[], bool],
        *,
        backtest_config: BacktestConfig | None = None,
        symbol: str = "UNKNOWN",
        momentum: MomentumParams | None = None,
        reversion: MeanReversionParams | None = None,
    ) -> None:
        self.config = config
        self.history = history
        self.apply_params = apply_params
        self.is_flat = is_flat
        self.backtest_config = backtest_config or BacktestConfig()
        self.symbol = symbol
        self.state = AutopilotState()

        self.incumbent_momentum = momentum or MomentumParams()
        self.incumbent_reversion = reversion or MeanReversionParams()
        self.ledger = PromotionLedger(config.ledger_path)

        self._approved: tuple[StrategyParams, StrategyParams] | None = None
        self._round_times: list[datetime] = []

    # -- scheduling --------------------------------------------------------

    async def run(self, poll_seconds: float = 300.0) -> None:
        """Background loop. Run as a task alongside the driver."""
        if not self.config.enabled:
            logger.info("autopilot disabled; parameters will not change on their own")
            return

        logger.warning(
            "autopilot ENABLED: retuning every %.1fh, gated on OOS evidence",
            self.config.interval_hours,
        )
        while True:
            try:
                await self.maybe_run()
                self.apply_pending()
            except asyncio.CancelledError:
                raise
            except Exception:
                # A failed search must never take down the trading loop. The
                # bot keeps trading its current configuration.
                self.state.blocked_reason = "search raised; see logs"
                logger.exception("autopilot round failed; continuing on current parameters")
            await asyncio.sleep(poll_seconds)

    def due(self, now: datetime | None = None) -> bool:
        """Whether a round is scheduled and within the daily budget."""
        now = now or datetime.now(UTC)
        if self.state.last_run_at is not None:
            elapsed = (now - self.state.last_run_at).total_seconds() / 3600.0
            if elapsed < self.config.interval_hours:
                return False

        cutoff = now - timedelta(days=1)
        self._round_times = [t for t in self._round_times if t > cutoff]
        if len(self._round_times) >= self.config.max_rounds_per_day:
            self.state.blocked_reason = "daily search budget reached"
            return False
        return True

    async def maybe_run(self, now: datetime | None = None) -> GateResult | None:
        """Run a round if one is due. Returns the gate result, if any."""
        if not self.due(now):
            return None

        bars = self.history()
        if bars is None or len(bars) < self.config.min_bars:
            self.state.blocked_reason = (
                f"need {self.config.min_bars} bars, have {0 if bars is None else len(bars)}"
            )
            return None

        # The search is CPU-bound and would otherwise stall the event loop --
        # which here means stalling market-data ingestion and order routing.
        return await asyncio.to_thread(self._run_round, bars, now or datetime.now(UTC))

    # -- the round ---------------------------------------------------------

    def _run_round(self, bars: pd.DataFrame, now: datetime) -> GateResult | None:
        logger.info("autopilot: searching %d candidates over %d bars",
                    self.config.search.n_candidates, len(bars))

        candidates = propose_candidates(
            self.incumbent_momentum, self.incumbent_reversion, self.config.search,
        )
        result = walk_forward_evaluate(
            bars, candidates,
            lambda m, r: (MomentumStrategy(m), MeanReversionStrategy(r)),
            self.backtest_config,
            train_size=self.config.train_size,
            test_size=self.config.test_size,
            symbol=self.symbol,
        )

        gate = evaluate_candidate(
            result, self.config.criteria, incumbent_sharpe=self.state.incumbent_sharpe,
        )

        self.state.rounds_run += 1
        self.state.last_run_at = now
        self.state.last_result = gate
        self.state.blocked_reason = None
        self._round_times.append(now)
        self.ledger.record(gate, symbol=self.symbol, note="autopilot")

        if not gate.promoted:
            self.state.rejections += 1
            logger.info("autopilot: rejected — %s", "; ".join(gate.failures))
            return gate

        # Recover the winning parameter pair from the fold that produced it.
        chosen = result.chosen_params[-1] if result.chosen_params else {}
        try:
            momentum = self.incumbent_momentum.replace(
                **{k[4:]: v for k, v in chosen.items() if k.startswith("mom_")}
            )
            reversion = self.incumbent_reversion.replace(
                **{k[4:]: v for k, v in chosen.items() if k.startswith("rev_")}
            )
        except (ValueError, TypeError) as exc:
            # A parameter set that cannot be reconstructed is not promoted.
            # Failing closed keeps the bot on a configuration known to be valid.
            self.state.rejections += 1
            logger.error("autopilot: winning parameters rejected as invalid: %s", exc)
            return gate

        self._approved = (momentum, reversion)
        self.state.pending_swap = True
        self.state.incumbent_sharpe = gate.oos_sharpe
        logger.warning(
            "autopilot: candidate PASSED (OOS Sharpe %.2f, DSR %.3f, PBO %.3f) — "
            "awaiting a flat book to swap",
            gate.oos_sharpe, gate.deflated_sharpe, gate.pbo,
        )
        return gate

    def apply_pending(self) -> bool:
        """Swap in an approved configuration once the book is flat.

        Returns whether a swap happened. Called on every poll, so an approved
        candidate lands at the first flat moment rather than waiting a full
        interval.
        """
        if self._approved is None:
            return False
        if self.config.require_flat and not self.is_flat():
            return False

        momentum, reversion = self._approved
        self.apply_params(momentum, reversion)
        self.incumbent_momentum = momentum
        self.incumbent_reversion = reversion
        self._approved = None
        self.state.pending_swap = False
        self.state.promotions += 1
        logger.warning("autopilot: parameters swapped — %s | %s",
                       momentum.to_dict(), reversion.to_dict())
        return True
