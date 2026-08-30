"""The promotion gate and its audit ledger.

This is the component that makes "self-improving" safe rather than merely
ambitious. A candidate configuration replaces the incumbent only by clearing
every one of a fixed set of statistical hurdles, all of them evaluated on
out-of-sample data the search never saw.

The gate is adversarial by construction: its job is to *reject*. A search loop
proposes; this module assumes the proposal is noise until the evidence says
otherwise. Each check closes a specific, documented failure mode:

* **Out-of-sample Sharpe floor** -- the basic bar. Cheap to pass, so it is never
  sufficient on its own.
* **Deflated Sharpe** -- corrects for having tried N configurations. This is what
  stops the loop rewarding itself for searching harder.
* **PBO** -- asks whether the *selection procedure* generalises at all. A
  candidate can post a fine OOS Sharpe while sitting inside a search whose
  winners are systematically noise; PBO catches that and DSR does not.
* **Drawdown ceiling** -- a Sharpe improvement bought with a deeper hole is not
  an improvement, and Sharpe alone is blind to path.
* **Parameter stability** -- parameters that jump between folds are being refit
  to noise, whatever the aggregate curve says.
* **Minimum relative improvement** -- churning the live configuration for a
  rounding-error gain costs real turnover and forfeits a known quantity.

Every decision, pass or fail, is appended to a JSON ledger. The ledger is the
self-*correction* half of the design: it records what was tried, what the
evidence was, and what was decided, so a later run can see that a direction has
already been explored and rejected.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from godalgo.backtest.metrics import (
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting,
)
from godalgo.evolve.walkforward import WalkForwardResult

__all__ = ["GateResult", "PromotionCriteria", "PromotionLedger", "evaluate_candidate"]


@dataclass(frozen=True, slots=True)
class PromotionCriteria:
    """Thresholds a candidate must clear. Not tunable by the search loop.

    These are configuration, deliberately held outside ``StrategyParams``. If
    the optimiser could adjust its own acceptance thresholds, the gate would be
    decorative.
    """

    min_oos_sharpe: float = 0.5
    """Annualised OOS Sharpe floor."""

    min_deflated_sharpe: float = 0.95
    """Probability the Sharpe is real, after correcting for N trials."""

    max_pbo: float = 0.35
    """Ceiling on the probability of backtest overfitting.

    0.5 is a coin flip -- selection carrying no information at all. 0.35 demands
    the procedure be meaningfully better than chance without requiring the
    near-perfection that would reject everything.
    """

    max_drawdown: float = 0.25
    """OOS drawdown ceiling, as a positive fraction."""

    max_parameter_instability: float = 0.60
    """Ceiling on mean coefficient of variation of parameters across folds."""

    min_relative_improvement: float = 0.10
    """Required OOS Sharpe gain over the incumbent, as a fraction.

    Applies only when an incumbent exists. Guards against replacing a known
    configuration with a statistically indistinguishable one.
    """

    min_oos_observations: int = 500
    """Minimum stitched OOS bars before any promotion is considered."""

    def __post_init__(self) -> None:
        if not 0 <= self.min_deflated_sharpe <= 1:
            raise ValueError("min_deflated_sharpe must be in [0, 1]")
        if not 0 <= self.max_pbo <= 1:
            raise ValueError("max_pbo must be in [0, 1]")
        if not 0 < self.max_drawdown < 1:
            raise ValueError("max_drawdown must be in (0, 1)")


@dataclass(frozen=True, slots=True)
class GateResult:
    """Outcome of evaluating one candidate."""

    promoted: bool
    checks: dict[str, bool]
    """Per-check pass/fail, keyed by criterion name."""

    failures: tuple[str, ...]
    """Human-readable reasons, empty when promoted."""

    oos_sharpe: float
    deflated_sharpe: float
    pbo: float
    oos_max_drawdown: float
    parameter_instability: float
    n_trials: int
    n_oos_observations: int
    incumbent_sharpe: float | None = None
    params: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        verdict = "PROMOTED" if self.promoted else "REJECTED"
        lines = [
            f"{verdict}",
            f"  OOS Sharpe        {self.oos_sharpe:>7.2f}"
            + (f"  (incumbent {self.incumbent_sharpe:.2f})" if self.incumbent_sharpe is not None else ""),
            f"  Deflated Sharpe   {self.deflated_sharpe:>7.3f}",
            f"  PBO               {self.pbo:>7.3f}",
            f"  OOS max drawdown  {self.oos_max_drawdown:>7.2%}",
            f"  Param instability {self.parameter_instability:>7.3f}",
            f"  Trials            {self.n_trials:>7d}",
            f"  OOS bars          {self.n_oos_observations:>7d}",
        ]
        if self.failures:
            lines.append("  failed: " + "; ".join(self.failures))
        return "\n".join(lines)


def evaluate_candidate(
    result: WalkForwardResult,
    criteria: PromotionCriteria | None = None,
    incumbent_sharpe: float | None = None,
    *,
    pbo_splits: int = 10,
) -> GateResult:
    """Run a walk-forward result through every promotion check.

    All checks are evaluated even after one fails, so the ledger records the
    complete picture rather than stopping at the first problem. Knowing a
    candidate failed on three axes rather than one is what tells a later round
    whether the direction is worth revisiting.

    Args:
        result: Output of ``walk_forward_evaluate``.
        criteria: Thresholds. Defaults used if omitted.
        incumbent_sharpe: OOS Sharpe of the configuration currently live, if
            any. When None the relative-improvement check is skipped.
        pbo_splits: Blocks for the CSCV estimate; must be even.

    Returns:
        A ``GateResult``. ``promoted`` is True only if every check passed.
    """
    crit = criteria or PromotionCriteria()

    oos_sharpe = result.oos_stats.sharpe
    oos_dd = result.oos_stats.max_drawdown
    n_obs = result.oos_stats.n_observations

    # Variance of Sharpe across candidates, in per-observation units -- the
    # scale the DSR formula expects.
    trial_sharpes = _per_observation_sharpes(result.fold_trial_returns)
    sharpe_variance = float(np.var(trial_sharpes, ddof=1)) if trial_sharpes.size > 1 else 0.0

    dsr = deflated_sharpe_ratio(
        result.oos_returns, n_trials=max(result.n_trials, 1), sharpe_variance=sharpe_variance
    )

    pbo = (
        probability_of_backtest_overfitting(result.fold_trial_returns, n_splits=pbo_splits)
        if result.fold_trial_returns.shape[1] >= 2
        else 0.5
    )

    stability = result.parameter_stability
    instability = float(np.mean(list(stability.values()))) if stability else 0.0

    checks: dict[str, bool] = {
        "oos_observations": n_obs >= crit.min_oos_observations,
        "oos_sharpe": oos_sharpe >= crit.min_oos_sharpe,
        "deflated_sharpe": dsr >= crit.min_deflated_sharpe,
        "pbo": pbo <= crit.max_pbo,
        "max_drawdown": oos_dd <= crit.max_drawdown,
        "parameter_stability": instability <= crit.max_parameter_instability,
    }

    if incumbent_sharpe is not None:
        required = incumbent_sharpe * (1.0 + crit.min_relative_improvement)
        # An incumbent at or below zero cannot set a meaningful relative bar;
        # fall back to the absolute floor so a negative incumbent is not an
        # accidentally easy target.
        if incumbent_sharpe <= 0:
            required = max(crit.min_oos_sharpe, 0.0)
        checks["beats_incumbent"] = oos_sharpe >= required

    failures = tuple(
        _describe_failure(name, crit, oos_sharpe, dsr, pbo, oos_dd, instability, n_obs, incumbent_sharpe)
        for name, passed in checks.items()
        if not passed
    )

    return GateResult(
        promoted=all(checks.values()),
        checks=checks,
        failures=failures,
        oos_sharpe=oos_sharpe,
        deflated_sharpe=dsr,
        pbo=pbo,
        oos_max_drawdown=oos_dd,
        parameter_instability=instability,
        n_trials=result.n_trials,
        n_oos_observations=n_obs,
        incumbent_sharpe=incumbent_sharpe,
        params=result.chosen_params[-1] if result.chosen_params else {},
    )


def _per_observation_sharpes(trial_returns: pd.DataFrame) -> np.ndarray:
    """Per-observation Sharpe of each candidate column, NaNs dropped."""
    if trial_returns.empty:
        return np.array([])
    data = trial_returns.to_numpy(dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.nanmean(data, axis=0)
        sd = np.nanstd(data, axis=0, ddof=1)
        sharpes = np.where(sd > 0, mean / sd, np.nan)
    return sharpes[np.isfinite(sharpes)]


def _describe_failure(
    name: str,
    crit: PromotionCriteria,
    sharpe: float,
    dsr: float,
    pbo: float,
    dd: float,
    instability: float,
    n_obs: int,
    incumbent: float | None,
) -> str:
    match name:
        case "oos_observations":
            return f"only {n_obs} OOS bars, need {crit.min_oos_observations}"
        case "oos_sharpe":
            return f"OOS Sharpe {sharpe:.2f} < {crit.min_oos_sharpe}"
        case "deflated_sharpe":
            return f"deflated Sharpe {dsr:.3f} < {crit.min_deflated_sharpe} (likely selection noise)"
        case "pbo":
            return f"PBO {pbo:.3f} > {crit.max_pbo} (selection does not generalise)"
        case "max_drawdown":
            return f"OOS drawdown {dd:.2%} > {crit.max_drawdown:.2%}"
        case "parameter_stability":
            return f"parameter instability {instability:.3f} > {crit.max_parameter_instability}"
        case "beats_incumbent":
            return f"OOS Sharpe {sharpe:.2f} does not beat incumbent {incumbent:.2f} by {crit.min_relative_improvement:.0%}"
        case _:
            return name


class PromotionLedger:
    """Append-only JSONL record of every promotion decision.

    Append-only matters. A ledger that could be rewritten would let a later run
    erase the evidence that a direction was already tried and rejected, which is
    exactly the memory the loop needs to avoid rediscovering the same noise.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        result: GateResult,
        *,
        symbol: str,
        note: str | None = None,
    ) -> None:
        """Append one decision."""
        entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "symbol": symbol,
            "promoted": result.promoted,
            "checks": result.checks,
            "failures": list(result.failures),
            "metrics": {
                "oos_sharpe": result.oos_sharpe,
                "deflated_sharpe": result.deflated_sharpe,
                "pbo": result.pbo,
                "oos_max_drawdown": result.oos_max_drawdown,
                "parameter_instability": result.parameter_instability,
                "n_trials": result.n_trials,
                "n_oos_observations": result.n_oos_observations,
                "incumbent_sharpe": result.incumbent_sharpe,
            },
            "params": _jsonable(result.params),
            "note": note,
        }
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, sort_keys=True) + "\n")

    def entries(self) -> list[dict[str, Any]]:
        """Read the full ledger, oldest first."""
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]

    def last_promoted(self, symbol: str | None = None) -> dict[str, Any] | None:
        """Most recent promotion, optionally filtered by symbol."""
        for entry in reversed(self.entries()):
            if entry.get("promoted") and (symbol is None or entry.get("symbol") == symbol):
                return entry
        return None

    def rejection_summary(self) -> dict[str, int]:
        """Count of how often each check has blocked a promotion.

        The most useful diagnostic the ledger offers. If PBO blocks nearly
        everything, the search is too wide for the data. If it is always the
        relative-improvement check, the incumbent is genuinely good and the loop
        should be left alone rather than pushed harder.
        """
        counts: dict[str, int] = {}
        for entry in self.entries():
            if entry.get("promoted"):
                continue
            for name, passed in entry.get("checks", {}).items():
                if not passed:
                    counts[name] = counts.get(name, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True))


def _jsonable(obj: Any) -> Any:
    """Coerce numpy scalars so ``json.dumps`` does not choke on them."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj
