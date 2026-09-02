"""What a retail account can actually reach, as opposed to what it should hold.

``godalgo.portfolio.book`` decides the right allocation. This decides whether
that allocation is reachable this week, which on a small account is a tighter
constraint than any cost model and is usually left out entirely.

Two rules, both structural rather than economic:

* **Settlement.** In a cash account, proceeds from a sale are not spendable
  until they settle, currently the next business day. An allocator that treats
  the sale price as immediately available buying power will produce a target
  the account cannot fund, and the broker will reject the order that completes
  the rotation -- leaving the book half-rebalanced, which is a position nobody
  chose.

* **The day-trade budget at portfolio level.** ``venue_rules`` refuses the
  fourth day trade one order at a time. That is necessary and not sufficient:
  a rebalance that requires six round trips this week is already impossible
  when it is *planned*, and discovering it on the fourth order means the first
  three were spent on a rotation that cannot complete. Checking the whole
  target at once is the difference between a plan that fits and a plan that
  strands.

The output is never a silent trim. A target that cannot be reached comes back
with the reason and with what *can* be reached, because "rejected" tells an
operator nothing about what to do next.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

__all__ = ["AccountState", "SettlementPlan", "plan_rebalance"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AccountState:
    """What the broker says is available, not what we computed."""

    equity: float
    cash: float = 0.0
    settled_cash: float = 0.0
    """Spendable now. In a margin account this equals cash; in a cash account
    it lags every sale by a settlement cycle."""

    is_margin: bool = True
    day_trades_left: int = 99
    pdt_constrained: bool = False

    @property
    def buying_power(self) -> float:
        """What can actually be deployed today."""
        return self.cash if self.is_margin else self.settled_cash

    @property
    def unsettled(self) -> float:
        return max(0.0, self.cash - self.settled_cash)


@dataclass(frozen=True, slots=True)
class SettlementPlan:
    """A rebalance that fits, and an account of what was left out."""

    reachable: dict[str, float]
    deferred: dict[str, float] = field(default_factory=dict)
    """Targets that could not be funded or afforded in day trades this week."""

    day_trades_needed: int = 0
    day_trades_available: int = 99
    cash_needed: float = 0.0
    cash_available: float = 0.0
    reasons: tuple[str, ...] = ()

    @property
    def complete(self) -> bool:
        return not self.deferred

    def to_dict(self) -> dict[str, Any]:
        return {
            "reachable": self.reachable, "deferred": self.deferred,
            "complete": self.complete,
            "day_trades_needed": self.day_trades_needed,
            "day_trades_available": self.day_trades_available,
            "cash_needed": self.cash_needed,
            "cash_available": self.cash_available,
            "reasons": list(self.reasons),
        }


def plan_rebalance(
    targets: dict[str, float],
    current: dict[str, float],
    account: AccountState,
    *,
    crypto: set[str] | None = None,
    same_day_exits: set[str] | None = None,
) -> SettlementPlan:
    """Work out which parts of a target allocation can actually be reached.

    Args:
        targets: Desired weights by symbol.
        current: Present weights by symbol.
        account: The broker's own view of the account.
        crypto: Symbols exempt from settlement and the day-trade rule. Crypto
            settles immediately and is not a security for FINRA's purposes,
            which is most of why it is the sensible thing to rotate on a small
            account.
        same_day_exits: Symbols opened today. Closing one of these is a day
            trade; closing a position opened last week is not, and treating
            every exit as a day trade would refuse trades that are free.

    Returns:
        A plan naming what fits and, for anything that does not, why.
    """
    crypto = crypto or set()
    same_day_exits = same_day_exits or set()
    reasons: list[str] = []

    increases: list[tuple[str, float]] = []
    day_trades = 0
    for symbol, target in targets.items():
        held = current.get(symbol, 0.0)
        delta = target - held
        if abs(delta) < 1e-9:
            continue
        if delta > 0:
            increases.append((symbol, delta))
        elif symbol in same_day_exits and symbol not in crypto:
            # Bought and sold on the same session: that is a day trade.
            day_trades += 1

    # Buys that close a same-day short are day trades too.
    for symbol, _ in increases:
        if symbol in same_day_exits and symbol not in crypto and current.get(symbol, 0.0) < 0:
            day_trades += 1

    reachable = dict(targets)
    deferred: dict[str, float] = {}

    # --- the day-trade budget, checked against the whole plan --------------
    if account.pdt_constrained and day_trades > account.day_trades_left:
        # Defer the equity legs, cheapest first, until the plan fits. Crypto
        # legs are never deferred for this reason -- they do not consume the
        # budget at all.
        equity_legs = sorted(
            (s for s in targets if s not in crypto and s in same_day_exits),
            key=lambda s: abs(targets[s] - current.get(s, 0.0)),
        )
        while day_trades > account.day_trades_left and equity_legs:
            symbol = equity_legs.pop(0)
            deferred[symbol] = reachable.pop(symbol, targets[symbol])
            day_trades -= 1
        reasons.append(
            f"this rebalance needs {len(deferred)} more day trade(s) than the "
            f"{account.day_trades_left} left this week; those legs are held "
            f"over. Holding overnight is not a day trade, and crypto is exempt "
            f"from the rule entirely."
        )

    # --- funding -----------------------------------------------------------
    needed = sum(
        max(0.0, reachable.get(s, 0.0) - current.get(s, 0.0)) for s in reachable
    ) * account.equity
    available = account.buying_power
    if needed > available and needed > 0:
        if not account.is_margin and account.unsettled > 0:
            reasons.append(
                f"${needed:,.0f} of buying is planned against ${available:,.0f} "
                f"settled — ${account.unsettled:,.0f} of the cash is from "
                f"recent sales and is not spendable until it settles. The "
                f"increases are scaled to what has settled."
            )
        else:
            reasons.append(
                f"${needed:,.0f} of buying is planned against ${available:,.0f} "
                f"of buying power; the increases are scaled to fit."
            )
        scale = available / needed if needed > 0 else 0.0
        for symbol in list(reachable):
            held = current.get(symbol, 0.0)
            delta = reachable[symbol] - held
            if delta > 0:
                reachable[symbol] = held + delta * scale

    return SettlementPlan(
        reachable=reachable, deferred=deferred,
        day_trades_needed=day_trades,
        day_trades_available=account.day_trades_left,
        cash_needed=needed, cash_available=available,
        reasons=tuple(reasons),
    )
