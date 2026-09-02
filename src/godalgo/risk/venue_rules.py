"""Rules the venue imposes, enforced before an order is worth computing.

``RiskLimits`` bounds what the *portfolio* may do. Nothing there knows that a
market can be shut, or that a broker will refuse a fourth day trade on a small
account. Both of those turn a perfectly sized order into one that cannot
happen, and a bot that discovers this from a rejection has already wasted a
signal -- and, on the day-trade rule, has spent a scarce resource it did not
know it was spending.

Like every other limit in this package, these are **deterministic and not
tunable**. They are not in any strategy's ``SPACE``, the optimiser never sees
them, and they are constructed from configuration. A search that could relax
the day-trade counter would learn to trade its way into a ninety-day
restriction.

Two rules, and they apply to different things:

* **Market hours.** US equities trade in a session. Crypto does not stop. The
  clock comes from the venue rather than from a timezone calculation here,
  because holidays and early closes are the venue's business and a bot that
  decides for itself will be wrong on exactly the days it is most costly.

* **The pattern-day-trader rule.** In a margin account under $25,000, a fourth
  day trade inside five business days gets the account restricted, typically
  for ninety days. That is not a fee, it is the bot being switched off, and it
  is the single most likely way an intraday equities strategy destroys its own
  ability to trade in week one. Crypto is exempt, which is a large part of why
  crypto is the sensible thing to run first on a small account.

The day-trade budget is treated as a resource to be spent deliberately: the
guard stops *opening* new equity positions once the remaining budget is down to
its reserve, while always allowing a position to be closed. Refusing to close
would trap a position with no stop management, which is worse than any rule
this module enforces.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

__all__ = ["PDT_EQUITY_FLOOR", "TradeGate", "VenueState", "check_tradeable"]

logger = logging.getLogger(__name__)

PDT_EQUITY_FLOOR = 25_000.0
"""Account equity below which the day-trade rule applies, in USD.

A regulatory threshold, not a preference, which is why it is a constant here
rather than a tunable.
"""

_PDT_WINDOW = 4
"""Day trades within five business days that trigger the restriction."""


@dataclass(frozen=True, slots=True)
class VenueState:
    """What the venue currently says about itself and the account.

    Every field is read from the venue, never inferred. The whole value of this
    check is that it reflects the broker's own view -- our own count of day
    trades would drift from theirs, and theirs is the one that restricts the
    account.
    """

    market_open: bool = True
    equity: float = 0.0
    day_trades_used: int = 0
    """Day trades in the last five business days, as the broker counts them."""

    flagged_pattern_day_trader: bool = False
    trading_blocked: bool = False
    shorting_enabled: bool = True

    @property
    def day_trades_left(self) -> int:
        """How many remain before the restriction bites.

        Unlimited above the equity floor, which is the whole point of the
        threshold.
        """
        if self.equity >= PDT_EQUITY_FLOOR:
            return 99
        return max(0, _PDT_WINDOW - 1 - self.day_trades_used)

    @property
    def pdt_constrained(self) -> bool:
        return self.equity < PDT_EQUITY_FLOOR

    def to_dict(self) -> dict[str, Any]:
        return {
            "market_open": self.market_open,
            "equity": self.equity,
            "day_trades_used": self.day_trades_used,
            "day_trades_left": self.day_trades_left,
            "pdt_constrained": self.pdt_constrained,
            "trading_blocked": self.trading_blocked,
            "shorting_enabled": self.shorting_enabled,
        }

    @classmethod
    def from_account(cls, account: dict[str, Any], *, market_open: bool) -> VenueState:
        """Build from Alpaca's account object."""
        def _f(key: str) -> float:
            try:
                return float(account.get(key) or 0.0)
            except (TypeError, ValueError):
                return 0.0

        return cls(
            market_open=market_open,
            equity=_f("equity") or _f("portfolio_value"),
            day_trades_used=int(_f("daytrade_count")),
            flagged_pattern_day_trader=bool(account.get("pattern_day_trader")),
            trading_blocked=bool(account.get("trading_blocked"))
            or bool(account.get("account_blocked")),
            shorting_enabled=bool(account.get("shorting_enabled", True)),
        )


@dataclass(frozen=True, slots=True)
class TradeGate:
    """Whether a trade may proceed, and why not when it may not."""

    allowed: bool
    reason: str = ""
    kind: str = ""
    """Machine-readable class, for lamps and filtering. Empty when allowed."""

    def __bool__(self) -> bool:
        return self.allowed


def check_tradeable(
    symbol: str,
    state: VenueState,
    *,
    is_crypto: bool,
    opening: bool = True,
    reserve: int = 1,
) -> TradeGate:
    """Whether this instrument can be traded right now.

    Args:
        symbol: What is being traded, for the message.
        state: The venue's own view of itself and the account.
        is_crypto: Crypto is exempt from both rules here.
        opening: Whether this would open or increase a position. Closing is
            held to a laxer standard on purpose -- see below.
        reserve: Day trades to hold back rather than spend. One by default, so
            a position opened today can still be closed today without the
            closing trade being the one that trips the rule.
    """
    if state.trading_blocked:
        return TradeGate(
            False,
            "The broker has blocked trading on this account. Nothing here can "
            "clear that — check the account status with the broker.",
            "account_blocked",
        )

    if is_crypto:
        # Neither rule applies. Trades continuously and is not a security for
        # the purposes of the day-trade rule.
        return TradeGate(True)

    if not state.market_open:
        return TradeGate(
            False,
            f"The market is closed, so {symbol} cannot trade. Equities trade "
            f"in a session; crypto does not stop.",
            "market_closed",
        )

    if not opening:
        # Always allow a close. Refusing would strand a position with nothing
        # managing its stop, which is a worse outcome than any rule here.
        return TradeGate(True)

    if state.pdt_constrained and state.day_trades_left <= reserve:
        return TradeGate(
            False,
            f"Only {state.day_trades_left} day trade(s) remain before the "
            f"pattern-day-trader rule restricts this account, and {reserve} "
            f"is held back so open positions can still be closed. Account "
            f"equity is under ${PDT_EQUITY_FLOOR:,.0f}. Crypto is exempt from "
            f"this rule.",
            "pdt_budget",
        )

    return TradeGate(True)
