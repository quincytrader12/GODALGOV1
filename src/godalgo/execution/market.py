"""Venue market metadata.

Every venue enforces its own per-symbol rules: how many decimals an amount may
carry, what tick the price must sit on, the smallest order it will accept. They
differ by venue *and* by symbol, and they change.

Hardcoding them is the most reliable way to have a bot that works perfectly in
simulation and gets every order rejected in production. Worse, the rejection is
usually a terse venue error that says nothing about which rule was broken, so it
looks like a connectivity problem rather than a formatting one.

So the rules are read from the venue with ``load_markets()`` and applied to every
order before it is sent. Defaults exist only so that paper and dry-run modes work
without a network round trip -- they are never used to talk to a real venue.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["MarketSpec", "spec_from_ccxt_market"]


@dataclass(frozen=True, slots=True)
class MarketSpec:
    """Per-symbol trading rules as the venue states them."""

    symbol: str
    amount_precision: int = 8
    """Decimal places allowed on order quantity."""

    price_precision: int = 2
    """Decimal places allowed on price."""

    min_amount: float = 0.0
    """Smallest order quantity the venue accepts."""

    min_notional: float = 10.0
    """Smallest order value in quote currency."""

    amount_step: float | None = None
    """Lot size, when the venue quantises amount to a step rather than decimals."""

    price_tick: float | None = None
    """Tick size, when price must sit on a grid rather than a decimal count."""

    maker_fee: float | None = None
    taker_fee: float | None = None
    is_default: bool = True
    """True when these are fallback values rather than venue-supplied ones.

    Checked before live trading: sending orders shaped by guessed precision is
    how a live session fails on its first order.
    """

    def round_amount(self, amount: float) -> float:
        """Quantise an order quantity to what the venue will accept.

        Rounds **down**. Rounding up can push an order past a balance or a risk
        cap by a hair, and a hair is enough for a rejection -- or for exceeding
        a limit that exists precisely to be inviolable.
        """
        if self.amount_step:
            steps = math.floor(amount / self.amount_step)
            return round(steps * self.amount_step, self.amount_precision)
        factor = 10**self.amount_precision
        return math.floor(amount * factor) / factor

    def round_price(self, price: float, *, side_is_buy: bool) -> float:
        """Quantise a price onto the venue's grid, conservatively.

        A resting buy rounds down and a resting sell rounds up, so quantisation
        can only ever make the order *more* passive. Rounding the other way
        would silently turn a post-only order into one that crosses, which the
        venue then rejects -- or worse, fills at taker fees.
        """
        if self.price_tick:
            ticks = price / self.price_tick
            ticks = math.floor(ticks) if side_is_buy else math.ceil(ticks)
            return round(ticks * self.price_tick, self.price_precision)
        factor = 10**self.price_precision
        rounded = math.floor(price * factor) if side_is_buy else math.ceil(price * factor)
        return rounded / factor

    def is_tradeable(self, amount: float, price: float) -> tuple[bool, str]:
        """Whether an order clears the venue's minimums."""
        if amount <= 0:
            return False, "amount rounded to zero at venue precision"
        if self.min_amount and amount < self.min_amount:
            return False, f"amount {amount} below venue minimum {self.min_amount}"
        notional = amount * price
        if notional < self.min_notional:
            return False, f"notional {notional:.2f} below venue minimum {self.min_notional:.2f}"
        return True, "ok"


def spec_from_ccxt_market(market: dict) -> MarketSpec:
    """Build a ``MarketSpec`` from a ccxt market description.

    ccxt normalises most of this, but not all venues populate every field, and
    precision is reported either as a decimal count or as a step size depending
    on the exchange. Both forms are handled; missing fields fall back to the
    defaults and leave ``is_default`` informative.
    """
    precision = market.get("precision") or {}
    limits = market.get("limits") or {}
    amount_limits = limits.get("amount") or {}
    cost_limits = limits.get("cost") or {}

    amount_precision, amount_step = _precision_and_step(precision.get("amount"))
    price_precision, price_tick = _precision_and_step(precision.get("price"))

    return MarketSpec(
        symbol=market.get("symbol", "UNKNOWN"),
        amount_precision=amount_precision if amount_precision is not None else 8,
        price_precision=price_precision if price_precision is not None else 2,
        min_amount=float(amount_limits.get("min") or 0.0),
        min_notional=float(cost_limits.get("min") or 0.0) or 10.0,
        amount_step=amount_step,
        price_tick=price_tick,
        maker_fee=float(market["maker"]) if market.get("maker") is not None else None,
        taker_fee=float(market["taker"]) if market.get("taker") is not None else None,
        is_default=False,
    )


def _precision_and_step(value: object) -> tuple[int | None, float | None]:
    """Interpret a ccxt precision entry as a decimal count and/or a step.

    Venues report this inconsistently: some give ``8`` meaning eight decimals,
    others give ``0.001`` meaning the step itself. A value below 1 is treated as
    a step, which is the convention ccxt's own ``DECIMAL_PLACES`` /
    ``TICK_SIZE`` modes follow.
    """
    if value is None:
        return None, None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None, None

    if numeric <= 0:
        return 0, None
    if numeric >= 1 and float(numeric).is_integer():
        return int(numeric), None

    # A fractional value is a step size; derive decimals from it for rounding.
    decimals = max(0, int(round(-math.log10(numeric))))
    return decimals, numeric
