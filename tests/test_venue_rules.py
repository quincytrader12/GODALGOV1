"""Rules the venue imposes, which no amount of correct sizing can satisfy.

Two ways an equities bot destroys itself in week one, neither of which the
portfolio risk layer can see:

* trading into a closed market, which is not a bad trade but an impossible one;
* spending its fourth day trade on a sub-$25k margin account, which gets the
  account restricted for around ninety days. That is the bot being switched
  off, not a fee.

Crypto is exempt from both, which is most of the argument for running crypto
first on a small account.
"""

from __future__ import annotations

import pytest

from godalgo.risk.venue_rules import (
    PDT_EQUITY_FLOOR,
    TradeGate,
    VenueState,
    check_tradeable,
)


def _state(**kw) -> VenueState:
    base = {"market_open": True, "equity": 10_000.0, "day_trades_used": 0}
    return VenueState(**{**base, **kw})


# --------------------------------------------------------------------------
# market hours
# --------------------------------------------------------------------------

def test_equities_cannot_trade_into_a_closed_market():
    gate = check_tradeable("AAPL", _state(market_open=False), is_crypto=False)
    assert not gate
    assert gate.kind == "market_closed"


def test_crypto_ignores_the_clock_entirely():
    assert check_tradeable("BTC/USD", _state(market_open=False), is_crypto=True)


def test_equities_trade_when_the_market_is_open():
    assert check_tradeable("AAPL", _state(), is_crypto=False)


# --------------------------------------------------------------------------
# the pattern-day-trader rule
# --------------------------------------------------------------------------

def test_a_large_account_is_not_day_trade_constrained():
    """Above the threshold the rule does not apply, which is the point of it."""
    state = _state(equity=PDT_EQUITY_FLOOR + 1, day_trades_used=50)
    assert state.pdt_constrained is False
    assert check_tradeable("AAPL", state, is_crypto=False)


def test_opening_stops_before_the_rule_trips():
    """Not on the fourth trade -- before it. Discovering this from a broker
    restriction is discovering it far too late."""
    gate = check_tradeable("AAPL", _state(day_trades_used=2), is_crypto=False)
    assert not gate
    assert gate.kind == "pdt_budget"


def test_a_trade_is_held_back_so_open_positions_can_be_closed():
    """Spending the last day trade on an entry strands the exit."""
    state = _state(day_trades_used=2)
    assert state.day_trades_left == 1
    assert not check_tradeable("AAPL", state, is_crypto=False, opening=True)


def test_closing_is_always_allowed():
    """Refusing to close strands a position with nothing managing its stop,
    which is worse than anything this module is protecting against."""
    exhausted = _state(day_trades_used=99)
    assert check_tradeable("AAPL", exhausted, is_crypto=False, opening=False)


def test_crypto_is_exempt_from_the_day_trade_rule():
    assert check_tradeable("BTC/USD", _state(day_trades_used=99), is_crypto=True)


def test_the_budget_counts_down():
    assert _state(day_trades_used=0).day_trades_left == 3
    assert _state(day_trades_used=1).day_trades_left == 2
    assert _state(day_trades_used=3).day_trades_left == 0
    assert _state(day_trades_used=9).day_trades_left == 0   # never negative


# --------------------------------------------------------------------------
# a blocked account
# --------------------------------------------------------------------------

def test_a_blocked_account_stops_everything_including_crypto():
    """Nothing the bot does can clear this, so it must not keep trying."""
    for crypto in (True, False):
        gate = check_tradeable("X", _state(trading_blocked=True), is_crypto=crypto)
        assert not gate
        assert gate.kind == "account_blocked"


def test_a_blocked_account_also_blocks_closing():
    gate = check_tradeable(
        "AAPL", _state(trading_blocked=True), is_crypto=False, opening=False
    )
    assert not gate


# --------------------------------------------------------------------------
# reading the broker's own view
# --------------------------------------------------------------------------

def test_the_state_comes_from_the_brokers_account_object():
    """Our own day-trade count would drift from the broker's, and the broker's
    is the one that restricts the account."""
    state = VenueState.from_account(
        {
            "equity": "12345.67",
            "daytrade_count": 2,
            "pattern_day_trader": False,
            "trading_blocked": False,
            "shorting_enabled": True,
        },
        market_open=True,
    )
    assert state.equity == pytest.approx(12_345.67)
    assert state.day_trades_used == 2
    assert state.day_trades_left == 1


def test_a_missing_or_junk_field_does_not_raise():
    """A malformed account payload must not take down the trading loop."""
    state = VenueState.from_account({"equity": "not a number"}, market_open=False)
    assert state.equity == 0.0
    assert state.market_open is False


def test_portfolio_value_is_accepted_where_equity_is_absent():
    state = VenueState.from_account({"portfolio_value": "5000"}, market_open=True)
    assert state.equity == 5_000.0


def test_either_block_flag_blocks():
    for key in ("trading_blocked", "account_blocked"):
        state = VenueState.from_account({key: True}, market_open=True)
        assert state.trading_blocked is True


# --------------------------------------------------------------------------
# the invariant the whole risk package rests on
# --------------------------------------------------------------------------

def test_these_rules_are_not_in_any_search_space():
    """A search that could relax the day-trade counter would learn to trade
    its way into a ninety-day restriction."""
    import inspect

    from godalgo.strategies.mean_reversion import MeanReversionParams
    from godalgo.strategies.momentum import MomentumParams

    for params in (MomentumParams, MeanReversionParams):
        names = set(params.SPACE)
        assert names, "a strategy with an empty search space would pass vacuously"
        assert not names & {
            "day_trades_used", "day_trades_left", "equity", "market_open",
            "reserve", "PDT_EQUITY_FLOOR",
        }

    # And nothing in the module offers a way to widen the threshold.
    source = inspect.getsource(
        __import__("godalgo.risk.venue_rules", fromlist=["x"])
    )
    assert "PDT_EQUITY_FLOOR = 25_000.0" in source


def test_a_gate_is_truthy_only_when_allowed():
    assert bool(TradeGate(True)) is True
    assert bool(TradeGate(False, "no")) is False
