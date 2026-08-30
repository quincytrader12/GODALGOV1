"""Feasibility arithmetic.

The governing fact: volatility scales as sqrt(time), costs do not scale at all.
Below some bar length, no signal pays for itself.
"""

import pytest

from godalgo.execution.broker import FeeSchedule
from godalgo.feasibility import (
    assess_frequency,
    expected_edge_bps,
    minimum_viable_bar_seconds,
)


def test_edge_grows_as_sqrt_of_bar_length():
    one = expected_edge_bps(0.60, 60, 0.30, 10)
    four = expected_edge_bps(0.60, 240, 0.30, 10)
    assert four / one == pytest.approx(2.0, rel=1e-6)


def test_edge_grows_as_sqrt_of_holding_period():
    short = expected_edge_bps(0.60, 60, 0.30, 10)
    long = expected_edge_bps(0.60, 60, 0.30, 40)
    assert long / short == pytest.approx(2.0, rel=1e-6)


def test_costs_do_not_scale_with_bar_length():
    """The asymmetry that creates a frequency floor."""
    fast = assess_frequency(1, annual_vol=0.60, conviction=0.30, holding_bars=10)
    slow = assess_frequency(86_400, annual_vol=0.60, conviction=0.30, holding_bars=10)
    assert fast.maker_cost_bps == slow.maker_cost_bps
    assert slow.edge_bps > fast.edge_bps


def test_high_frequency_is_not_tradeable_at_typical_costs():
    report = assess_frequency(1, annual_vol=0.60, conviction=0.30, holding_bars=10)
    assert not report.tradeable_as_maker
    assert report.headroom < 1.0


def test_hourly_bars_clear_comfortably():
    report = assess_frequency(3600, annual_vol=0.60, conviction=0.30, holding_bars=25)
    assert report.tradeable_as_maker
    assert report.tradeable_as_taker


def test_holding_longer_lowers_the_frequency_floor():
    """One of only two levers that move the floor."""
    short = minimum_viable_bar_seconds(0.60, 0.30, 3)
    long = minimum_viable_bar_seconds(0.60, 0.30, 48)
    assert long < short
    # 16x the hold -> 1/16th the bar length, since edge goes as sqrt of both.
    assert short / long == pytest.approx(16.0, rel=1e-6)


def test_crossing_the_spread_raises_the_floor_sharply():
    """The other lever."""
    maker = minimum_viable_bar_seconds(0.60, 0.30, 25, maker=True)
    taker = minimum_viable_bar_seconds(0.60, 0.30, 25, maker=False)
    assert taker > maker * 10


def test_zero_conviction_is_never_tradeable():
    assert minimum_viable_bar_seconds(0.60, 0.0, 10) == float("inf")


def test_report_explains_what_would_help_and_what_would_not():
    report = assess_frequency(1, annual_vol=0.60, conviction=0.30, holding_bars=5)
    text = report.describe()
    assert "NOT TRADEABLE" in text
    # The gate is a measurement, not an obstacle to be lowered.
    assert "Lowering min_edge_multiple is NOT on this list" in text


def test_cheaper_fees_lower_the_floor():
    expensive = minimum_viable_bar_seconds(
        0.60, 0.30, 25, fees=FeeSchedule(maker=0.001, taker=0.001)
    )
    cheap = minimum_viable_bar_seconds(
        0.60, 0.30, 25, fees=FeeSchedule(maker=0.0001, taker=0.0006)
    )
    assert cheap < expensive
