# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Performance metrics on hand-checkable series, including the degenerate ones (no trades, flat, wiped out)."""

import math

import numpy as np
import pandas as pd
import pytest

from tradebot.algorithms import BaseAlgorithm
from tradebot.backtest import backtest, batch_fitness
from tradebot.backtest.engine import bps
from tradebot.backtest.metrics import (
    compute_metrics,
    deflated_sharpe,
    extract_trades,
    fmt,
    infer_periods_per_year,
    max_drawdown,
)


def series(values, start="2024-01-01"):
    return pd.Series(np.asarray(values, float), index=pd.date_range(start, periods=len(values), freq="D", tz="UTC"))


def no_trades():
    return pd.DataFrame({"return": pd.Series([], dtype=float)})


def trades(*returns):
    return pd.DataFrame({"return": list(returns)})


# -- max_drawdown ------------------------------------------------------------------------------------------
def test_drawdown_is_measured_from_the_running_peak():
    # peak 120, trough 90 afterwards: 1 - 90/120 = 0.25 (the start at 100 is not the peak)
    assert max_drawdown(np.array([100.0, 120.0, 90.0, 130.0])) == pytest.approx(0.25)


def test_a_rising_curve_has_no_drawdown():
    assert max_drawdown(np.array([1.0, 2.0, 3.0])) == 0.0


def test_an_empty_curve_has_no_drawdown():
    assert max_drawdown(np.array([])) == 0.0


def test_a_wiped_out_account_has_a_full_drawdown():
    assert max_drawdown(np.array([1.0, 0.0])) == 1.0


# -- infer_periods_per_year ------------------------------------------------------------------------------
def test_daily_bars_without_gaps_are_365_and_a_quarter_a_year():
    assert infer_periods_per_year(series([1, 2]).index) == pytest.approx(365.25)


def test_hourly_bars_are_counted_per_hour():
    idx = pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC")
    assert infer_periods_per_year(idx) == pytest.approx(365.25 * 24)


@pytest.mark.parametrize("index", [
    pd.DatetimeIndex([]),
    pd.DatetimeIndex(["2024-01-01"]),
    pd.DatetimeIndex(["2024-01-01"] * 3),  # no time span: can't measure, so the stock-market default
])
def test_unmeasurable_frequency_falls_back_to_252(index):
    assert infer_periods_per_year(index) == 252.0


# -- extract_trades ----------------------------------------------------------------------------------------
def test_a_closed_trade_compounds_its_bars():
    # long on bars 1-2, flat on bar 3: earns bars 1, 2 and 3 (the exit bar's move): 1.0 * 1.1 * 1.1 - 1 = 0.21
    out = extract_trades(series([0, 1, 1, 0, 0]), series([0, 0, 0.1, 0.1, 0.5]))
    assert out["return"].tolist() == pytest.approx([0.21])
    assert out["side"].tolist() == ["long"]


def test_a_trade_open_at_the_end_is_closed_on_the_last_bar():
    out = extract_trades(series([0, 1, 1]), series([0, 0, 0.1]))
    assert out["exit"].tolist() == [series([0, 1, 1]).index[-1]]
    assert out["return"].tolist() == pytest.approx([0.1])


def test_a_direct_flip_from_long_to_short_is_two_trades():
    out = extract_trades(series([1, -1]), series([0.0, 0.0]))
    assert out["side"].tolist() == ["long", "short"]


def test_a_change_of_size_in_the_same_direction_stays_one_trade():
    out = extract_trades(series([0, 0.5, 1, 0]), series([0, 0, 0, 0]))
    assert len(out) == 1


def test_never_in_the_market_means_no_trades_but_the_columns_exist():
    out = extract_trades(series([0, 0, 0]), series([0.1, -0.1, 0.1]))
    assert out.empty and list(out.columns) == ["entry", "exit", "side", "return"]


def test_an_empty_backtest_has_no_trades():
    out = extract_trades(series([]), series([]))
    assert out.empty


# -- compute_metrics ---------------------------------------------------------------------------------------
def test_sharpe_is_annualised_mean_over_population_std():
    # returns 0.02, 0.0: mean 0.01, population std 0.01 -> 1 per bar, * sqrt(4 bars a year) = 2
    out = compute_metrics(series([0.02, 0.0]), series([1, 1]), no_trades(), ppy=4)
    assert out["sharpe"] == pytest.approx(2.0)


def test_flat_returns_have_a_zero_sharpe_not_a_division_error():
    out = compute_metrics(series([0.0, 0.0, 0.0]), series([0, 0, 0]), no_trades(), ppy=252)
    assert out["sharpe"] == 0.0 and out["sortino"] == 0.0


def test_no_trades_have_zero_win_rate_and_profit_factor():
    out = compute_metrics(series([0.0, 0.0]), series([0, 0]), no_trades(), ppy=252)
    assert (out["trades"], out["win_rate"], out["profit_factor"]) == (0, 0.0, 0.0)


def test_only_winning_trades_have_an_infinite_profit_factor():
    out = compute_metrics(series([0.0, 0.1]), series([1, 1]), trades(0.05, 0.1), ppy=252)
    assert out["profit_factor"] == math.inf and out["win_rate"] == 1.0


def test_profit_factor_is_gross_win_over_gross_loss():
    # wins 0.3 + 0.1 = 0.4, losses 0.2 -> 2.0; a 0.0 trade counts as a loss (not a win)
    out = compute_metrics(series([0.0, 0.0]), series([1, 1]), trades(0.3, -0.2, 0.1, 0.0), ppy=252)
    assert out["profit_factor"] == pytest.approx(2.0)
    assert out["win_rate"] == pytest.approx(0.5)


def test_a_wiped_out_account_has_no_cagr():
    out = compute_metrics(series([0.0, -1.0]), series([1, 1]), trades(-1.0), ppy=252)
    assert math.isnan(out["cagr"])
    assert out["total_return"] == -1.0 and out["max_drawdown"] == 1.0


def test_one_bar_has_no_cagr():
    out = compute_metrics(series([0.1]), series([1]), no_trades(), ppy=252)
    assert math.isnan(out["cagr"])


def test_cagr_compounds_over_the_years():
    # 3 bars at 1 bar a year = 2 years; 21% total -> sqrt(1.21) - 1 = 10% a year
    out = compute_metrics(series([0.0, 0.1, 0.1]), series([1, 1, 1]), no_trades(), ppy=1)
    assert out["cagr"] == pytest.approx(0.1)


def test_exposure_is_the_share_of_bars_in_the_market():
    out = compute_metrics(series([0, 0, 0, 0]), series([0, 1, -1, 0]), no_trades(), ppy=252)
    assert out["exposure"] == 0.5


# -- fmt -------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("key, value, text", [
    ("trades", 5, "5"),
    ("total_return", 0.1234, "12.34%"),
    ("max_drawdown", 1.0, "100.00%"),
    ("total_return", -0.000001, "0.00%"),  # never "-0.00%"
    ("vs buy & hold", 0.05, "+5.00%"),
    ("vs buy & hold", -0.05, "-5.00%"),
    ("sharpe", 1.234, "1.23"),
    ("profit_factor", math.inf, "inf"),
    ("cagr", math.nan, "nan%"),
])
def test_metric_formatting(key, value, text):
    assert fmt(key, value) == text


# -- deflated_sharpe ----------------------------------------------------------------------------------------
def test_deflated_sharpe_of_a_single_strategy():
    # r = 1, 2, 3: Sharpe 2/sqrt(2/3) = 2.449, skew 0, kurtosis 1.5 -> spread 1 + 0.5/4 * 6 = 1.75;
    # z = 2.449 * sqrt(2) / sqrt(1.75) = 2.619 -> Phi(2.619) = 0.9956; one trial: no luck to beat
    prob, luck = deflated_sharpe(np.array([1.0, 2.0, 3.0]), np.array([]))
    assert prob == pytest.approx(0.9956, abs=1e-3)
    assert luck == 0.0


def test_more_trials_raise_the_bar_luck_has_to_beat():
    returns = np.array([0.01, -0.005, 0.02, 0.0, 0.015, -0.01, 0.01])
    alone, _ = deflated_sharpe(returns, np.array([0.1]))
    many, luck = deflated_sharpe(returns, np.array([0.1, -0.2, 0.3, 0.0, 0.25]), trials=1000)
    assert luck > 0 and many < alone


@pytest.mark.parametrize("returns", [
    np.array([0.01, 0.02]),  # fewer than three
    np.array([0.01, 0.01, 0.01]),  # no spread
    np.array([np.nan, np.inf, 0.01, 0.02]),  # only two finite ones left
])
def test_deflated_sharpe_is_undefined_on_too_little_information(returns):
    prob, luck = deflated_sharpe(returns, np.array([0.1, 0.2]))
    assert math.isnan(prob) and math.isnan(luck)


# -- batch_fitness and backtest guards ------------------------------------------------------------------------
def test_return_objective_is_the_compounded_return_per_row():
    out = batch_fitness(np.array([[0.1, 0.1], [0.0, -0.5]]), periods_per_year=252, objective="return")
    assert out.tolist() == pytest.approx([0.21, -0.5])


def test_an_unknown_objective_is_rejected():
    with pytest.raises(ValueError, match="unknown objective 'profit'"):
        batch_fitness(np.array([[0.1, 0.1]]), periods_per_year=252, objective="profit")


class AlwaysLong(BaseAlgorithm):
    name = ""  # stays out of the registry

    def signals(self, o, p):
        return o.close * 0 + 1


@pytest.mark.parametrize("candles", [0, 1])
def test_backtest_needs_two_candles(candles):
    c = np.ones(candles)
    df = pd.DataFrame({"open": c, "high": c, "low": c, "close": c},
                      index=pd.date_range("2024-01-01", periods=candles, freq="D", tz="UTC"))
    with pytest.raises(ValueError, match="at least two candles"):
        backtest(df, AlwaysLong())


def test_basis_points_are_ten_thousandths():
    assert bps(5) == pytest.approx(0.0005)
