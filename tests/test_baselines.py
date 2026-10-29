# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Buy & hold and monthly investing as baselines for backtests and optimizations (#19)."""

import numpy as np
import pandas as pd
import pytest

from tradebot.algorithms import build_algorithm
from tradebot.backtest.baselines import (BUY_HOLD, MONTHLY, baselines, buy_and_hold, month_starts,
                                         monthly_investing, xirr)
from tradebot.backtest.engine import from_positions
from tradebot.backtest.metrics import SECONDS_PER_YEAR, fmt
from tradebot.optimize import GeneticOptimizer

from .conftest import make_ohlc

YEAR = pd.Timedelta(seconds=SECONDS_PER_YEAR)


def frame(dates, closes):
    idx = pd.to_datetime(dates, utc=True)
    return pd.DataFrame({c: np.asarray(closes, float) for c in ("open", "high", "low", "close")}, index=idx)


# Three months; purchases on Jan 1, Feb 1 and Mar 1 at 100, 50 and 100.
DCA = frame(["2024-01-01", "2024-01-15", "2024-02-01", "2024-02-15", "2024-03-01"], [100, 110, 50, 60, 100])


def test_xirr_known_rates():
    t0 = pd.Timestamp("2023-01-01", tz="UTC")
    assert xirr([t0, t0 + YEAR], [-100, 110]) == pytest.approx(0.10)
    # 100 grows for two years, 100 more for one: 100 * 1.1**2 + 100 * 1.1 = 231
    assert xirr([t0, t0 + YEAR, t0 + 2 * YEAR], [-100, -100, 231]) == pytest.approx(0.10)
    assert xirr([t0, t0 + YEAR], [-100, 50]) == pytest.approx(-0.50)
    assert np.isnan(xirr([t0, t0], [-100, 110]))  # no time passed
    assert np.isnan(xirr([t0, t0 + YEAR], [-100, -1]))  # nothing paid out


def test_monthly_investing_by_hand():
    fee, capital = 0.01, 3000
    res = monthly_investing(DCA, fee=fee, initial_capital=capital)
    assert res.params["amount"] == 1000  # capital / 3 months
    units = 1000 * (1 - fee) * np.array([1 / 100, 1 / 100, 1 / 100 + 1 / 50, 1 / 100 + 1 / 50, 2 / 100 + 1 / 50])
    cash = np.array([2000, 2000, 1000, 1000, 0])
    np.testing.assert_allclose(res.equity, cash + units * DCA["close"].to_numpy())
    assert res.equity.iloc[-1] == pytest.approx(3960)  # 39.6 units at 100; 1% of each 1,000 went to fees
    # time-weighted: price moves on the units held, times the fee factor on each purchase bar
    twr = ((1 - fee) * 1.10 * (50 / 110) * (495 + 990) / (495 + 1000) * (60 / 50)
           * (100 / 60) * (2970 + 990) / (2970 + 1000))
    assert res.metrics["total_return"] == pytest.approx(twr - 1)
    # money-weighted: the three payments grow into 3,960 at the XIRR
    years = np.array([(pd.Timestamp("2024-03-01") - pd.Timestamp(d)).days for d in
                      ("2024-01-01", "2024-02-01", "2024-03-01")]) * 86400 / SECONDS_PER_YEAR
    assert np.sum(1000 * (1 + res.metrics["xirr"]) ** years) == pytest.approx(3960)


def test_monthly_investing_custom_amount_and_intraday_months():
    df = frame(pd.date_range("2024-01-31 20:00", periods=8, freq="h"), [100] * 8)
    assert month_starts(df.index).tolist() == [True, False, False, False, True, False, False, False]
    res = monthly_investing(df, fee=0.0, initial_capital=1000, amount=100)
    assert res.params["amount"] == 100
    assert res.equity.iloc[-1] == pytest.approx(1000)  # 800 cash + 200 invested at an unchanged price


def test_buy_and_hold_pays_one_fee():
    df = make_ohlc(400, seed=4)
    res = buy_and_hold(df, fee=0.002, initial_capital=1000)
    growth = df["close"].iloc[-1] / df["close"].iloc[0]
    assert res.metrics["total_return"] == pytest.approx((1 - 0.002) * growth - 1)
    years = (df.index[-1] - df.index[0]).total_seconds() / SECONDS_PER_YEAR
    assert res.metrics["xirr"] == pytest.approx(((1 - 0.002) * growth) ** (1 / years) - 1)
    assert res.metrics["trades"] == 1 and res.metrics["exposure"] == 1.0


def test_a_strategy_that_stays_out_of_a_rising_market_is_flagged():
    df = frame(pd.date_range("2024-01-01", periods=120, freq="D"), np.linspace(100, 200, 120))
    flat = from_positions(df, pd.Series(0.0, index=df.index), "flat")
    bl = baselines(df, flat)
    assert bl.strategy_xirr == pytest.approx(0)
    assert bl.excess[BUY_HOLD] < 0 and bl.excess[MONTHLY] < 0
    assert bl.not_beaten == [BUY_HOLD, MONTHLY]
    cols = bl.columns("strategy", flat)
    assert list(cols) == ["strategy", BUY_HOLD, MONTHLY]
    assert cols["strategy"]["vs buy & hold"] == bl.excess[BUY_HOLD]
    assert fmt("vs buy & hold", 0.1234) == "+12.34%" and fmt("xirr", -1e-17) == "0.00%"


def test_optimizer_reports_baselines_on_the_same_split():
    df = make_ohlc(600, seed=7)
    res = GeneticOptimizer(build_algorithm("sma_cross"), df, backend="numpy", population=16, generations=2,
                           seed=0, dca_amount=500).run()
    for part, bl in ((res.train, res.train_baselines), (res.test, res.test_baselines)):
        for base in bl.results.values():
            assert base.equity.index.equals(part.equity.index)
        assert bl.results[MONTHLY].params["amount"] == 500


def test_cli_backtest_and_optimize_show_the_baselines(monkeypatch):
    from typer.testing import CliRunner

    from tradebot.cli import main as cli

    rising = frame(pd.date_range("2022-01-01", periods=400, freq="D"), np.linspace(100, 300, 400))
    monkeypatch.setattr(cli, "load_data", lambda *a, **k: rising)
    res = CliRunner().invoke(cli.app, ["backtest", "X", "sma_cross", "--no-plot", "--dca-amount", "250"],
                             env={"COLUMNS": "140"})
    assert res.exit_code == 0, res.output
    assert all(s in res.output for s in ("buy & hold", "monthly investing", "xirr", "vs buy & hold"))
    assert "does not beat buy & hold" in res.output  # sma_cross misses the start of the rise
    res = CliRunner().invoke(cli.app, ["optimize", "X", "sma_cross", "--population", "16", "--generations", "2",
                                       "--backend", "numpy", "--min-trades", "0"], env={"COLUMNS": "140"})
    assert res.exit_code == 0, res.output
    assert "in-sample" in res.output and "out-of-sample" in res.output and "monthly investing" in res.output
