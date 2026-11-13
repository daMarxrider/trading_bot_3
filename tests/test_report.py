# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Backtest report (#152): extra statistics, monthly returns, underwater curve, CSV and HTML export."""

import os

import numpy as np
import pandas as pd
import pytest

from tradebot.algorithms import build_algorithm
from tradebot.backtest import backtest
from tradebot.backtest import report as rep

from .conftest import make_ohlc

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_report_stats_on_known_trades():
    idx = pd.date_range("2024-01-01", periods=6, freq="D")
    equity = pd.Series([1.0, 1.1, 0.99, 0.99, 1.2, 1.0], index=idx)
    trades = pd.DataFrame({"entry": idx[[0, 2, 4]], "exit": idx[[1, 3, 5]], "side": "long",
                           "return": [0.1, -0.1, -0.05]})
    s = rep.report_stats(equity, trades)
    assert s["expectancy"] == pytest.approx(-0.05 / 3)
    assert s["avg_win"] == pytest.approx(0.1) and s["avg_loss"] == pytest.approx(-0.075)
    assert s["longest_losing_streak"] == 2
    assert s["time_under_water"] == pytest.approx(3 / 6)
    assert s["avg_holding_days"] == pytest.approx(1.0)
    assert s["longest_under_water_days"] == pytest.approx(3.0)  # peak on day 2, recovered on day 5


def test_empty_trades_do_not_crash():
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    s = rep.report_stats(pd.Series([1.0, 1.0, 1.0], index=idx), pd.DataFrame(columns=["entry", "exit", "side", "return"]))
    assert s["expectancy"] == 0 and s["longest_losing_streak"] == 0 and s["time_under_water"] == 0


def test_monthly_returns_compound_and_year_column():
    idx = pd.date_range("2023-12-30", periods=5, freq="D")
    r = pd.Series([0.1, 0.1, 0.0, -0.5, 0.0], index=idx)
    t = rep.monthly_returns(r)
    assert t.loc[2023, "Dec"] == pytest.approx(1.1 * 1.1 - 1)
    assert t.loc[2024, "Jan"] == pytest.approx(-0.5)
    assert np.isnan(t.loc[2024, "Feb"])
    assert t.loc[2024, "Year"] == pytest.approx(-0.5)


def test_drawdown_series():
    dd = rep.drawdown_series(pd.Series([1.0, 2.0, 1.0, 3.0]))
    assert list(dd) == [0.0, 0.0, -0.5, 0.0]


def test_exports(tmp_path):
    res = backtest(make_ohlc(400), build_algorithm("sma_cross"))
    csv = rep.export_trades_csv(res, tmp_path / "t.csv")
    back = pd.read_csv(csv)
    assert len(back) == len(res.trades) and "holding_days" in back
    page = rep.export_html(res, tmp_path / "r.html", "x <b>").read_text()
    assert "<svg" in page and "expectancy" in page and "x &lt;b&gt;" in page
    assert "http://" not in page.replace("http://www.w3.org/2000/svg", "")  # self-contained


def test_cli_backtest_writes_files(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from tradebot.cli import main

    monkeypatch.setattr(main, "load_data", lambda *a, **k: make_ohlc(400))
    out = CliRunner().invoke(main.app, ["backtest", "TEST", "sma_cross", "--no-plot", "--trades-csv", str(tmp_path / "t.csv"),
                                        "--report", str(tmp_path / "r.html")])
    assert out.exit_code == 0, out.output
    assert (tmp_path / "t.csv").exists() and (tmp_path / "r.html").exists()
