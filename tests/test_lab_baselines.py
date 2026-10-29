# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""The backtest lab shows buy & hold and monthly investing next to every result (#19)."""

import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from tradebot.backtest import backtest  # noqa: E402
from tradebot.gui.theme import ThemeManager  # noqa: E402
from tradebot.gui.views import BacktestView  # noqa: E402
from tradebot.optimize import GeneticOptimizer  # noqa: E402

from .conftest import make_ohlc  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def lab(app, tmp_path):
    themes = ThemeManager(user_dir=tmp_path)
    themes.apply("dark")
    view = BacktestView(None, themes, ["TEST"])
    view.show()
    view.algo.setCurrentText("sma_cross")
    QApplication.processEvents()
    return view


def run(lab, df, kind="backtest"):
    """What the lab's worker returns, computed here without a thread."""
    lab._data = lambda: df
    captured = {}
    import tradebot.gui.views as views

    original = views.run_async
    views.run_async = lambda work, done, fail: captured.setdefault("payload", work())
    try:
        (lab.run_backtest if kind == "backtest" else lab.run_optimizer)()
    finally:
        views.run_async = original
    lab._show(captured["payload"])
    return list(lab.metrics.df.columns), [name for name, *_ in lab._curves]


def test_backtest_shows_both_baselines_with_xirr(lab):
    columns, curves = run(lab, make_ohlc(500, seed=5))
    assert columns == ["metric", "backtest", "buy & hold", "monthly investing"]
    assert curves == ["buy & hold", "monthly investing", "backtest"]
    metrics = lab.metrics.df.set_index("metric")
    assert {"xirr", "vs buy & hold", "vs monthly investing"} <= set(metrics.index)
    assert metrics.loc["vs buy & hold", "buy & hold"] == "–"  # a baseline isn't compared with itself


def test_optimizer_shows_baselines_per_period(lab):
    lab.population.setValue(16)
    lab.generations.setValue(2)
    columns, _ = run(lab, make_ohlc(600, seed=6), kind="optimize")
    assert columns == ["metric", "in-sample", "out-of-sample", "buy & hold in-sample", "monthly investing in-sample",
                       "buy & hold out-of-sample", "monthly investing out-of-sample"]


def test_losing_to_a_baseline_is_said_plainly(lab):
    rising = make_ohlc(300, seed=1)
    rising["close"] = np.linspace(100, 300, 300)
    run(lab, rising)
    assert lab.baseline_note.isVisibleTo(lab) and "does not beat buy & hold" in lab.baseline_note.text()


def test_monthly_amount_reaches_the_baseline(lab):
    lab.dca_amount.setValue(123)
    run(lab, make_ohlc(400, seed=2))
    assert lab.dca_amount.text() == "123.00" and lab._amount == 123
    lab.dca_amount.setValue(0)
    assert lab.dca_amount.text() == "auto" and lab._amount is None
