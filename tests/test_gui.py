# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Offscreen GUI smoke tests: every theme renders every page without network access."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from tradebot.algorithms import build_algorithm  # noqa: E402
from tradebot.backtest import backtest  # noqa: E402
from tradebot.gui.chart import ChartView  # noqa: E402
from tradebot.gui.theme import BUILTIN, ThemeManager, qcolor, stylesheet  # noqa: E402
from tradebot.gui.views import BacktestView, DonutChart  # noqa: E402
from tradebot.indicators.builtin import MACD, RSI, Bollinger  # noqa: E402

from .conftest import make_ohlc  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.mark.parametrize("key", list(BUILTIN))
def test_theme_renders_chart(app, key, tmp_path):
    themes = ThemeManager(user_dir=tmp_path)
    themes.apply(key)
    assert app.styleSheet() == stylesheet(BUILTIN[key])
    chart = ChartView(None, themes, ["TEST"])
    chart.resize(900, 600)
    chart.df = make_ohlc(300)
    chart.indicators = [Bollinger(), RSI(), MACD()]
    chart.algorithm = build_algorithm("rsi_threshold")
    chart.redraw(autorange=True)
    chart.show()
    app.processEvents()
    assert len(chart.panes) == 2  # RSI and MACD get their own panes
    img = chart.grab().toImage()
    assert not img.isNull() and img.width() > 0
    # the chart is actually painted, not a blank widget
    colors = {img.pixel(x, y) for x in range(0, img.width(), 5) for y in range(0, img.height(), 5)}
    assert len(colors) > 5


def test_user_theme_inherits_base(tmp_path, app):
    (tmp_path / "midnight.json").write_text('{"name": "Midnight", "base": "dark", "accent": "#ff6ad5"}')
    themes = ThemeManager(user_dir=tmp_path)
    t = themes.themes["midnight"]
    assert t.accent == "#ff6ad5" and t.bg == BUILTIN["dark"].bg
    themes.apply("midnight")


def test_rgba_parsing():
    c = qcolor("rgba(255, 128, 0, 0.5)")
    assert (c.red(), c.green(), c.blue(), c.alpha()) == (255, 128, 0, 127)


def test_backtest_view_shows_results(app, tmp_path):
    themes = ThemeManager(user_dir=tmp_path)
    themes.apply("glass")
    view = BacktestView(None, themes, ["TEST"])
    df = make_ohlc(400)
    res = backtest(df, build_algorithm("sma_cross"))
    view._show((df, {"backtest": res}))
    assert view.metrics.rowCount() == len(res.metrics)


def test_donut(app, tmp_path):
    import pandas as pd

    d = DonutChart(ThemeManager(user_dir=tmp_path))
    d.set_weights(pd.Series({f"S{i}": 1 / 10 for i in range(10)}), top=5)
    assert list(d.weights.index)[-1] == "other" and d.weights.sum() == pytest.approx(1)
    d.resize(400, 300)
    assert not d.grab().isNull()
