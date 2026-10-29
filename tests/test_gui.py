# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Offscreen GUI smoke tests: every theme renders every page without network access."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

import pandas as pd  # noqa: E402
from PyQt6.QtCore import QThreadPool  # noqa: E402
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


def test_backtest_view_shows_ga_pso_comparison(app, tmp_path):
    from tradebot.optimize import run_comparison

    themes = ThemeManager(user_dir=tmp_path)
    themes.apply("dark")
    view = BacktestView(None, themes, ["TEST"])
    df = make_ohlc(500, seed=4)
    cmp = run_comparison(build_algorithm("sma_cross"), df, population=16, generations=3, seed=2, backend="numpy",
                         min_trades=0)
    view._show((df, {"GA in-sample": cmp.ga.train, "GA out-of-sample": cmp.ga.test,
                     "PSO in-sample": cmp.pso.train, "PSO out-of-sample": cmp.pso.test,
                     "_params": cmp.better.best_params, "_comparison": cmp}))
    assert not view.verdict_card.isHidden()
    assert cmp.verdict.upper() in view.verdict.text() and "agreement" in view.verdict.text()
    assert list(view.metrics.df.columns)[1:] == ["GA in-sample", "GA out-of-sample", "PSO in-sample", "PSO out-of-sample"]
    view._show((df, {"backtest": cmp.ga.train}))  # a plain backtest hides the verdict again
    assert view.verdict_card.isHidden()


def test_chart_draws_elliott_wave_labels(app, tmp_path):
    import pyqtgraph as pg

    from tradebot.indicators.elliott import ElliottWaves

    from .test_elliott import path

    themes = ThemeManager(user_dir=tmp_path)
    themes.apply("dark")
    chart = ChartView(None, themes, ["TEST"])
    chart.df = path()
    chart.indicators = [ElliottWaves()]
    chart.redraw(autorange=True)
    texts = sorted(i.toPlainText() for i in chart.price.items if isinstance(i, pg.TextItem))
    assert texts == sorted("012345ABC")
    assert not chart.panes  # the hidden causal state is not drawn as a pane or line


@pytest.mark.parametrize("key", list(BUILTIN))
def test_param_dialog_uses_theme_background(app, key, tmp_path):
    from tradebot.gui.params import ParamDialog

    themes = ThemeManager(user_dir=tmp_path)
    t = themes.apply(key)
    algo = build_algorithm("rsi_threshold")
    dlg = ParamDialog("rsi_threshold parameters", algo.param_space(), algo.values)
    dlg.show()
    app.processEvents()
    img = dlg.grab().toImage()
    # a dialog is its own window: without a background rule it was default light grey, which made
    # the glass theme's near-white text invisible
    assert img.pixelColor(1, 1).name() == qcolor(t.bg).name()
    dlg.close()


class RecordingEngine:
    def __init__(self, df):
        self.df, self.calls = df, []

    def get_ohlc(self, symbol, timeframe="1d", start=None, end=None, asset_class=None, bars=None):
        self.calls.append({"start": start, "end": end, "bars": bars})
        return self.df

    def peek(self, *args, **kwargs):  # nothing on disk
        return None


def test_date_range_widget(app):
    from PyQt6.QtCore import QDate

    from tradebot.gui.common import DateRange

    dr = DateRange()
    assert dr.range() == (None, None)  # unset: candle count, up to now
    assert dr.start.text() == "from: auto"
    dr.start.setDate(QDate(2024, 3, 1))
    dr.end.setDate(QDate(2024, 6, 30))
    start, end = dr.range()
    assert start == pd.Timestamp("2024-03-01", tz="UTC")
    assert end == pd.Timestamp("2024-07-01", tz="UTC")  # exclusive end: June 30 is included
    dr.clear()
    assert dr.range() == (None, None)


def test_chart_and_lab_request_a_candle_count_or_the_date_range(app, tmp_path):
    from PyQt6.QtCore import QDate

    themes = ThemeManager(user_dir=tmp_path)
    themes.apply("dark")
    engine = RecordingEngine(make_ohlc(300))
    chart = ChartView(engine, themes, ["TEST"])
    lab = BacktestView(engine, themes, ["TEST"])
    messages = []
    chart.status.connect(messages.append)

    chart.bars.setCurrentText("1000")  # triggers a load
    QThreadPool.globalInstance().waitForDone()
    app.processEvents()
    assert engine.calls[-1] == {"start": None, "end": None, "bars": 1000}  # a count, not a calendar span
    assert "only 300 of 1000 candles available" in messages[-1]

    lab._data()
    assert engine.calls[-1] == {"start": None, "end": None, "bars": 1500}
    lab.dates.start.setDate(QDate(2020, 1, 1))
    lab._data()
    assert engine.calls[-1]["start"] == pd.Timestamp("2020-01-01", tz="UTC")  # the dates win over the count


def _contrast(a, b) -> float:
    """WCAG contrast ratio of two QColors."""
    def lum(c):
        ch = [v / 255 for v in (c.red(), c.green(), c.blue())]
        ch = [v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4 for v in ch]
        return 0.2126 * ch[0] + 0.7152 * ch[1] + 0.0722 * ch[2]

    hi, lo = sorted((lum(a), lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


@pytest.mark.parametrize("key", list(BUILTIN))
def test_date_fields_and_calendar_are_readable(app, key, tmp_path):
    from PyQt6.QtCore import QDate

    from tradebot.gui.common import DateRange
    from tradebot.gui.theme import GlassBackdrop

    themes = ThemeManager(user_dir=tmp_path)
    t = themes.apply(key)
    host = GlassBackdrop(themes)  # the glass inputs are translucent: render them over the real backdrop
    dates = DateRange(host)
    dates.start.setDate(QDate(2024, 3, 1))
    host.resize(420, 60)
    host.show()
    app.processEvents()
    from PyQt6.QtCore import QPoint, QRect

    # the empty "to: now" field, grabbed from the window so translucent glass shows what's really behind it
    field = host.grab(QRect(dates.end.mapTo(host, QPoint(0, 0)), dates.end.size())).toImage()
    background = field.pixelColor(field.width() - 30, field.height() // 2)
    # the text uses the theme's text colour; unstyled date edits were white, i.e. unreadable in dark themes
    assert _contrast(background, qcolor(t.text)) > 4.5
    calendar = dates.start.calendarWidget()
    calendar.resize(300, 220)
    calendar.show()
    app.processEvents()
    grid = calendar.grab().toImage()
    assert _contrast(grid.pixelColor(grid.width() // 2, grid.height() - 3), qcolor(t.text)) > 4.5
    calendar.hide()
