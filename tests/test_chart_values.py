# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Values of the bar under the cursor in each pane's top-left corner (#45)."""

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

from PyQt6.QtCore import QEvent, QPointF  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from tradebot.algorithms import build_algorithm  # noqa: E402
from tradebot.gui.chart import ChartView, fmt_value  # noqa: E402
from tradebot.gui.theme import BUILTIN, ThemeManager  # noqa: E402
from tradebot.indicators.builtin import MACD, RSI, Bollinger  # noqa: E402

from .conftest import make_ohlc  # noqa: E402


def test_fmt_value():
    assert fmt_value(93564.4) == "93,564"
    assert fmt_value(184.2) == "184.20"
    assert fmt_value(-1.95349) == "-1.9535"
    assert fmt_value(0.123456) == "0.12346"
    assert fmt_value(float("nan")) == "–" and fmt_value(0) == "0" and fmt_value(66.0) == "66"


@pytest.fixture
def chart(tmp_path):
    app = QApplication.instance() or QApplication([])
    themes = ThemeManager(user_dir=tmp_path)
    themes.apply("dark")
    c = ChartView(None, themes, ["TEST"])
    c.resize(1100, 800)
    c.show()
    c._loaded("BTC/USDT", make_ohlc(300))
    c.indicators = [Bollinger(), RSI(), MACD()]
    c.algorithm = build_algorithm("rsi_threshold")
    c.redraw()
    app.processEvents()
    return c


def text(chart, plot):
    return chart._legend[plot].text


def hover(chart, plot, i, y=None):
    y = float(np.mean(plot.vb.viewRange()[1])) if y is None else y
    chart._mouse_moved(plot.vb.mapViewToScene(QPointF(chart._x[i], y)))


def test_latest_bar_shown_without_cursor(chart):
    last = chart.df.iloc[-1]
    price = text(chart, chart.price)
    assert "BTC/USDT · 1d" in price and fmt_value(last.close) in price
    rsi = RSI()(chart.df)["rsi_value"].iloc[-1]
    assert fmt_value(rsi) in text(chart, chart.panes[0])


@pytest.mark.parametrize("pane", [0, 1, 2])  # hovering the price, RSI or MACD pane
def test_every_pane_shows_the_hovered_bar(chart, pane):
    i = 150
    hover(chart, [chart.price, *chart.panes][pane], i)
    row = chart.df.iloc[i]
    price = text(chart, chart.price)
    for v in (row.open, row.high, row.low, row.close):
        assert fmt_value(v) in price
    change = row.close - chart.df.close.iloc[i - 1]
    assert f"{change / chart.df.close.iloc[i - 1] * 100:+.2f}%" in price
    bands = Bollinger()(chart.df).iloc[i]
    assert all(fmt_value(v) in price for v in bands)  # overlay values in the price pane's corner
    assert fmt_value(RSI()(chart.df)["rsi_value"].iloc[i]) in text(chart, chart.panes[0])
    macd = MACD()(chart.df).iloc[i]
    assert all(fmt_value(v) in text(chart, chart.panes[1]) for v in macd)


def test_values_use_their_line_colours(chart):
    t = BUILTIN["dark"]
    hover(chart, chart.price, 120)
    price, rsi = text(chart, chart.price), text(chart, chart.panes[0])
    up_or_down = t.up if chart.df.close.iloc[120] >= chart.df.open.iloc[120] else t.down
    assert f"color:{up_or_down}'>{fmt_value(chart.df.close.iloc[120])}" in price
    for color in t.series[:3]:  # the three Bollinger lines, drawn first
        assert f"color:{color}'" in price
    assert f"color:{t.series[3]}'>{fmt_value(RSI()(chart.df)['rsi_value'].iloc[120])}" in rsi
    assert "bbands 20 2" in price and "rsi 14" in rsi and "macd 12 26 9" in text(chart, chart.panes[1])


def test_signal_line(chart):
    pos = chart._positions
    entries = np.flatnonzero(np.diff(np.r_[0.0, pos]) > 0)
    assert len(entries)
    hover(chart, chart.price, int(entries[0]))
    assert "rsi_threshold 14 33 66" in text(chart, chart.price) and ">buy<" in text(chart, chart.price)
    holding = next(i for i in range(int(entries[0]) + 1, len(pos)) if pos[i] == pos[i - 1] == 1)
    hover(chart, chart.price, holding)
    assert ">long<" in text(chart, chart.price)


def test_warm_up_shows_a_dash(chart):
    hover(chart, chart.panes[0], 2)  # RSI(14) has no value yet
    assert "rsi 14</span> <span" in text(chart, chart.panes[0]) and ">–<" in text(chart, chart.panes[0])


def test_leaving_the_chart_shows_the_latest_bar_again(chart):
    hover(chart, chart.price, 50)
    assert fmt_value(chart.df.close.iloc[50]) in text(chart, chart.price)
    QApplication.sendEvent(chart.canvas, QEvent(QEvent.Type.Leave))
    assert chart._hover is None and fmt_value(chart.df.close.iloc[-1]) in text(chart, chart.price)


def test_values_never_change_the_view(chart):
    plots = [chart.price, *chart.panes]
    before = [p.vb.viewRange() for p in plots]
    for i in (10, 150, 290):
        hover(chart, chart.panes[0], i)
    assert [p.vb.viewRange() for p in plots] == before
    for plot in plots:
        assert chart._legend[plot] not in plot.vb.addedItems  # not a data item: no effect on auto-fit


def test_redraw_keeps_the_hovered_bar_and_panes(chart):
    hover(chart, chart.price, 77)
    chart.indicators = [RSI()]
    chart.redraw()
    assert set(chart._legend) == {chart.price, *chart.panes}
    assert fmt_value(chart.df.close.iloc[77]) in text(chart, chart.price)
    assert fmt_value(RSI()(chart.df)["rsi_value"].iloc[77]) in text(chart, chart.panes[0])


def test_title_shows_the_full_name_once_known(chart):
    assert "BTC/USDT · 1d" in text(chart, chart.price)  # the ticker until the lookup answers (offline in tests)
    chart._named("BTC/USDT", "1d", "Bitcoin / TetherUS")
    assert "Bitcoin / TetherUS · 1d" in text(chart, chart.price)
    chart._loaded("AAPL", chart.df)
    chart._named("BTC/USDT", "1d", "Bitcoin / TetherUS")  # a late answer for the previous symbol
    assert "AAPL · 1d" in text(chart, chart.price) and "Bitcoin" not in text(chart, chart.price)
    chart._named("AAPL", "1d", None)  # unknown: keep the ticker
    assert "AAPL · 1d" in text(chart, chart.price)
