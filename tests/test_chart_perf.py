# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Chart rendering at screen resolution (#68): merged candles, culling, downsampled curves, reused panes."""

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from tradebot.gui.chart import CandlestickItem, ChartView, Curve, HistogramItem, ohlc_buckets  # noqa: E402
from tradebot.gui.theme import ThemeManager  # noqa: E402
from tradebot.indicators.builtin import MACD, RSI, Bollinger  # noqa: E402

from .conftest import make_ohlc  # noqa: E402


def test_buckets_keep_first_open_extremes_and_last_close():
    x = np.arange(10.0)
    o, h, l, c = x + 100, x + 200, x, x + 50
    bx, bo, bh, bl, bc = ohlc_buckets(x, o, h, l, c, 3, 10, 4)
    # groups aligned to multiples of 4: [0..3], [4..7], [8..9]; the first starts before i0 so it stays put
    assert list(bo) == [100, 104, 108] and list(bh) == [203, 207, 209]
    assert list(bl) == [0, 4, 8] and list(bc) == [53, 57, 59]
    assert list(bx) == [1.5, 5.5, 8.5]  # the middle of each group


@pytest.fixture
def chart(tmp_path):
    app = QApplication.instance() or QApplication([])
    themes = ThemeManager(user_dir=tmp_path)
    themes.apply("dark")
    c = ChartView(None, themes, ["TEST"])
    c.resize(1000, 700)
    c.show()
    app.processEvents()
    return c


def item(plot, cls):
    return next(i for i in plot.items if isinstance(i, cls))


def test_many_candles_are_merged_per_pixel_and_few_are_drawn_one_by_one(chart):
    chart.df = make_ohlc(20000)
    chart.redraw(autorange=True)
    chart.canvas.grab()
    candles = item(chart.price, CandlestickItem)
    (i0, i1, k), shapes = candles._cache
    drawn = sum(len(rects) for _, _, rects in shapes)
    assert k > 1 and drawn <= 1.1 * chart.price.vb.width() + 2  # about one merged candle per pixel
    assert not candles._smooth  # no antialiasing for a dense chart

    x0, x1 = chart._x[100], chart._x[160]
    chart.price.setXRange(x0, x1, padding=0)
    chart.canvas.grab()
    (i0, i1, k), shapes = candles._cache
    assert k == 1 and i0 >= 90 and i1 <= 170  # only the visible ones (and a neighbour at each edge)
    assert sum(len(rects) for _, _, rects in shapes) == i1 - i0
    assert candles._smooth


def test_curves_and_histogram_are_downsampled_to_the_view(chart):
    chart.df = make_ohlc(20000)
    chart.indicators = [Bollinger(), MACD()]
    chart.redraw(autorange=True)
    chart.canvas.grab()
    for plot in [chart.price, *chart.panes]:
        for curve in (i for i in plot.items if isinstance(i, Curve)):
            assert len(curve.curve.xData) <= 2.2 * plot.vb.width() + 4  # "peak": a min and a max per pixel
            assert not curve.curve.opts["antialias"]
    hist = item(chart.panes[0], HistogramItem)
    assert hist._cache[0][2] > 1

    chart.price.setXRange(chart._x[100], chart._x[160], padding=0)
    chart.canvas.grab()
    curve = item(chart.price, Curve)
    assert len(curve.curve.xData) < 80 and curve.curve.opts["antialias"]  # zoomed in: smooth as before


def test_panes_are_reused_across_redraws(chart):
    chart.df = make_ohlc(300)
    chart.indicators = [RSI(), MACD()]
    chart.redraw(autorange=True)
    panes = list(chart.panes)
    chart.redraw()
    assert chart.panes == panes and len(chart._cross) == 3 and len(chart._legend) == 3
    assert sum(isinstance(i, Curve) for i in panes[0].items) == 1  # RSI's line, not the previous drawing's too
    chart.indicators = [MACD()]
    chart.redraw()
    assert chart.panes == panes[:1] and panes[1] not in chart._cross and panes[1].scene() is None
    assert item(chart.panes[0], HistogramItem)  # the first pane now shows MACD
    chart.indicators = []
    chart.redraw()
    assert chart.panes == [] and list(chart._cross) == [chart.price]
