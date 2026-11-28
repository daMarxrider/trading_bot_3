# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Chart rendering at screen resolution (#68, #228): merged candles, culling, downsampled curves and fills, reused
panes, a cheap auto-fit per scroll step."""

import os
import sys

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from tradebot.gui.chart import (  # noqa: E402
    Band,
    CandlestickItem,
    ChartView,
    Curve,
    DateAxis,
    HistogramItem,
    RangeIndex,
    ohlc_buckets,
    peak_buckets,
)
from tradebot.gui.theme import ThemeManager  # noqa: E402
from tradebot.indicators.builtin import MACD, RSI, Bollinger, Volume  # noqa: E402

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
            assert len(curve.visible()[0]) <= 2.2 * plot.vb.width() + 4  # "peak": a min and a max per pixel
            assert not curve.opts["antialias"]
    hist = item(chart.panes[0], HistogramItem)
    assert hist._cache[0][2] > 1

    chart.price.setXRange(chart._x[100], chart._x[160], padding=0)
    chart.canvas.grab()
    curve = item(chart.price, Curve)
    assert len(curve.visible()[0]) < 80 and curve.opts["antialias"]  # zoomed in: smooth as before


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


def test_peak_buckets_keep_each_columns_lowest_and_highest_value():
    x, y = np.arange(10.0), np.array([5.0, 1, 9, 4, 4, 4, 0, 7, 3, 3])
    bx, by = peak_buckets(x, y, 3, 10, 4)
    # groups aligned to multiples of 4, like the candles: [0..3], [4..7], [8..9]
    assert list(by) == [1, 9, 0, 7, 3, 3]
    assert list(bx) == [1.5, 1.5, 5.5, 5.5, 8.5, 8.5]


def test_range_index_matches_a_scan_of_every_value():
    rng = np.random.default_rng(1)
    n = 1000
    x = np.cumsum(rng.uniform(1, 3, n))  # uneven spacing, like candles with gaps
    a, b = rng.normal(0, 1, n), rng.normal(5, 1, n)
    a[rng.random(n) < 0.1] = np.nan
    b[:300] = np.nan  # a warm-up period
    index = RangeIndex(x, [a, b])
    for _ in range(200):
        i0, i1 = sorted(rng.integers(0, n + 1, 2))
        both = np.concatenate([a[i0:i1], b[i0:i1]])
        both = both[np.isfinite(both)]
        expected = (both.min(), both.max()) if both.size else None
        assert index.bounds(i0, i1) == expected
    assert RangeIndex(x, []).bounds(0, n) is None


def test_band_spans_both_lines_in_every_pixel_column(chart):
    chart.df = make_ohlc(20000)
    chart.indicators = [Bollinger()]
    chart.redraw(autorange=True)
    chart.canvas.grab()
    band = item(chart.price, Band)
    x, y = band._outline()
    half = len(x) // 2
    assert half <= 1.1 * chart.price.vb.width() + 2  # one column per pixel, not 10,000 points
    upper, lower = np.nanmax(band.a), np.nanmin(band.b)
    assert y[:half].max() == pytest.approx(upper) and y[half:].min() == pytest.approx(lower)


def test_dense_time_ticks_are_left_out_but_the_coarsest_level_stays():
    levels = [(365 * 86400.0, [1.0]), (30 * 86400.0, [2.0]), (86400.0, [3.0])]
    span = 6 * 365 * 86400.0  # six years on 1,200 pixels: 200 px a year, 16 a month, 0.5 a day
    assert DateAxis._sparse_levels(levels, span, 1200) == levels[:1]
    assert DateAxis._sparse_levels(levels, span / 10, 1200) == levels[:2]  # 160 px a month
    assert DateAxis._sparse_levels(levels, span * 100, 1200) == levels[:1]  # all too dense: still the years


def test_one_auto_fit_per_scroll_step_with_panes(chart, monkeypatch):
    chart.df = make_ohlc(2000)
    chart.indicators = [RSI(), MACD()]
    chart.redraw(autorange=True)
    calls = []
    fit = chart.fit_y
    monkeypatch.setattr(chart, "fit_y", lambda *a, **k: (calls.append(1), fit(*a, **k)))
    x0, x1 = chart.price.vb.viewRange()[0]
    chart.price.setXRange(x0 + 86400 * 10, x1 - 86400 * 10, padding=0)
    assert len(calls) == 1  # the linked panes moved too, but don't fit everything again


@pytest.mark.parametrize("n, zoom", [(20000, None), (300, None), (300, (100, 130))])
def test_every_item_paints_without_errors(chart, monkeypatch, n, zoom):
    """Qt only prints an exception raised in paint(): collect them, so a broken item fails here."""
    errors = []
    monkeypatch.setattr(sys, "excepthook", lambda *exc: errors.append(exc))
    chart.df = make_ohlc(n)
    chart.indicators = [Volume(), Bollinger(), RSI(), MACD()]
    chart._overlay = True  # the volume overlay too (no settings in this fixture)
    chart.redraw(autorange=True)
    if zoom:
        chart.price.setXRange(chart._x[zoom[0]], chart._x[zoom[1]], padding=0)
    chart.canvas.grab()
    chart.style_btn.setChecked(True)  # the line chart
    chart.canvas.grab()
    assert errors == []


def test_indicators_are_recomputed_only_for_new_candles(chart, monkeypatch):
    chart.df = make_ohlc(300)
    chart.indicators = [RSI(), MACD()]
    chart.redraw()
    calls = []
    call = RSI.__call__
    monkeypatch.setattr(RSI, "__call__", lambda self, df: (calls.append(1), call(self, df))[1])
    chart.style_btn.setChecked(True)  # line chart: same candles
    chart.themes.apply("light")
    assert calls == []
    chart.df = chart.df.copy()  # new candles (a load, a live tick)
    chart.redraw()
    assert calls == [1]
