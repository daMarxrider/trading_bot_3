# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Chart y-axis auto-fit: the visible-range calculation and the toggle's behaviour in the chart."""

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

from PyQt6.QtCore import QEvent, QPoint  # noqa: E402
from PyQt6.QtGui import QEventPoint  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from tradebot.gui.chart import FIT_PADDING, ChartView, visible_y_range  # noqa: E402
from tradebot.gui.theme import ThemeManager  # noqa: E402
from tradebot.indicators.builtin import RSI, Bollinger  # noqa: E402

from .conftest import make_ohlc  # noqa: E402
from .test_touch import touch  # noqa: E402

S = QEventPoint.State


# -- visible_y_range --------------------------------------------------------------------------------
def test_range_covers_only_visible_points():
    x = np.arange(10.0)
    low, high = np.arange(10.0), np.arange(10.0) + 5
    lo, hi = visible_y_range(x, [low, high], 2, 4, padding=0)
    assert (lo, hi) == (2, 9)  # lows/highs of x = 2, 3, 4 only (both ends inclusive)


def test_range_padding_and_nans():
    x = np.arange(5.0)
    lo, hi = visible_y_range(x, [np.array([np.nan, 10, 20, np.nan, 99])], 0, 3)
    assert (lo, hi) == pytest.approx((10 - FIT_PADDING * 10, 20 + FIT_PADDING * 10))


def test_range_flat_data_and_nothing_visible():
    x = np.arange(5.0)
    lo, hi = visible_y_range(x, [np.full(5, 50.0)], 0, 4)
    assert lo < 50 < hi
    assert visible_y_range(x, [np.full(5, np.nan)], 0, 4) is None  # e.g. an indicator's warm-up period
    assert visible_y_range(x, [np.arange(5.0)], 10, 20) is None  # scrolled past the data
    assert visible_y_range(x, [], 0, 4) is None


# -- in the chart -----------------------------------------------------------------------------------
@pytest.fixture
def chart(tmp_path):
    app = QApplication.instance() or QApplication([])
    themes = ThemeManager(user_dir=tmp_path)
    themes.apply("dark")
    c = ChartView(None, themes, ["TEST"])
    c.resize(1000, 700)
    c.df = make_ohlc(300)
    c.show()
    c.redraw(autorange=True)
    app.processEvents()
    return c


def visible(chart, i0, i1):
    return chart.df.iloc[i0: i1 + 1]


def assert_fits(plot, lo, hi):
    y0, y1 = plot.vb.viewRange()[1]
    pad = FIT_PADDING * (hi - lo)
    assert (y0, y1) == pytest.approx((lo - pad, hi + pad), rel=1e-6)


def test_on_by_default_in_bottom_left(chart):
    btn = chart.fit_btn
    assert btn.isCheckable() and btn.isChecked() and not btn.isHidden()
    assert btn.parent() is chart.canvas and btn.x() < 20
    assert btn.geometry().bottom() > chart.canvas.height() - 20


def test_zooming_time_fits_candles(chart):
    x = chart._x
    chart.price.setXRange(x[40], x[90], padding=0)
    v = visible(chart, 40, 90)
    assert_fits(chart.price, v.low.min(), v.high.max())
    chart.price.setXRange(x[200], x[220], padding=0)  # pan elsewhere: refits
    v = visible(chart, 200, 220)
    assert_fits(chart.price, v.low.min(), v.high.max())


def test_line_mode_fits_close(chart):
    chart.style_btn.setChecked(True)
    x = chart._x
    chart.price.setXRange(x[10], x[60], padding=0)
    v = visible(chart, 10, 60)
    assert_fits(chart.price, v.close.min(), v.close.max())


def test_overlays_and_panes_fit_too(chart):
    chart.indicators = [Bollinger(), RSI()]
    chart.redraw()
    x = chart._x
    chart.price.setXRange(x[100], x[150], padding=0)
    v = visible(chart, 100, 150)
    bands = Bollinger()(chart.df).iloc[100:151]
    assert_fits(chart.price, min(v.low.min(), bands.min().min()), max(v.high.max(), bands.max().max()))
    rsi = RSI()(chart.df).iloc[100:151].to_numpy()
    (pane,) = chart.panes
    assert_fits(pane, rsi.min(), rsi.max())


def test_drawings_do_not_skew_the_fit(chart):
    chart._add_level(10_000)
    x = chart._x
    chart.price.setXRange(x[40], x[90], padding=0)
    v = visible(chart, 40, 90)
    assert_fits(chart.price, v.low.min(), v.high.max())


class FakeWheel:
    def __init__(self, pos):
        self._pos = pos

    def delta(self):
        return 120

    def pos(self):
        return self._pos

    def accept(self):
        pass


def test_wheel_on_chart_keeps_fit_but_wheel_on_price_axis_turns_it_off(chart):
    vb = chart.price.vb
    center = vb.boundingRect().center()
    vb.wheelEvent(FakeWheel(center))  # wheel over the chart zooms time (and price, which gets refit)
    (x0, x1), _ = vb.viewRange()
    i0, i1 = np.searchsorted(chart._x, x0), np.searchsorted(chart._x, x1, "right") - 1
    v = visible(chart, i0, i1)
    assert chart.fit_btn.isChecked()
    assert_fits(chart.price, v.low.min(), v.high.max())

    before = vb.viewRange()[1]
    vb.wheelEvent(FakeWheel(center), axis=1)  # what scrolling over the price axis does
    assert not chart.fit_btn.isChecked()  # the user's manual y zoom wins
    assert vb.viewRange()[1] != pytest.approx(before)
    manual = vb.viewRange()[1]
    chart.price.setXRange(chart._x[0], chart._x[30], padding=0)
    assert vb.viewRange()[1] == pytest.approx(manual)  # stays manual while auto-fit is off

    chart.fit_btn.setChecked(True)  # turning it back on refits right away
    v = visible(chart, 0, 30)
    assert_fits(chart.price, v.low.min(), v.high.max())


def test_touch_pinch_and_reset_keep_the_fit(chart):
    vp = chart.canvas.viewport()
    c = vp.rect().center()
    touch(vp, QEvent.Type.TouchBegin, [(0, QPoint(c.x() - 50, c.y()), S.Pressed), (1, QPoint(c.x() + 50, c.y()), S.Pressed)])
    touch(vp, QEvent.Type.TouchUpdate, [(0, QPoint(c.x() - 200, c.y()), S.Updated),
                                        (1, QPoint(c.x() + 200, c.y()), S.Updated)])
    touch(vp, QEvent.Type.TouchEnd, [(0, QPoint(c.x() - 200, c.y()), S.Released),
                                     (1, QPoint(c.x() + 200, c.y()), S.Released)])
    (x0, x1), _ = chart.price.vb.viewRange()
    i0, i1 = np.searchsorted(chart._x, x0), np.searchsorted(chart._x, x1, "right") - 1
    v = visible(chart, i0, i1)
    assert chart.fit_btn.isChecked()
    assert_fits(chart.price, v.low.min(), v.high.max())

    chart.reset_view()  # what a double tap does: everything visible, still fitted
    assert chart.fit_btn.isChecked()
    assert_fits(chart.price, chart.df.low.min(), chart.df.high.max())
