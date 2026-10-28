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

    def ignore(self):  # a locked axis passes the wheel on
        pass


def test_wheel_zooms_time_only_and_the_price_axis_is_locked(chart):
    vb = chart.price.vb
    center = vb.boundingRect().center()
    vb.wheelEvent(FakeWheel(center))  # wheel over the chart zooms time; the price axis refits
    (x0, x1), _ = vb.viewRange()
    i0, i1 = np.searchsorted(chart._x, x0), np.searchsorted(chart._x, x1, "right") - 1
    v = visible(chart, i0, i1)
    assert_fits(chart.price, v.low.min(), v.high.max())

    fitted = vb.viewRange()[1]
    vb.wheelEvent(FakeWheel(center), axis=1)  # what scrolling over the price axis does
    assert vb.viewRange()[1] == pytest.approx(fitted)  # locked: nothing happens
    assert chart.fit_btn.isChecked()  # only the toggle switches auto-fit off


def test_cannot_pan_past_the_data(chart):
    x = chart._x
    span = x[50] - x[0]
    chart.price.setXRange(x[0] - 10 * span, x[0] - 9 * span, padding=0)  # far before the first candle
    x0, x1 = chart.price.vb.viewRange()[0]
    assert x0 >= x[0] - (x[1] - x[0]) * 1.01 and x1 > x[0]  # clamped: candles stay in view
    chart.price.setXRange(x[-1] + 9 * span, x[-1] + 10 * span, padding=0)  # far after the last one
    x0, x1 = chart.price.vb.viewRange()[0]
    assert x1 <= x[-1] + (x[1] - x[0]) * 1.01 and x0 < x[-1]


def test_toggle_off_unlocks_and_keeps_the_range(chart):
    vb = chart.price.vb
    chart.price.setXRange(chart._x[0], chart._x[30], padding=0)
    chart.fit_btn.setChecked(False)
    assert vb.state["mouseEnabled"] == [True, True]
    fitted = vb.viewRange()[1]
    assert vb.viewRange()[1] == pytest.approx(fitted)  # switching off doesn't jump
    vb.wheelEvent(FakeWheel(vb.boundingRect().center()), axis=1)
    assert vb.viewRange()[1] != pytest.approx(fitted)  # now the price axis is free
    chart.price.setXRange(chart._x[-1] + 1e9, chart._x[-1] + 2e9, padding=0)
    assert vb.viewRange()[0][0] > chart._x[-1]  # and time can be panned past the data

    chart.fit_btn.setChecked(True)  # back on: refits and locks again right away
    assert vb.state["mouseEnabled"] == [True, False]
    chart.price.setXRange(chart._x[0], chart._x[30], padding=0)
    v = visible(chart, 0, 30)
    assert_fits(chart.price, v.low.min(), v.high.max())


def test_touch_vertical_moves_leave_the_price_axis_alone(chart):
    vp = chart.canvas.viewport()
    c = vp.rect().center()
    fitted = chart.price.vb.viewRange()[1]
    touch(vp, QEvent.Type.TouchBegin, [(0, QPoint(c.x(), c.y() - 50), S.Pressed), (1, QPoint(c.x(), c.y() + 50), S.Pressed)])
    touch(vp, QEvent.Type.TouchUpdate, [(0, QPoint(c.x(), c.y() - 150), S.Updated),
                                        (1, QPoint(c.x(), c.y() + 150), S.Updated)])  # vertical spread
    touch(vp, QEvent.Type.TouchEnd, [(0, QPoint(c.x(), c.y() - 150), S.Released),
                                     (1, QPoint(c.x(), c.y() + 150), S.Released)])
    touch(vp, QEvent.Type.TouchBegin, [(0, c, S.Pressed)])
    touch(vp, QEvent.Type.TouchUpdate, [(0, QPoint(c.x(), c.y() + 200), S.Updated)])  # vertical drag
    touch(vp, QEvent.Type.TouchEnd, [(0, QPoint(c.x(), c.y() + 200), S.Released)])
    assert chart.price.vb.viewRange()[1] == pytest.approx(fitted)
    assert chart.fit_btn.isChecked()


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


def test_toggle_is_remembered_across_sessions(tmp_path):
    from tradebot.config import Settings

    app = QApplication.instance() or QApplication([])
    themes = ThemeManager(user_dir=tmp_path)
    themes.apply("dark")
    path = tmp_path / "settings.json"
    first = ChartView(None, themes, ["TEST"], Settings(path))
    assert first.fit_btn.isChecked()  # default on
    first.fit_btn.setChecked(False)
    second = ChartView(None, themes, ["TEST"], Settings(path))  # a new session
    assert not second.fit_btn.isChecked()
    second.fit_btn.setChecked(True)
    assert ChartView(None, themes, ["TEST"], Settings(path)).fit_btn.isChecked()
    app.processEvents()


# -- #36: adding indicators or switching the signal must not blank the chart ------------------------
def accept_next_dialog():
    from PyQt6.QtCore import QTimer

    def accept():
        dialog = QApplication.activeModalWidget()
        if dialog is None:
            QTimer.singleShot(10, accept)
        else:
            dialog.accept()

    QTimer.singleShot(0, accept)


def x_ranges(chart):
    return [tuple(p.vb.viewRange()[0]) for p in [chart.price, *chart.panes]]


@pytest.mark.parametrize("added", [["rsi", "stoch"], ["stoch", "bbands"], ["rsi", "stoch", "bbands"]])
def test_adding_indicators_through_the_menu_keeps_the_time_range(chart, added):
    from tradebot.indicators import INDICATORS

    app = QApplication.instance()
    before = chart.price.vb.viewRange()[0]
    for name in added:
        accept_next_dialog()
        chart._add_indicator(INDICATORS.get(name))  # what the Indicators menu does, dialog included
        app.processEvents()
    ranges = x_ranges(chart)
    assert all(r == pytest.approx(before, abs=1) for r in ranges)  # no collapse, no drift, all panes aligned
    overlays = [ind(chart.df) for ind in chart.indicators if ind.overlay]  # all candles are visible
    assert_fits(chart.price, min([chart.df.low.min()] + [f.min().min() for f in overlays]),
                max([chart.df.high.max()] + [f.max().max() for f in overlays]))


def test_switching_the_signal_keeps_the_time_range(chart):
    app = QApplication.instance()
    chart.indicators = [RSI()]
    chart.redraw()
    before = chart.price.vb.viewRange()[0]
    for name in ("rsi_threshold", "macd_cross"):
        accept_next_dialog()
        chart.algo_box.setCurrentText(name)  # what picking a signal does, dialog included
        app.processEvents()
    assert all(r == pytest.approx(before, abs=1) for r in x_ranges(chart))


def test_zoomed_view_survives_redraws(chart):
    x = chart._x
    chart.price.setXRange(x[100], x[160], padding=0)
    zoomed = chart.price.vb.viewRange()[0]
    for _ in range(3):
        chart.indicators = [RSI(), Bollinger()]
        chart.redraw()
    assert all(r == pytest.approx(zoomed, abs=1) for r in x_ranges(chart))
    v = visible(chart, 100, 160)
    assert chart.price.vb.viewRange()[1][0] < v.low.min() and chart.price.vb.viewRange()[1][1] > v.high.max()


def test_panning_a_pane_past_the_data_stays_clamped(chart):
    chart.indicators = [RSI()]
    chart.redraw()
    x = chart._x
    (pane,) = chart.panes
    pane.setXRange(x[0] - 1e9, x[0] - 1e9 + (x[50] - x[0]), padding=0)  # far before the first candle
    for x0, x1 in x_ranges(chart):
        assert x1 > x[0] and x0 >= x[0] - (x[1] - x[0]) * 1.01
