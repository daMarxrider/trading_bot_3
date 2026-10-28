# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Touch gestures: pure gesture recognition plus real QTouchEvents sent to the chart."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

from PyQt6.QtCore import QEvent, QPoint, QPointF, Qt  # noqa: E402
from PyQt6.QtGui import QEventPoint, QInputDevice, QPointingDevice, QTouchEvent  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from tradebot.gui.chart import ChartView  # noqa: E402
from tradebot.gui.theme import BUILTIN, TOUCH_QSS, ThemeManager, stylesheet  # noqa: E402
from tradebot.gui.touch import GestureHandler, TouchTracker  # noqa: E402

from .conftest import make_ohlc  # noqa: E402


class Recorder(GestureHandler):
    def __init__(self):
        self.events = []

    def on_pan(self, dx, dy, x, y):
        self.events.append(("pan", round(dx, 3), round(dy, 3)))

    def on_pinch(self, cx, cy, sx, sy):
        self.events.append(("pinch", round(sx, 3), round(sy, 3)))

    def on_tap(self, x, y):
        self.events.append(("tap", x, y))

    def on_double_tap(self, x, y):
        self.events.append(("double_tap", x, y))

    def on_long_press(self, x, y):
        self.events.append(("long_press", x, y))

    def kinds(self):
        return [e[0] for e in self.events]


@pytest.fixture
def rec():
    return Recorder()


def test_tap_and_double_tap(rec):
    tr = TouchTracker(rec)
    tr.update({0: (100, 100)}, 0.0)
    tr.release(0.1)
    tr.update({0: (105, 102)}, 0.25)
    tr.release(0.3)
    tr.update({0: (100, 100)}, 2.0)  # much later: a fresh single tap again
    tr.release(2.1)
    assert rec.kinds() == ["tap", "double_tap", "tap"]


def test_small_jitter_is_still_a_tap_but_drag_is_pan(rec):
    tr = TouchTracker(rec)
    tr.update({0: (100, 100)}, 0.0)
    tr.update({0: (104, 103)}, 0.05)  # within slop
    tr.release(0.1)
    assert rec.kinds() == ["tap"]
    rec.events.clear()
    tr.update({0: (100, 100)}, 1.0)
    tr.update({0: (130, 100)}, 1.05)
    tr.update({0: (150, 90)}, 1.1)
    tr.release(1.5)  # slow release: no flick, no tap
    assert rec.kinds() == ["pan", "pan"] and rec.events[-1] == ("pan", 20, -10)


def test_long_press_fires_once_and_suppresses_tap(rec):
    tr = TouchTracker(rec)
    tr.update({0: (50, 60)}, 0.0)
    tr.poll(0.3)
    assert rec.kinds() == []
    tr.poll(0.55)
    tr.poll(0.8)
    tr.release(0.9)
    assert rec.events == [("long_press", 50, 60)]


def test_pinch_is_per_axis(rec):
    tr = TouchTracker(rec)
    tr.update({0: (100, 100), 1: (200, 105)}, 0.0)  # fingers side by side
    tr.update({0: (50, 100), 1: (250, 105)}, 0.05)  # spread horizontally x2
    assert ("pinch", 2.0, 1.0) in rec.events  # vertical span < threshold: price axis untouched
    tr.release(0.1)
    assert "tap" not in rec.kinds()


def test_two_finger_drag_pans_without_zoom(rec):
    tr = TouchTracker(rec)
    tr.update({0: (100, 100), 1: (200, 200)}, 0.0)
    tr.update({0: (110, 130), 1: (210, 230)}, 0.05)
    assert rec.events == [("pan", 10, 30)]


def test_adding_a_finger_does_not_jump(rec):
    tr = TouchTracker(rec)
    tr.update({0: (100, 100)}, 0.0)
    tr.update({0: (140, 100)}, 0.05)
    tr.update({0: (140, 100), 1: (300, 100)}, 0.1)  # second finger lands: rebase only
    tr.update({0: (140, 100), 1: (300, 100)}, 0.15)
    assert rec.kinds() == ["pan"]


def test_flick_velocity(rec):
    tr = TouchTracker(rec)
    tr.update({0: (0, 0)}, 0.0)
    for i in range(1, 6):
        tr.update({0: (i * 40, 0)}, i * 0.02)  # 2000 px/s
    v = tr.release(0.105)
    assert v is not None and v[0] == pytest.approx(2000, rel=0.05)


# -- touch events against the chart -----------------------------------------------------------------
DEV = None


class FakePoint:
    """Stand-in for QEventPoint: PyQt6's QEventPoint constructor drops the point id (always 0),
    so multi-finger sequences can't be built from real points in a test."""

    def __init__(self, pid, state, glob):
        self._id, self._state, self._g = pid, state, glob

    def id(self):
        return self._id

    def state(self):
        return self._state

    def globalPosition(self):
        return self._g


class FakeTouchEvent:
    def __init__(self, et, points):
        self._et, self._points = et, points

    def type(self):
        return self._et

    def points(self):
        return self._points

    def accept(self):
        pass


def touch(widget, et, pts):
    """Deliver a touch event; ``pts`` = [(id, local QPoint, state)].

    Single-finger events go through Qt as genuine QTouchEvents; multi-finger events are fed
    straight into the widget's TouchController.
    """
    global DEV
    if DEV is None:
        DEV = QPointingDevice("test-touch", 4242, QInputDevice.DeviceType.TouchScreen,
                              QPointingDevice.PointerType.Finger, QInputDevice.Capability.Position, 10, 0)
    if len(pts) == 1:
        pid, local, state = pts[0]
        g = QPointF(widget.mapToGlobal(local))
        QApplication.sendEvent(widget, QTouchEvent(et, DEV, Qt.KeyboardModifier.NoModifier, [QEventPoint(pid, state, g, g)]))
        return
    owner = widget
    while not hasattr(owner, "touch"):  # the view that installed the TouchController
        owner = owner.parent()
    owner.touch.eventFilter(widget, FakeTouchEvent(et, [FakePoint(pid, st, QPointF(widget.mapToGlobal(p))) for pid, p, st in pts]))


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


S = QEventPoint.State


def test_pinch_zooms_time_axis(chart):
    chart.fit_btn.setChecked(False)  # test the gesture alone; auto-fit would refit the price axis
    vp = chart.canvas.viewport()
    c = vp.rect().center()
    (x0, x1), (y0, y1) = chart.price.vb.viewRange()
    a, b = QPoint(c.x() - 50, c.y()), QPoint(c.x() + 50, c.y())
    touch(vp, QEvent.Type.TouchBegin, [(0, a, S.Pressed), (1, b, S.Pressed)])
    touch(vp, QEvent.Type.TouchUpdate, [(0, QPoint(c.x() - 150, c.y()), S.Updated),
                                        (1, QPoint(c.x() + 150, c.y()), S.Updated)])
    touch(vp, QEvent.Type.TouchEnd, [(0, QPoint(c.x() - 150, c.y()), S.Released),
                                     (1, QPoint(c.x() + 150, c.y()), S.Released)])
    (nx0, nx1), (ny0, ny1) = chart.price.vb.viewRange()
    assert (nx1 - nx0) == pytest.approx((x1 - x0) / 3, rel=0.02)  # spread x3 -> 3x zoom on time
    assert (ny1 - ny0) == pytest.approx(y1 - y0, rel=0.02)  # price axis unchanged


def test_drag_pans_and_double_tap_resets(chart):
    vp = chart.canvas.viewport()
    c = vp.rect().center()
    chart.reset_view()
    QApplication.processEvents()
    before = chart.price.vb.viewRange()[0]
    touch(vp, QEvent.Type.TouchBegin, [(0, c, S.Pressed)])
    touch(vp, QEvent.Type.TouchUpdate, [(0, QPoint(c.x() + 200, c.y()), S.Updated)])
    touch(vp, QEvent.Type.TouchEnd, [(0, QPoint(c.x() + 200, c.y()), S.Released)])
    after = chart.price.vb.viewRange()[0]
    assert after[0] < before[0] and after[1] - after[0] == pytest.approx(before[1] - before[0], rel=1e-6)
    assert not chart.price.vb.state["autoRange"][0]  # panning leaves auto range
    for _ in range(2):
        touch(vp, QEvent.Type.TouchBegin, [(0, c, S.Pressed)])
        touch(vp, QEvent.Type.TouchEnd, [(0, c, S.Released)])
    QApplication.processEvents()
    assert chart.price.vb.state["autoRange"][0]  # double tap restored auto range
    assert chart.price.vb.viewRange()[0] == pytest.approx(before, rel=1e-3)


def test_tap_places_level_with_drawing_tool(chart):
    vp = chart.canvas.viewport()
    chart.tools["hline"].setChecked(True)
    p = vp.rect().center()
    touch(vp, QEvent.Type.TouchBegin, [(0, p, S.Pressed)])
    touch(vp, QEvent.Type.TouchEnd, [(0, p, S.Released)])
    assert len(chart.drawings) == 1 and not chart.tools["hline"].isChecked()


def test_context_menu_actions(chart):
    vp = chart.canvas.viewport()
    p = vp.rect().center()
    chart._context_menu(chart.canvas.mapToScene(p), vp.mapToGlobal(p))
    labels = [a.text() for a in chart._menu.actions() if a.text()]
    assert labels[0] == "Reset zoom" and any(t.startswith("Add level at") for t in labels)
    next(a for a in chart._menu.actions() if a.text().startswith("Add level")).trigger()
    assert len(chart.drawings) == 1
    chart._menu.close()


def test_touch_mode_stylesheet():
    t = BUILTIN["dark"]
    assert "min-height: 30px" not in stylesheet(t) and "min-height: 30px" in stylesheet(t, touch=True)
    assert TOUCH_QSS
