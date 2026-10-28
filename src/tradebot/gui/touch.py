# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Touchscreen gestures.

Gesture set, following common platform guidelines (Material, Fiori, Windows):

==============================  ====================================================
gesture                         action on charts
==============================  ====================================================
one-finger drag                 pan
flick (fast drag + release)     pan with momentum
pinch / spread (two fingers)    zoom around the fingers, per axis: a horizontal
                                pinch zooms time, a vertical pinch zooms price
two-finger drag                 pan while zooming
tap                             crosshair / OHLC readout, places drawing points
double-tap                      reset the view (auto range)
long press (500 ms)             context menu (secondary actions only)
==============================  ====================================================

Scrollable lists and tables get kinetic flick scrolling via :func:`enable_kinetic_scroll`.
Two-finger tap is deliberately not used (poor discoverability).

:class:`TouchTracker` holds the platform-independent gesture logic and is
unit-testable without a touchscreen; :class:`TouchController` feeds it real
Qt touch events and runs the long-press timer and flick momentum.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from PyQt6.QtCore import QEvent, QObject, QPoint, QPointF, Qt, QTimer
from PyQt6.QtGui import QEventPoint, QInputDevice
from PyQt6.QtWidgets import QAbstractItemView, QScroller, QWidget

TAP_SLOP_PX = 12.0  #: movement below this still counts as a tap / long press
LONG_PRESS_S = 0.5
DOUBLE_TAP_S = 0.3
DOUBLE_TAP_SLOP_PX = 40.0
MIN_PINCH_SPAN_PX = 40.0  #: finger distance on an axis below this does not zoom that axis
FLICK_MIN_SPEED = 400.0  #: px/s at release to start momentum
FLICK_FRICTION = 4.0  #: exponential decay per second


class GestureHandler:
    """Receives recognized gestures; positions are widget-local pixels. Override what you need."""

    def on_pan(self, dx: float, dy: float, x: float, y: float) -> None: ...
    def on_pinch(self, cx: float, cy: float, sx: float, sy: float) -> None: ...
    def on_tap(self, x: float, y: float) -> None: ...
    def on_double_tap(self, x: float, y: float) -> None: ...
    def on_long_press(self, x: float, y: float) -> None: ...
    def on_gesture_start(self, x: float, y: float) -> None: ...


@dataclass
class TouchTracker:
    """Turns raw touch point updates into gestures.

    Feed :meth:`update` with the full set of current touch points
    (``{id: (x, y)}``) on every touch event, :meth:`release` when the last finger
    lifts, and call :meth:`poll` periodically for long-press detection.
    """

    handler: GestureHandler
    points: dict[int, tuple[float, float]] = field(default_factory=dict)
    _start: tuple[float, float] | None = None
    _start_t: float = 0.0
    _moved: bool = False
    _max_fingers: int = 0
    _long_pressed: bool = False
    _last_tap: tuple[float, float, float] | None = None  # x, y, t
    _samples: list[tuple[float, float, float]] = field(default_factory=list)  # x, y, t of the single finger

    def update(self, points: dict[int, tuple[float, float]], t: float) -> None:
        if not self.points:  # first finger down: new gesture
            x, y = next(iter(points.values()))
            self._start, self._start_t = (x, y), t
            self._moved, self._max_fingers, self._long_pressed = False, 0, False
            self._samples = [(x, y, t)]
            self.handler.on_gesture_start(x, y)
            self.points = dict(points)
            self._max_fingers = len(points)
            return
        self._max_fingers = max(self._max_fingers, len(points))
        if set(points) != set(self.points):
            # finger added or lifted: rebase so the remaining fingers don't cause a jump
            self.points = dict(points)
            self._samples = []
            return
        if len(points) == 1:
            (pid, (x, y)), = points.items()
            px, py = self.points[pid]
            if not self._moved and math.dist((x, y), self._start) > TAP_SLOP_PX:
                self._moved = True
            if self._moved and not self._long_pressed:
                self.handler.on_pan(x - px, y - py, x, y)
                self._samples = [s for s in self._samples if t - s[2] <= 0.1] + [(x, y, t)]
        elif len(points) >= 2:
            self._moved = True
            ids = sorted(points)[:2]
            (ax, ay), (bx, by) = (points[i] for i in ids)
            (pax, pay), (pbx, pby) = (self.points[i] for i in ids)
            cx, cy = (ax + bx) / 2, (ay + by) / 2
            pcx, pcy = (pax + pbx) / 2, (pay + pby) / 2
            if cx != pcx or cy != pcy:
                self.handler.on_pan(cx - pcx, cy - pcy, cx, cy)
            sx = _ratio(abs(ax - bx), abs(pax - pbx))
            sy = _ratio(abs(ay - by), abs(pay - pby))
            if sx != 1.0 or sy != 1.0:
                self.handler.on_pinch(cx, cy, sx, sy)
        self.points = dict(points)

    def poll(self, t: float) -> None:
        """Fire a long press once a single finger rests long enough."""
        if (len(self.points) == 1 and self._max_fingers == 1 and not self._moved and not self._long_pressed
                and t - self._start_t >= LONG_PRESS_S):
            self._long_pressed = True
            self.handler.on_long_press(*self._start)

    def release(self, t: float) -> tuple[float, float] | None:
        """Last finger lifted. Returns a flick velocity (px/s) if the gesture ended in a flick."""
        velocity = None
        if self._max_fingers == 1 and self._start is not None:
            if not self._moved and not self._long_pressed and t - self._start_t < LONG_PRESS_S:
                x, y = self._start
                last = self._last_tap
                if last and t - last[2] <= DOUBLE_TAP_S and math.dist((x, y), last[:2]) <= DOUBLE_TAP_SLOP_PX:
                    self._last_tap = None
                    self.handler.on_double_tap(x, y)
                else:
                    self._last_tap = (x, y, t)
                    self.handler.on_tap(x, y)
            elif self._moved and len(self._samples) >= 2:
                (x0, y0, t0), (x1, y1, t1) = self._samples[0], self._samples[-1]
                if t1 > t0 and t - t1 < 0.05:
                    vx, vy = (x1 - x0) / (t1 - t0), (y1 - y0) / (t1 - t0)
                    if math.hypot(vx, vy) >= FLICK_MIN_SPEED:
                        velocity = (vx, vy)
        self.points = {}
        self._start = None
        return velocity


def _ratio(new: float, old: float) -> float:
    if new < MIN_PINCH_SPAN_PX or old < MIN_PINCH_SPAN_PX:
        return 1.0
    return new / old


class TouchController(QObject):
    """Installs on a widget (e.g. a graphics view's viewport) and drives a :class:`TouchTracker`.

    Touch events are consumed, so Qt does not additionally synthesize mouse
    events for them. Mouse and wheel input keep working unchanged.
    """

    def __init__(self, widget: QWidget, handler: GestureHandler):
        super().__init__(widget)
        self.widget, self.handler = widget, handler
        self.tracker = TouchTracker(handler)
        widget.setAttribute(Qt.WidgetAttribute.WA_AcceptTouchEvents, True)
        widget.installEventFilter(self)
        self._long_press_timer = QTimer(self, interval=50, timeout=self._poll)
        self._flick_timer = QTimer(self, interval=16, timeout=self._flick_step)
        self._flick_v = (0.0, 0.0)
        self._flick_pos = (0.0, 0.0)
        self._flick_t = 0.0

    def eventFilter(self, obj, event):
        et = event.type()
        if et not in (QEvent.Type.TouchBegin, QEvent.Type.TouchUpdate, QEvent.Type.TouchEnd, QEvent.Type.TouchCancel):
            return False
        now = time.monotonic()
        if et == QEvent.Type.TouchBegin:
            self._flick_timer.stop()  # touching the screen stops momentum
        if et == QEvent.Type.TouchCancel:
            self.tracker.points = {}
            self._long_press_timer.stop()
            return True
        active = {}
        for p in event.points():
            if p.state() != QEventPoint.State.Released:
                local = self.widget.mapFromGlobal(p.globalPosition())
                active[p.id()] = (local.x(), local.y())
        if active:
            self.tracker.update(active, now)
            self._long_press_timer.start()
        if et == QEvent.Type.TouchEnd or not active:
            self._long_press_timer.stop()
            velocity = self.tracker.release(now)
            if velocity:
                self._start_flick(velocity, event.points()[0])
        event.accept()
        return True

    def _poll(self):
        self.tracker.poll(time.monotonic())
        if not self.tracker.points or self.tracker._long_pressed or self.tracker._moved:
            self._long_press_timer.stop()

    def _start_flick(self, velocity, point):
        local = self.widget.mapFromGlobal(point.globalPosition())
        self._flick_v, self._flick_pos, self._flick_t = velocity, (local.x(), local.y()), time.monotonic()
        self._flick_timer.start()

    def _flick_step(self):
        now = time.monotonic()
        dt, self._flick_t = now - self._flick_t, now
        decay = math.exp(-FLICK_FRICTION * dt)
        vx, vy = self._flick_v[0] * decay, self._flick_v[1] * decay
        self._flick_v = (vx, vy)
        if math.hypot(vx, vy) < 30:
            self._flick_timer.stop()
            return
        x, y = self._flick_pos
        self.handler.on_pan(vx * dt, vy * dt, x, y)


class PlotTouchHandler(GestureHandler):
    """Chart gestures for pyqtgraph views (``GraphicsLayoutWidget`` / ``PlotWidget``).

    ``plot_at(scene_pos)`` returns the PlotItem under a point; ``on_tap`` /
    ``on_long_press`` receive the scene position (and the global position for menus).
    """

    def __init__(self, view, plot_at, on_tap=None, on_long_press=None):
        self.view, self.plot_at = view, plot_at
        self.tap_cb, self.long_press_cb = on_tap, on_long_press
        self._plot = None

    def _scene(self, x, y) -> QPointF:
        return self.view.mapToScene(QPoint(round(x), round(y)))

    def on_gesture_start(self, x, y):
        self._plot = self.plot_at(self._scene(x, y))

    def on_pan(self, dx, dy, x, y):
        if self._plot is None:
            return
        vb = self._plot.vb
        a = vb.mapSceneToView(self._scene(x - dx, y - dy))
        b = vb.mapSceneToView(self._scene(x, y))
        vb.translateBy(x=a.x() - b.x(), y=a.y() - b.y())  # content follows the finger

    def on_pinch(self, cx, cy, sx, sy):
        if self._plot is None:
            return
        vb = self._plot.vb
        vb.scaleBy(s=(1 / sx, 1 / sy), center=vb.mapSceneToView(self._scene(cx, cy)))

    def on_double_tap(self, x, y):
        plot = self.plot_at(self._scene(x, y)) or self._plot
        if plot is not None:
            plot.autoRange()  # fit now (this also disables auto range in pyqtgraph) ...
            plot.enableAutoRange()  # ... then keep following new data

    def on_tap(self, x, y):
        if self.tap_cb:
            self.tap_cb(self._scene(x, y))

    def on_long_press(self, x, y):
        if self.long_press_cb:
            self.long_press_cb(self._scene(x, y), self.view.mapToGlobal(QPoint(round(x), round(y))))


def enable_kinetic_scroll(view: QWidget) -> None:
    """Flick scrolling with momentum for lists, tables and scroll areas."""
    if isinstance(view, QAbstractItemView):
        view.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        view.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
    target = view.viewport() if hasattr(view, "viewport") else view
    QScroller.grabGesture(target, QScroller.ScrollerGestureType.TouchGesture)


def has_touchscreen() -> bool:
    return any(d.type() == QInputDevice.DeviceType.TouchScreen for d in QInputDevice.devices())
