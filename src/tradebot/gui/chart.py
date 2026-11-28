# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Interactive OHLC chart: candles/line, indicator overlays and panes,
algorithm buy/sell markers, crosshair, and drawing tools
(trendline, horizontal support/resistance line, rectangle)."""

from __future__ import annotations

from dataclasses import replace
from html import escape

import numpy as np
import pandas as pd
import pyqtgraph as pg
from pyqtgraph.Qt.internals import PrimitiveArray
from PyQt6.QtCore import QEvent, QLineF, QObject, QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen, QTransform
from PyQt6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QColorDialog,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from tradebot.analysis.charttypes import heikin_ashi, percent_change, renko
from tradebot.algorithms import ALGORITHMS, build_algorithm
from tradebot import economic, fundamentals
from tradebot.config import Settings
from tradebot.core.series import MARKET_COLUMNS, volume_of
from tradebot.data.market import wanted_by
from tradebot.core.timeframes import TIMEFRAMES
from tradebot.gui.common import (
    DateRange,
    FlowLayout,
    SymbolCompleter,
    epoch_seconds,
    StateOverlay,
    run_async,
    shortfall,
    toolbar_group,
)
from tradebot.gui.drawings import (
    SNAP_RADIUS,
    SNAP_RADIUS_TOUCH,
    FibItem,
    MeasureItem,
    NoteItem,
    color_of,
    set_color,
    snap_point,
    snap_time,
    to_spec,
)
from tradebot.gui.icons import bind_icon
from tradebot.gui.info import ECON_LETTERS, LETTERS, InfoPanel, event_colour
from tradebot.gui.live import LiveCandles
from tradebot.gui.news import NewsPanel
from tradebot.gui.params import ParamDialog
from tradebot.gui.depth import DepthPanel
from tradebot.gui.sidebar import ChartSidebar
from tradebot.gui.style_dialog import StyleDialog
from tradebot.gui.symbol_dialog import forward_key
from tradebot.gui.theme import Theme, ThemeManager, qcolor, resolve_color
from tradebot.gui.touch import PlotTouchHandler, TouchController
from tradebot.gui.watchlists import WatchlistPanel
from tradebot.indicators import INDICATORS, Style
from tradebot.loadouts import LoadoutStore, capture, restore
from tradebot.symbols import SymbolSearch, symbol_name

pg.setConfigOptions(antialias=True)
DASHES = {"dash": Qt.PenStyle.DashLine, "dot": Qt.PenStyle.DotLine}  #: Style.dash -> Qt pen style
FIT_PADDING = 0.05  #: auto-fit margin above and below the visible data, as a fraction of its span


def opengl_viewport():
    """An OpenGL viewport for the chart canvas (#71), or None when no OpenGL context can be made here (no
    driver, a VM, the offscreen platform, a Qt without its OpenGL modules).

    A plain ``QOpenGLWidget`` with 4x multisampling: QPainter draws through Qt's OpenGL paint engine, so pens,
    dashes, fills and antialiasing look as they do in software. (pyqtgraph's own ``GraphicsViewGLWidget``
    would switch curves to native GL drawing, which ignores pen width and antialiasing.)"""
    try:
        from PyQt6.QtGui import QOffscreenSurface, QOpenGLContext, QSurfaceFormat
        from PyQt6.QtOpenGLWidgets import QOpenGLWidget
    except ImportError:
        return None
    context, surface = QOpenGLContext(), QOffscreenSurface()
    try:
        surface.create()
        if not (surface.isValid() and context.create() and context.makeCurrent(surface)):
            return None
        context.doneCurrent()
    except Exception:  # a broken driver must never stop the app from starting
        return None
    finally:
        surface.destroy()
    widget = QOpenGLWidget()
    fmt = QSurfaceFormat()
    fmt.setSamples(4)
    widget.setFormat(fmt)
    return widget


def view_slice(item, x, pad: float = 0.0):
    """``(i0, i1, k)``: the points of the sorted ``x`` inside ``item``'s view (and ``pad`` beyond each edge, plus
    one more point, so lines leave the view instead of ending inside it), and how many of them share a pixel
    column (at least 1). Without a view: everything, one at a time."""
    vb = item.getViewBox()
    if not isinstance(vb, pg.ViewBox):
        return 0, len(x), 1
    x0, x1 = vb.viewRange()[0]
    i0 = max(int(np.searchsorted(x, x0 - pad)) - 1, 0)
    i1 = min(int(np.searchsorted(x, x1 + pad)) + 1, len(x))
    return i0, i1, max(int((i1 - i0) / max(vb.width(), 1.0)), 1)


def bucket_edges(i0: int, i1: int, k: int):
    """``(start, edges, last)`` of the points ``i0 <= i < i1`` taken ``k`` at a time, the groups aligned to multiples
    of ``k`` so they stay the same while panning (no shimmer): ``edges`` are the first index of each group,
    ``last`` the last index of each group."""
    start = i0 - i0 % k
    edges = np.arange(start, i1, k)
    return start, edges, np.minimum(edges + k, i1) - 1


def peak_buckets(x, y, i0: int, i1: int, k: int):
    """The points ``i0 <= i < i1`` of a line as the lowest and highest value of every ``k`` (one pixel column), at
    the middle of the group: every spike stays visible."""
    start, edges, last = bucket_edges(i0, i1, k)
    lo = np.minimum.reduceat(y[start:i1], edges - start)
    hi = np.maximum.reduceat(y[start:i1], edges - start)
    return np.repeat((x[edges] + x[last]) / 2, 2), np.column_stack([lo, hi]).ravel()


def _to_device(p: QPainter, x, y):
    """View coordinates -> the painter's device pixels, so the painter can draw without a transform: Qt strokes
    a wide cosmetic pen two to three times faster in device coordinates than through the view's scaling."""
    return _to_device_tr(p.transform(), x, y)


def _to_device_tr(tr, x, y):
    return x * tr.m11() + y * tr.m21() + tr.m31(), x * tr.m12() + y * tr.m22() + tr.m32()


def _point_array(x, y):
    arr = PrimitiveArray(QPointF, 2)
    arr.resize(len(x))
    mem = arr.ndarray()
    mem[:, 0], mem[:, 1] = x, y
    return arr


class Curve(pg.GraphicsObject):
    """A line drawn at screen resolution (#68, #228): only its points in view, and with more points than pixels the
    lowest and highest of every pixel column ("peak" downsampling, so no spike is lost). Antialiased only while
    there are fewer points than pixels: smoothing a dense line shows no difference but costs many times the time.

    Painted in device pixels, and a dense wide line as separate segments: both are much faster in Qt's raster
    engine than a transformed polyline. NaNs are dropped (the line joins over them, as before)."""

    def __init__(self, x, y, pen=None, name: str | None = None):
        super().__init__()
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        self.xData, self.yData = x[ok], y[ok]
        self.opts = {"pen": pg.mkPen(pen if pen is not None else pg.getConfigOption("foreground")), "name": name,
                     "antialias": False}
        self._cache = (None, None)  # ((i0, i1, k), (x, y) to draw)
        if len(self.xData):
            lo, hi = self.yData.min(), self.yData.max()
            self._data_rect = QRectF(self.xData[0], lo, self.xData[-1] - self.xData[0], hi - lo)
        else:
            self._data_rect = QRectF()

    def name(self):
        return self.opts["name"]

    def getData(self):
        return self.xData, self.yData

    def dataBounds(self, ax, frac=1.0, orthoRange=None):
        d = self.xData if ax == 0 else self.yData
        return (float(d.min()), float(d.max())) if len(d) else (None, None)

    def boundingRect(self):
        # the data, plus the pen's width in view units, so partial repaints never cut its edge
        pw = self.opts["pen"].widthF() or 1.0
        dx, dy = (self.pixelWidth() or 0.0) * pw, (self.pixelHeight() or 0.0) * pw
        return self._data_rect.adjusted(-dx, -dy, dx, dy)

    def viewTransformChanged(self):
        self.prepareGeometryChange()  # the pen's margin in view units changed

    def visible(self):
        """``(x, y)`` as drawn now: the points in view, peak-downsampled when they're denser than the pixels."""
        x, y = self.xData, self.yData
        i0, i1, k = view_slice(self, x)
        vb = self.getViewBox()
        self.opts["antialias"] = bool(pg.getConfigOption("antialias")) and (
            isinstance(vb, pg.ViewBox) and i1 - i0 <= vb.width())
        key = (i0, i1, k)
        if self._cache[0] != key:
            pts = peak_buckets(x, y, i0, i1, k) if k > 1 else (x[i0:i1], y[i0:i1])
            self._cache = (key, pts)
        return self._cache[1]

    def paint(self, p, *args):
        x, y = self.visible()
        if len(x) < 2:
            return
        pen = self.opts["pen"]
        aa = self.opts["antialias"]
        dx, dy = _to_device(p, x, y)
        p.save()
        p.resetTransform()
        p.setRenderHint(QPainter.RenderHint.Antialiasing, aa)
        p.setPen(pen)
        if pen.widthF() > 1.0 and pen.style() == Qt.PenStyle.SolidLine and pen.isSolid() and pen.color().alpha() == 255:
            # separate segments: Qt strokes a wide polyline (joins and all) five to ten times slower. Their ends
            # overlap, which only shows on a see-through pen (excluded); smoothed, flat ends keep the overlap to the
            # soft edge pixels of each joint, indistinguishable from a polyline at 1:1 (#228)
            if aa:
                pen = QPen(pen)
                pen.setCapStyle(Qt.PenCapStyle.FlatCap)
                p.setPen(pen)
            lines = PrimitiveArray(QLineF, 4)
            lines.resize(len(dx) - 1)
            mem = lines.ndarray()
            mem[:, 0], mem[:, 1], mem[:, 2], mem[:, 3] = dx[:-1], dy[:-1], dx[1:], dy[1:]
            p.drawLines(*lines.drawargs())
        else:
            p.drawPolyline(*_point_array(dx, dy).drawargs())
        p.restore()


class Band(pg.GraphicsObject):
    """The area between two lines (e.g. inside the Bollinger bands), drawn at screen resolution like
    :class:`Curve`: only the part in view, and with more points than pixels, one column per pixel from the lowest
    to the highest of both lines. Points where either line is NaN are left out."""

    def __init__(self, x, a, b, brush):
        super().__init__()
        x, a, b = (np.asarray(v, dtype=float) for v in (x, a, b))
        ok = np.isfinite(x) & np.isfinite(a) & np.isfinite(b)
        self.x, self.a, self.b = x[ok], a[ok], b[ok]
        self._brush = pg.mkBrush(brush)
        self._cache = (None, None)
        if len(self.x):
            lo, hi = min(self.a.min(), self.b.min()), max(self.a.max(), self.b.max())
            self._bounds = QRectF(self.x[0], lo, self.x[-1] - self.x[0], hi - lo)
        else:
            self._bounds = QRectF()

    def brush(self):
        return self._brush

    def boundingRect(self):
        return QRectF(self._bounds)

    def _outline(self):
        i0, i1, k = view_slice(self, self.x)
        if self._cache[0] != (i0, i1, k):
            x, a, b = self.x, self.a, self.b
            if k > 1:
                start, edges, last = bucket_edges(i0, i1, k)
                xs = (x[edges] + x[last]) / 2
                top = np.maximum.reduceat(np.maximum(a[start:i1], b[start:i1]), edges - start)
                bottom = np.minimum.reduceat(np.minimum(a[start:i1], b[start:i1]), edges - start)
            else:
                xs, top, bottom = x[i0:i1], a[i0:i1], b[i0:i1]
            self._cache = ((i0, i1, k), (np.r_[xs, xs[::-1]], np.r_[top, bottom[::-1]]))
        return self._cache[1]

    def paint(self, p, *args):
        x, y = self._outline()
        if len(x) < 4:
            return
        dx, dy = _to_device(p, x, y)
        p.save()
        p.resetTransform()
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)  # a soft-edged see-through area: no visible steps
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(self._brush)
        p.drawPolygon(*_point_array(dx, dy).drawargs(), Qt.FillRule.OddEvenFill)  # crossing lines fill both lobes
        p.restore()


def _decimals(v: float) -> int:
    """About five significant digits: 93,564 · 184.20 · 0.12345."""
    v = abs(float(v))
    return 2 if v == 0 or not np.isfinite(v) else int(np.clip(4 - np.floor(np.log10(v)), 0, 8))


def fmt_value(v) -> str:
    """A value for the cursor readout; NaN (e.g. an indicator's warm-up) shows as a dash."""
    v = float(v)
    if not np.isfinite(v):
        return "–"
    return f"{v:,.0f}" if v.is_integer() else f"{v:,.{_decimals(v)}f}"  # levels read "66", not "66.000"


def fmt_amount(v) -> str:
    """A volume for the cursor readout, abbreviated: 950 · 12.3K · 1.2M · 4.5B; NaN shows as a dash."""
    v = float(v)
    if not np.isfinite(v):
        return "–"
    for div, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(v) >= div:
            return f"{v / div:.{1 if abs(v) >= 10 * div else 2}f}{suffix}"
    return fmt_value(v)


def _spaced_label(name: str, values: dict) -> str:
    """``rsi 14`` / ``macd 12 26 9``: the name followed by the parameter values, as in the readout."""
    return " ".join([name, *(f"{v:g}" if isinstance(v, float) else str(v) for v in values.values())])


def nearest_bar(x, t: float) -> int:
    """Index of the bar in the sorted ``x`` closest to time ``t``."""
    i = int(np.clip(np.searchsorted(x, t), 1, len(x) - 1)) if len(x) > 1 else 0
    return i - 1 if len(x) > 1 and t - x[i - 1] < x[i] - t else i


def visible_y_range(x, series, x0, x1, padding=FIT_PADDING):
    """Padded (low, high) of all ``series`` (arrays aligned with the sorted ``x``) over ``x0 <= x <= x1``.

    NaNs (and infinities, e.g. log10(0)) are ignored; returns None if nothing is visible.
    """
    return RangeIndex(x, series).y_range(x0, x1, padding)


class RangeIndex:
    """The lowest and highest value of several series (aligned with the sorted ``x``) over any x-range, for the
    auto-fit on every scroll step (#228): the series are folded into one low and one high per point, and those
    into blocks, so a query reads about ``n / BLOCK + 2 * BLOCK`` numbers instead of every visible value of every
    series. NaNs and infinities (e.g. log10(0)) are ignored."""

    BLOCK = 64

    def __init__(self, x, series):
        self.x = np.asarray(x, dtype=float)
        n, b = len(self.x), self.BLOCK
        lo, hi = np.full(n, np.inf), np.full(n, -np.inf)
        for s in series:
            s = np.asarray(s, dtype=float)
            ok = np.isfinite(s)
            np.minimum(lo, np.where(ok, s, np.inf), out=lo)
            np.maximum(hi, np.where(ok, s, -np.inf), out=hi)
        self.lo, self.hi = lo, hi
        whole = n - n % b
        self.block_lo = lo[:whole].reshape(-1, b).min(axis=1) if whole else np.empty(0)
        self.block_hi = hi[:whole].reshape(-1, b).max(axis=1) if whole else np.empty(0)

    def bounds(self, i0: int, i1: int):
        """(low, high) over the points ``i0 <= i < i1``, or None when none of them has a value."""
        b = self.BLOCK
        a, z = -(-i0 // b), i1 // b  # the whole blocks inside
        if a >= z:
            parts = [(self.lo[i0:i1], self.hi[i0:i1])]
        else:
            parts = [(self.lo[i0:a * b], self.hi[i0:a * b]), (self.block_lo[a:z], self.block_hi[a:z]),
                     (self.lo[z * b:i1], self.hi[z * b:i1])]
        lo = min((float(p.min()) for p, _ in parts if p.size), default=np.inf)
        hi = max((float(p.max()) for _, p in parts if p.size), default=-np.inf)
        return (lo, hi) if lo <= hi else None

    def y_range(self, x0, x1, padding=FIT_PADDING):
        """Padded (low, high) over ``x0 <= x <= x1``, or None if nothing there has a value."""
        found = self.bounds(int(np.searchsorted(self.x, x0, "left")), int(np.searchsorted(self.x, x1, "right")))
        if found is None:
            return None
        lo, hi = found
        span = (hi - lo) or abs(hi) * 0.01 or 1.0  # flat data: open up a small band around it
        return lo - padding * span, hi + padding * span


class PlotAutoFit(QObject):
    """The chart's ``↕`` y-axis auto-fit for one ``PlotWidget`` (#96), as in the Backtest lab (#74).

    A toggle in the plot's bottom-left corner, on by default and remembered under ``key``. While it's on, the
    y-axis fits the :meth:`set_data` series inside the visible x-range, mouse and touch move only the x-axis,
    and the view can't be panned past the data. A double click or double tap shows everything, still fitted.
    """

    def __init__(self, plot: pg.PlotWidget, settings: Settings | None = None, key: str | None = None,
                 touch: bool = False, tip: str = "Auto-fit value axis to visible curves"):
        super().__init__(plot)
        self.plot, self.settings, self.key = plot, settings, key
        self.x = np.empty(0)
        self.series: list = []  # arrays aligned with self.x, in view coordinates (log10 on the log scale)
        self._index = RangeIndex(self.x, [])
        self._adjusting = False
        self.button = QToolButton(plot, text="↕", checkable=True, toolTip=tip, objectName="ChartCorner")
        self.button.setChecked(bool(settings.get(key, True)) if settings and key else True)
        size = 40 if touch else 28  # finger-sized in touch mode
        self.button.setFixedSize(size, size)
        self.button.toggled.connect(self._toggled)
        self.vb.sigRangeChanged.connect(self._range_changed)
        plot.scene().sigMouseClicked.connect(lambda ev: ev.double() and self.reset())
        self.touch = TouchController(plot.viewport(), PlotTouchHandler(plot, lambda _: plot.getPlotItem(),
                                                                       on_reset=self.reset))
        plot.installEventFilter(self)  # keeps the button in the bottom-left corner

    @property
    def vb(self) -> pg.ViewBox:
        return self.plot.getPlotItem().vb

    @property
    def on(self) -> bool:
        return self.button.isChecked()

    def set_data(self, x, series, log: bool = False):
        """The curves to fit: sorted ``x`` and arrays aligned with it (NaN where a curve has no value).
        Then everything is shown, fitted while auto-fit is on."""
        self.x = np.asarray(x, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            self.series = [np.log10(s) if log else np.asarray(s, dtype=float) for s in series]
        self._index = RangeIndex(self.x, self.series)
        self._lock()
        self.reset()

    def reset(self):
        """Everything in view, still fitted while auto-fit is on (double click or tap, new data)."""
        plot = self.plot.getPlotItem()
        self._adjusting = True
        try:
            plot.autoRange()  # autoRange() disables auto range in pyqtgraph, so re-enable afterwards
            plot.enableAutoRange()
        finally:
            self._adjusting = False
        self.fit_y()

    def fit_y(self, force: bool = False):
        """Fit the y-axis to the series inside the visible x-range (while auto-fit is on, or ``force``)."""
        if not (self.on or force):
            return
        x0, x1 = self.vb.viewRange()[0]
        self._adjusting = True
        try:
            self.plot.getPlotItem().enableAutoRange(y=False)  # pyqtgraph would otherwise fit y to all data
            bounds = self._index.y_range(x0, x1)
            if bounds is not None:
                self.plot.setYRange(*bounds, padding=0)
        finally:
            self._adjusting = False

    def _range_changed(self, _vb, _ranges, changed):
        if changed[0] and self.on and not self._adjusting:
            self.fit_y()  # scrolled or zoomed along x

    def _lock(self):
        """While auto-fit is on, mouse and touch move only the x-axis, and not past the data."""
        on = self.on and len(self.x) > 0
        margin = float(np.median(np.diff(self.x))) if len(self.x) > 1 else 1.0
        self.vb.setMouseEnabled(x=True, y=not on)
        self.vb.setLimits(**(dict(xMin=self.x[0] - margin, xMax=self.x[-1] + margin) if on
                             else dict(xMin=None, xMax=None)))

    def _toggled(self, on):
        if self.settings and self.key:
            self.settings.set(self.key, on)
        self._lock()
        if on:
            self.fit_y()
        else:
            self.plot.getPlotItem().enableAutoRange(y=False)  # keep the current y-range until the user changes it

    def eventFilter(self, obj, event):
        if obj is getattr(self, "plot", None) and event.type() == QEvent.Type.Resize:  # (gone while Qt tears down)
            self.button.move(8, self.plot.height() - self.button.height() - 8)
        return super().eventFilter(obj, event)


def price_label(p: float) -> str:
    """A price-axis tick: 100,000 · 250 · 0.05."""
    if p >= 1000:
        return f"{round(p):,}"
    return np.format_float_positional(float(f"{p:.6g}"), trim="-")


class PriceAxis(pg.AxisItem):
    """Price axis that, on the log scale (#60), labels a log10 view with real prices at round values."""

    log = False
    MIN_SPACING = 40  #: pixels between labels

    def tickValues(self, minVal, maxVal, size):
        if not self.log:
            return super().tickValues(minVal, maxVal, size)
        fits = max(int(size / self.MIN_SPACING), 2)
        decades = range(int(np.floor(minVal)) - 1, int(np.ceil(maxVal)) + 1)
        for steps in ((1, 2, 5), (1, 3), (1,)):  # 1-2-5 per decade while they fit, then fewer
            ticks = [np.log10(m * 10.0 ** d) for d in decades for m in steps]
            ticks = [v for v in ticks if minVal <= v <= maxVal]
            if 2 <= len(ticks) <= fits:
                return [(1.0, ticks)]
        if len(ticks) > fits:  # many decades: every n-th power of ten
            return [(1.0, ticks[:: int(np.ceil(len(ticks) / fits))])]
        # less than about a factor of 2.5 in view: round linear steps of the real prices
        levels = super().tickValues(10.0 ** minVal, 10.0 ** maxVal, size)
        return [(1.0, [np.log10(v) for v in values if v > 0]) for _, values in levels]

    def tickStrings(self, values, scale, spacing):
        if not self.log:
            return super().tickStrings(values, scale, spacing)
        return [price_label(10.0 ** v) for v in values]


class DateAxis(pg.DateAxisItem):
    """The time axis. The x-linked panes show the same times, so they share the (slow, time-zone aware) tick
    computation: it is done once per view range instead of once per pane (#68)."""

    _shared: dict = {}  # (kind, arguments) -> result, of the latest few calls by any pane

    def _cached(self, kind, key, compute):
        key = (kind, *key)
        if key not in DateAxis._shared:
            if len(DateAxis._shared) > 16:  # only the current frame's ticks are worth keeping
                DateAxis._shared.clear()
            DateAxis._shared[key] = compute()
        return DateAxis._shared[key]

    MIN_GRID_PX = 24  #: tick levels whose lines would be closer than this are left out (#228)

    def tickValues(self, minVal, maxVal, size):
        key = (minVal, maxVal, size, self.fontMetrics.height(), self.utcOffset)
        values, self.zoomLevel, self.minSpacing = self._cached("values", key, lambda: (
            self._sparse_levels(super(DateAxis, self).tickValues(minVal, maxVal, size), maxVal - minVal, size),
            self.zoomLevel, self.minSpacing))
        return values

    @classmethod
    def _sparse_levels(cls, levels, span, size):
        """The tick levels without the dense ones (the coarsest always stays). Lines that close are too faint to see
        and too tight to be labelled, but each costs a span per pixel row in every pane on every scroll step: at 76
        monthly lines per pane they took 8 ms of a 20 ms frame."""
        if span <= 0 or size <= 0:
            return levels
        kept = [(spacing, ticks) for spacing, ticks in levels if spacing / span * size >= cls.MIN_GRID_PX]
        return kept or levels[:1]

    def tickStrings(self, values, scale, spacing):
        key = (tuple(values), spacing, id(self.zoomLevel), self.utcOffset)
        return self._cached("strings", key, lambda: super(DateAxis, self).tickStrings(values, scale, spacing))


class _PriceFormat:
    """Stands in for an InfiniteLine label's format string: shows the price at the line's view y."""

    def __init__(self, chart: ChartView):
        self.chart = chart

    def format(self, value):
        return f"{self.chart.price_at(value):.6g}"


def _offline_note(df) -> str:
    return " · offline: showing cached candles, they may be outdated" if df.attrs.get("offline") else ""


def ohlc_buckets(x, o, h, l, c, i0: int, i1: int, k: int):
    """Candles ``i0 <= i < i1`` merged ``k`` at a time (first open, highest high, lowest low, last close), the
    groups as in :func:`bucket_edges`; x is the middle of each group."""
    start, edges, last = bucket_edges(i0, i1, k)
    return ((x[edges] + x[last]) / 2, o[edges], np.maximum.reduceat(h[start:i1], edges - start),
            np.minimum.reduceat(l[start:i1], edges - start), c[last])


def _rects(x, bottom, top, width):
    """``(n, 4)`` rectangles ``(left, bottom, width, height)`` for QPainter.drawRects(), far faster than filling a
    path of them (#68); a flat one is still drawn: the pen strokes it."""
    height = np.where(top == bottom, 1e-12, top - bottom)
    return np.column_stack([x - width / 2, bottom, np.full(len(x), float(width)), height])


def _wicks(x, low, high):
    """``(n, 4)`` vertical lines ``(x, low, x, high)`` for QPainter.drawLines()."""
    return np.column_stack([x, low, x, high])


def _primitives(kind, rows):
    """An ``(n, 4)`` array as a QLineF or QRectF array that QPainter draws in one call, without a Python object
    per row."""
    arr = PrimitiveArray(kind, 4)
    arr.resize(len(rows))
    arr.ndarray()[:] = rows
    return arr


def _device_rects(tr, rects):
    """``(left, bottom, width, height)`` rows mapped by a scale-and-move transform, with positive sizes."""
    x, w = rects[:, 0] * tr.m11() + tr.m31(), rects[:, 2] * tr.m11()
    y, h = rects[:, 1] * tr.m22() + tr.m32(), rects[:, 3] * tr.m22()
    x, w = np.where(w < 0, x + w, x), np.abs(w)
    y, h = np.where(h < 0, y + h, y), np.abs(h)
    return np.column_stack([x, y, w, h])


def _device_lines(tr, lines):
    x0, y0 = _to_device_tr(tr, lines[:, 0], lines[:, 1])
    x1, y1 = _to_device_tr(tr, lines[:, 2], lines[:, 3])
    return np.column_stack([x0, y0, x1, y1])


class ZonesItem(pg.GraphicsObject):
    """Horizontal bands (support / resistance, #164): each run of one level is a flat band from the bar it starts
    on to the bar it ends on, with a thin line at the level, so levels read as areas instead of lines that slant
    from one level to the next."""

    def __init__(self, segments, color: QColor, text: str, text_color: QColor):
        super().__init__()
        #: ``(x0, x1, low, high, level)`` per run
        self.segments = segments
        fill = QColor(color)
        fill.setAlpha(46)
        self._fill, self._line, self._text, self._tcolor = fill, color, text, text_color
        self._bounds = QRectF()
        if segments:
            x0, x1 = min(s[0] for s in segments), max(s[1] for s in segments)
            lo, hi = min(s[2] for s in segments), max(s[3] for s in segments)
            self._bounds = QRectF(x0, lo, x1 - x0, max(hi - lo, 1e-12))
        self.setZValue(-4)

    def boundingRect(self):
        return QRectF(self._bounds)

    def paint(self, p, *_):
        p.setPen(pg.mkPen(None))
        p.setBrush(pg.mkBrush(self._fill))
        for x0, x1, lo, hi, _level in self.segments:
            p.drawRect(QRectF(x0, lo, x1 - x0, max(hi - lo, 1e-12)))
        pen = pg.mkPen(self._line, width=1)
        pen.setCosmetic(True)
        p.setPen(pen)
        for x0, x1, _lo, _hi, level in self.segments:
            p.drawLine(QPointF(x0, level), QPointF(x1, level))


def zone_segments(x, y, half, width_x):
    """``(x0, x1, low, high, level)`` per run of one level in ``y`` (NaN ends a run); ``half`` is the band's half
    height in price (an array like ``y``), ``width_x`` the space one candle takes on the x-axis."""
    y = np.asarray(y, dtype=float)
    out = []
    start = None
    for i in range(len(y) + 1):
        cur = y[i] if i < len(y) else np.nan
        if start is not None and not (np.isfinite(cur) and cur == y[start]):
            level = y[start]
            out.append((x[start] - width_x / 2, x[i - 1] + width_x / 2, level - half[start], level + half[start], level))
            start = None
        if start is None and np.isfinite(cur):
            start = i
    return out


class CandlestickItem(pg.GraphicsObject):
    """Candles: only the visible ones are drawn, and with more candles than pixels, one merged candle per
    pixel (#68), so panning stays fast with tens of thousands of bars."""

    def __init__(self):
        super().__init__()
        self._data = None
        self._bounds = QRectF()
        self._cache = (None, None)  # (visible slice and bucket size, shapes per style)
        self._smooth = True  # antialiased: only while there are fewer candles than pixels, as for Curve

    def set_data(self, x, o, h, l, c, width, up, down):
        self._data = tuple(np.asarray(a, dtype=float) for a in (x, o, h, l, c))
        self.width = width
        self.pens = [pg.mkPen(color, width=1) for color in (up, down)]  # cosmetic: 1 px at any zoom
        self.brushes = [pg.mkBrush(color) for color in (up, down)]
        self.prepareGeometryChange()
        if len(x):
            lo, hi = np.nanmin(l), np.nanmax(h)
            self._bounds = QRectF(x[0] - width / 2, lo, x[-1] - x[0] + width, (hi - lo) or 1e-12)
        else:
            self._bounds = QRectF()
        self._cache = (None, None)
        self.update()

    def _shapes(self, x, o, h, l, c, width):
        """[(style index, wick lines or None, body rectangles)] for the given (visible, maybe merged) candles."""
        rising = c >= o
        return [(style, _wicks(x[m], l[m], h[m]), _rects(x[m], o[m], c[m], width))
                for style, m in ((0, rising), (1, ~rising)) if m.any()]

    def _paths(self):
        x, o, h, l, c = self._data
        vb = self.getViewBox()
        i0, i1, k = 0, len(x), 1
        if isinstance(vb, pg.ViewBox):
            x0, x1 = vb.viewRange()[0]
            i0 = max(int(np.searchsorted(x, x0 - self.width)) - 1, 0)
            i1 = min(int(np.searchsorted(x, x1 + self.width)) + 1, len(x))
            k = max(int(np.ceil((i1 - i0) / max(vb.width(), 1.0))), 1)  # candles per pixel: at most one bar each
            self._smooth = i1 - i0 <= vb.width()
        key = (i0, i1, k)
        if self._cache[0] != key:
            if i1 <= i0:
                shapes = []
            elif k > 1:
                shapes = self._shapes(*self._buckets(i0, i1, k), self.width * k)
            else:
                shapes = self._shapes(*(a[i0:i1] for a in (x, o, h, l, c)), self.width)
            self._cache = (key, shapes)
        return self._cache[1]

    def _buckets(self, i0: int, i1: int, k: int):
        """The visible candles merged ``k`` at a time (see :func:`ohlc_buckets`)."""
        return ohlc_buckets(*self._data, i0, i1, k)

    def paint(self, p, *args):
        if self._data is None:
            return
        shapes = self._paths()
        if not shapes:
            return
        tr = p.transform()
        device = tr.type().value <= QTransform.TransformationType.TxScale.value  # a view's transform: in device pixels (#228)
        p.save()
        if device:
            p.resetTransform()
        p.setRenderHint(QPainter.RenderHint.Antialiasing, self._smooth and pg.getConfigOption("antialias"))
        for style, wicks, rects in shapes:
            p.setPen(self.pens[style])
            if wicks is not None:
                p.setBrush(Qt.BrushStyle.NoBrush)  # lines only: a brush makes OpenGL paint fill between the wicks
                p.drawLines(*_primitives(QLineF, _device_lines(tr, wicks) if device else wicks).drawargs())
            p.setBrush(self.brushes[style])
            p.drawRects(*_primitives(QRectF, _device_rects(tr, rects) if device else rects).drawargs())
        p.restore()

    def boundingRect(self):
        return QRectF(self._bounds)


class HistogramItem(CandlestickItem):
    """Bars from 0 (e.g. the MACD histogram), culled and merged per pixel like the candles."""

    def set_data(self, x, y, width, brush, pen):
        y = np.asarray(y, dtype=float)
        zero = np.zeros_like(y)
        super().set_data(x, zero, np.maximum(y, 0), np.minimum(y, 0), y, width, brush, brush)
        self.pens = [pen]

    def _shapes(self, x, o, h, l, c, width):
        return [(0, None, _rects(x, l, h, width))] if len(x) else []  # a merged bar spans its group's range


def volume_buckets(x, o, v, c, i0: int, i1: int, k: int):
    """Volume bars ``i0 <= i < i1`` merged ``k`` at a time like :func:`ohlc_buckets`, their volumes summed (a gap
    counts as 0), so a merged bar is the volume traded over its candles; the first open and last close give its
    direction."""
    start, edges, last = bucket_edges(i0, i1, k)
    return (x[edges] + x[last]) / 2, o[edges], np.add.reduceat(np.nan_to_num(v[start:i1]), edges - start), c[last]


class VolumeBarsItem(CandlestickItem):
    """Volume bars from 0 in the candles' up/down colours (#116), culled and merged per pixel like the candles;
    merged bars add up their volume instead of taking the largest."""

    def set_data(self, x, volume, o, c, width, up, down):
        v = np.asarray(volume, dtype=float)
        super().set_data(x, o, v, np.zeros_like(v), c, width, up, down)

    def _buckets(self, i0: int, i1: int, k: int):
        x, o, v, _, c = self._data
        xm, om, vm, cm = volume_buckets(x, o, v, c, i0, i1, k)
        return xm, om, vm, np.zeros_like(vm), cm

    def _shapes(self, x, o, h, l, c, width):
        rising = c >= o
        return [(style, None, _rects(x[m], l[m], h[m], width)) for style, m in ((0, rising), (1, ~rising)) if m.any()]


CHART_TYPES = ("Standard", "Heikin-Ashi", "Renko", "Compare", "Ratio")


class ChartView(QWidget):
    status = pyqtSignal(str)

    def __init__(self, engine, themes: ThemeManager, watchlist: list[str], settings: Settings | None = None,
                 symbol_search: SymbolSearch | None = None, opengl: bool = False):
        super().__init__()
        self.engine, self.themes, self.settings = engine, themes, settings
        self.df: pd.DataFrame | None = None
        self.indicators: list = []
        self._market_errors: dict[tuple, str] = {}  #: (symbol, timeframe, column) -> why it can't be had (#166)
        self._market_loading: set[tuple] = set()
        self.algorithm = None
        self.drawings: list = []
        self._preview = None  # the rubber-band shape between the first click of a two-point tool and the cursor
        self._pending_point = None
        self._x = np.empty(0)
        self._fit_series: dict = {}  # plot -> the arrays its y-axis auto-fit covers
        self._fit_index: dict = {}  # plot -> RangeIndex of those arrays
        self._frames: dict = {}  # indicator key -> (candles, its frame) of the last redraw
        self._adjusting = False  # set while the chart changes view ranges itself, as opposed to the user
        self._title = ""  # "SYMBOL · 1d" of the loaded data
        self._hover: int | None = None  # bar under the cursor; None = show the latest bar
        self._rows: dict = {}  # plot -> [(label, [(values, colour), …])] for the value lines (#45)
        self._legend: dict = {}  # plot -> LabelItem in its top-left corner
        self._positions = None  # the signal's target position per bar
        self._slots: dict[int, int] = {}  # id(indicator) -> first palette index of its outputs
        self._next_slot = 0
        self._loaded_symbol = ""  # the symbol of self.df, remembered for the next start
        self._pending_note = ""  # a message from restoring the last session, shown with the first load
        self._log = False  # the price pane's view y is log10(price) (#60)

        # -- toolbar -----------------------------------------------------------------------------
        self.symbol = QLineEdit(watchlist[0] if watchlist else "BTC/USDT")
        self.symbols = SymbolCompleter(self.symbol, symbol_search or SymbolSearch(lambda: watchlist))
        self.symbols.chosen.connect(lambda _: self.load())
        self.symbol.setFixedWidth(150)
        self.symbol.returnPressed.connect(self.load)
        self.tf = QComboBox()
        self.tf.addItems(list(TIMEFRAMES))
        self.tf.setCurrentText("1d")
        self.tf.currentTextChanged.connect(lambda _: self.load())
        self.bars = QComboBox()
        self.bars.addItems(["200", "500", "1000", "2000", "5000"])
        self.bars.setCurrentText("500")
        self.bars.currentTextChanged.connect(lambda _: self.load())
        self.dates = DateRange()
        load = QPushButton("Load")
        load.setObjectName("Primary")
        load.clicked.connect(self.load)
        self.live = LiveCandles(self, themes)  # the Live button: the forming candle follows the market (#158)
        self.style_btn = QToolButton(text="Line", checkable=True)
        self.style_btn.toggled.connect(lambda _: self.redraw())
        # how the price is drawn (#163): Heikin-Ashi and Renko are display transforms, Compare overlays other symbols as
        # percent change, Ratio charts this symbol divided by the "vs" one. Strategies always use the real candles.
        self.type_box = QComboBox(toolTip="Chart type. Heikin-Ashi and Renko only change the drawing; Compare overlays "
                                          "the symbols in 'vs' as percent change; Ratio charts symbol / vs")
        self.type_box.addItems(CHART_TYPES)
        self._type = CHART_TYPES[0]
        self.type_box.currentTextChanged.connect(self._type_changed)
        self.versus = QLineEdit(placeholderText="vs: ETH/USDT, SOL/USDT", toolTip="Symbols to compare with, or the "
                                "second symbol of a ratio (comma separated)")
        self.versus.setFixedWidth(190)
        self.versus.returnPressed.connect(self.load)
        self.versus.setVisible(False)
        self.compare: dict[str, pd.DataFrame] = {}  # the other symbols' candles in Compare mode

        self.events_btn = QToolButton(text="Events", checkable=True, toolTip="Mark earnings (E), splits (S) and ex-dividend "
                                      "dates (D) of stocks under the candles, and high-impact economic events above them: "
                                      "rate decisions (R), CPI (C), jobs data (J), others (!)")
        self._econ: list = []  # economic.EconEvent of the loaded range (#167)
        self.events_btn.setChecked(bool(settings.get(self.EVENTS_KEY, False)) if settings else False)
        self.events_btn.toggled.connect(self._events_toggled)
        self._info = None  # fundamentals.Snapshot of the charted symbol, once fetched
        self._info_for = ""

        # indicators and the signal: one dialog (#230), also opened with "/" over the chart
        self.studies_btn = QToolButton(text="Indicators", toolTip="Indicators and signals (/)")
        self.studies_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        bind_icon(self.studies_btn, "layers", themes, off="muted")
        self.studies_btn.clicked.connect(lambda: self.open_studies())
        self._studies = None  # the dialog, built on first use and reused

        # drawing tools: an icon rail on the chart's left edge, part of the chart (#230); the names are in the tooltips
        self.tools: dict[str, QToolButton] = {}
        tool_group = []

        def rail_button(btn: QToolButton, glyph: str, checked_on: bool = True) -> QToolButton:
            btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)  # 32 px square, 44 in touch mode (theme)
            bind_icon(btn, glyph, themes, off="muted", on="accent" if checked_on else None, size=18)
            btn.setAccessibleName(btn.text())
            return btn
        for key, glyph, label, tip in (("trend", "trend", "Trend", "Trendline: click two points"),
                                ("hline", "hline", "Level", "Support/resistance: click a price"),
                                ("rect", "rect", "Zone", "Rectangle: click two corners"),
                                ("fib", "fib", "Fib", "Fibonacci retracement/extension: click the 0 and the 1 point"),
                                ("vline", "vline", "Time", "Vertical line: click a point in time"),
                                ("text", "text", "Note", "Text note: click where it goes, then type it"),
                                ("measure", "ruler", "Measure", "Price range measure: click two points to read the change "
                                                         "in price, percent and time")):
            btn = rail_button(QToolButton(text=label, checkable=True, toolTip=f"{label} · {tip}"), glyph)
            btn.toggled.connect(lambda on, k=key: self._tool_toggled(k, on))
            self.tools[key] = btn
            tool_group.append(btn)
        clear = rail_button(QToolButton(text="Clear", toolTip="Clear · remove all drawings of this symbol (right-click a "
                                                              "single drawing to recolour, edit or delete it)"), "trash",
                            checked_on=False)
        clear.clicked.connect(self.clear_drawings)
        self.magnet_btn = rail_button(QToolButton(text="Magnet", checkable=True, toolTip="Magnet · snap drawing points to the "
                                                  "nearest candle open, high, low or close (hold Ctrl to flip it while drawing)"),
                                      "magnet")
        self.magnet_btn.setChecked(bool(settings.get(self.MAGNET_KEY, False)) if settings else False)
        self.magnet_btn.toggled.connect(self._magnet_toggled)
        self._snap_marker = None  # dot on the value the next point would snap to
        self._snap_cache = None
        self._last_view_x = None
        self._snapping = False
        self.loadouts = LoadoutStore(settings)
        self.loadout_btn = QToolButton(text="Loadouts", toolTip="Saved sets of indicators and a signal")
        self.loadout_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        bind_icon(self.loadout_btn, "folder", themes, off="muted", size=14)
        self.loadout_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.loadout_menu = QMenu(self)
        self.loadout_menu.aboutToShow.connect(self._fill_loadout_menu)
        self.loadout_btn.setMenu(self.loadout_menu)

        # what is charted (left) and how (right); on a narrow window the second group wraps onto a row of its own
        bar = FlowLayout(spacing=12, last_right=True)
        for group in ((self.symbol, self.tf, self.bars, self.dates, load, self.live.button),
                      (self.type_box, self.versus, self.style_btn, self.events_btn, self.studies_btn, self.loadout_btn)):
            bar.addWidget(toolbar_group(*group))
        self.tool_rail = QFrame(objectName="ToolRail")
        rail = QVBoxLayout(self.tool_rail)
        rail.setContentsMargins(6, 8, 6, 8)
        rail.setSpacing(4)
        for btn in tool_group:
            rail.addWidget(btn)
        rail.addStretch(1)
        rail.addWidget(self.magnet_btn)
        rail.addWidget(clear)

        # -- plot area ---------------------------------------------------------------------------
        self.canvas = pg.GraphicsLayoutWidget()
        self.opengl = False  # whether the canvas draws through OpenGL (#71)
        self.opengl_note = ""  # why the OpenGL viewport couldn't be used, for the status bar
        if opengl:
            viewport = opengl_viewport()
            if viewport is None:
                self.opengl_note = "GPU rendering unavailable (no OpenGL context): using the normal renderer"
            else:
                self.canvas.setViewport(viewport)
                self.opengl = True
        self.price_axis = PriceAxis("right")
        self.price = self.canvas.addPlot(row=0, col=0, axisItems={"bottom": DateAxis(), "right": self.price_axis})
        self.price.showGrid(x=True, y=True, alpha=0.15)
        self.price.setMenuEnabled(False)
        self.price.getAxis("right").setWidth(70)
        self.price.getAxis("right").setStyle(maxTickLevel=0)  # minor y gridlines cost half of every pan/zoom frame
        self.price.showAxis("right")
        self.price.hideAxis("left")
        self.panes: list[pg.PlotItem] = []
        # crosshair: per pane a vertical line (shown in every pane) and a horizontal one (hovered pane only)
        self._cross: dict = {}
        self.vline, self.hline = self._add_crosshair(self.price)
        self.hline.label.format = _PriceFormat(self)
        self._add_legend(self.price)
        self.canvas.scene().sigMouseMoved.connect(self._mouse_moved)
        self.canvas.scene().sigMouseClicked.connect(self._mouse_clicked)
        self.price.vb.sigRangeChanged.connect(self._range_changed)

        # the chart-corner buttons: no padding or min-height from the theme, so both are exactly `size` square
        self.fit_btn = QToolButton(self.canvas, text="↕", checkable=True, toolTip="Auto-fit price axis to visible candles",
                                   objectName="ChartCorner")
        # on by default (a checked= keyword is applied before checkable and lost)
        self.fit_btn.setChecked(bool(settings.get("chart.autofit", True)) if settings else True)
        size = 40 if themes.touch else 28  # finger-sized in touch mode
        self.fit_btn.setFixedSize(size, size)
        self.fit_btn.toggled.connect(self._autofit_toggled)
        self.log_btn = QToolButton(self.canvas, text="log", checkable=True, toolTip="Logarithmic price scale",
                                   objectName="ChartCorner")
        self.log_btn.setChecked(bool(settings.get("chart.log", False)) if settings else False)
        self.log_btn.setFixedSize(size, size)
        self.log_btn.toggled.connect(self._log_toggled)
        # the volume pane (#116) is the "volume" indicator under the candles, so loadouts and the session keep it
        self.vol_btn = QToolButton(self.canvas, text="vol", checkable=True, toolTip="Volume pane under the candles",
                                   objectName="ChartCorner")
        self.vol_btn.setFixedSize(size, size)
        self.vol_btn.toggled.connect(self._volume_toggled)
        self.state = StateOverlay(themes, self.canvas)  # loading / error message over the plot (#191)
        self.canvas.installEventFilter(self)  # keeps the button in the bottom-left corner

        # -- sidebar at the right (#41): news for the charted symbol, the watchlists (#67) ----------------
        self.sidebar = ChartSidebar(settings, touch=themes.touch, themes=themes)
        self.news = NewsPanel(themes)
        self.sidebar.add_category("news", "news", "News", self.news)
        self.watchlists = WatchlistPanel(engine, themes, settings, symbol_search)
        self.watchlists.open_symbol.connect(self.open_symbol)
        self.watchlists.status.connect(self.status)
        self.sidebar.add_category("watchlists", "star", "Watchlists", self.watchlists)
        self.depth = DepthPanel(engine, themes)  # the order book of a crypto pair (#166)
        self.sidebar.add_category("depth", "depth", "Order book depth", self.depth)
        self.info = InfoPanel(themes)  # last, so the pages that were there keep their place in the strip
        self.sidebar.add_category("info", "info", "Fundamentals", self.info)
        self.sidebar.shown.connect(self._sidebar_shown)
        self.sidebar.toggle.clicked.connect(self._sidebar_toggled)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.addLayout(bar)
        body = QHBoxLayout()
        body.setSpacing(10)
        chart_frame = QFrame(objectName="ChartFrame")  # the rail and the plot: one surface
        cl = QHBoxLayout(chart_frame)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(0)
        cl.addWidget(self.tool_rail)
        cl.addWidget(self.canvas, 1)
        body.addWidget(chart_frame, 1)
        body.addWidget(self.sidebar)
        lay.addLayout(body, 1)
        self.touch = TouchController(self.canvas.viewport(), PlotTouchHandler(
            self.canvas, self._plot_at, on_tap=self._tap, on_long_press=self._context_menu, on_reset=self.reset_view))
        themes.changed.connect(lambda _: self.redraw())
        self._restore_session()

    # -- data ------------------------------------------------------------------------------------
    def open_symbol(self, symbol: str):
        self.symbol.setText(symbol)
        self.load()

    def reload_indicators(self, names: list[str]):
        """Scripts were reloaded: the indicators of these names are redrawn from the new classes."""
        from tradebot.indicators import INDICATORS

        changed = False
        for i, ind in enumerate(self.indicators):
            if ind.name in names and type(ind) is not INDICATORS.get(ind.name):
                try:
                    self.indicators[i] = INDICATORS.get(ind.name)(tf=ind.tf, **ind.values)
                except Exception:  # the inputs no longer fit the new version: its defaults
                    self.indicators[i] = INDICATORS.get(ind.name)(tf=ind.tf)
                changed = True
        if changed:
            self.redraw()

    def show_scanned(self, symbol: str, timeframe: str, indicator: str, params: dict):
        """Chart a Scanner result: its timeframe, with the scanned indicator added (once)."""
        from tradebot.indicators import INDICATORS

        self.tf.blockSignals(True)
        self.tf.setCurrentText(timeframe)
        self.tf.blockSignals(False)
        ind = INDICATORS.get(indicator)(**params)  # params may hold "tf" (a higher timeframe, #161)
        if not any(i.name == ind.name and i.values == ind.values and i.tf == ind.tf for i in self.indicators):
            self.indicators.append(ind)
        self.open_symbol(symbol)

    def load(self):
        symbol, tf, bars = self.symbol.text().strip(), self.tf.currentText(), int(self.bars.currentText())
        if not symbol:
            return
        start, end = self.dates.range()
        others = self._others()
        if self._type == "Ratio" and others:  # the engine builds "A:B" from both legs
            symbol = f"{symbol}:{others[0]}"
        self.status.emit(f"loading {symbol} {tf} …")
        self.compare = {k: v for k, v in self.compare.items() if k in others and self._type == "Compare"}
        if self._type == "Compare":
            for other in others:
                run_async(lambda o=other: self.engine.get_ohlc(o, tf, start=start, end=end, bars=bars),
                          lambda df, o=other: self._compared(o, df), lambda err, o=other: self.status.emit(f"{o}: {err}"))
        cached = self.engine.peek(symbol, tf, start=start, end=end, bars=bars)  # from disk, no network (#63)
        if cached is not None:
            self._loaded(symbol, cached, " · from the cache, updating …")
        elif self.df is None:
            self.state.show_loading(f"Loading {symbol} {tf} …")  # nothing to show yet

        def failed(err):
            self.status.emit(f"{symbol}: {err}")
            if cached is None:  # nothing on screen to fall back on: say so on the chart, with a way to try again
                self.state.show_error(f"Couldn't load {symbol} {tf}", str(err), self.load)

        run_async(lambda: self.engine.get_ohlc(symbol, tf, start=start, end=end, bars=bars),
                  lambda df: self._loaded(symbol, df, shortfall(df, bars, start) + _offline_note(df),
                                          refresh=cached is not None),
                  failed)

    def _others(self) -> list[str]:
        return [t for t in (p.strip() for p in self.versus.text().split(",")) if t]

    def _type_changed(self, kind: str):
        old, self._type = self._type, kind
        self.versus.setVisible(kind in ("Compare", "Ratio"))
        if self.settings is not None and kind not in ("Compare", "Ratio"):
            self.settings.set("chart.type", kind)
        if "Ratio" in (old, kind) or kind == "Compare" and self._others():
            self.load()  # other data is needed (or the plain symbol again)
        else:
            self.redraw()

    def _compared(self, symbol: str, df: pd.DataFrame):
        if self._type == "Compare" and symbol in self._others():
            self.compare[symbol] = df
            self.redraw()

    def _loaded(self, symbol, df, note="", refresh=False):
        """Show loaded candles; ``refresh``: newer candles for what the chart already shows from the cache,
        so the view stays where it is."""
        if refresh and (symbol != self._loaded_symbol or self.df is None):
            return  # the user moved on to another symbol meanwhile
        self.df = df
        self.state.clear()
        if symbol != self._loaded_symbol:
            self._switch_drawings(symbol)
        self._loaded_symbol = symbol
        note, self._pending_note = note + self._pending_note, ""
        tf = self.tf.currentText()
        self._title = f"{symbol} · {tf}"  # the ticker until the full name arrives
        self._hover = None
        self.news.set_symbol(symbol)
        self.watchlists.set_symbol(symbol)
        self.depth.set_symbol(symbol)
        self._ensure_market()
        self.info.set_symbol(symbol)
        self._info, self._info_for = (self._info if self._info_for == symbol else None), symbol
        self._fetch_info()
        self._fetch_econ()
        run_async(lambda: symbol_name(symbol), lambda name: self._named(symbol, tf, name))
        self.status.emit(f"{symbol}: {len(df)} candles {df.index[0]:%Y-%m-%d} → {df.index[-1]:%Y-%m-%d %H:%M}{note}")
        self.redraw(autorange=not refresh)
        self.live.follow()

    def _market_wanted(self) -> list[str]:
        """Crypto market data columns (#166) the indicators and the algorithm on the chart read."""
        inds = list(self.indicators)
        try:
            inds += self.algorithm.chart_indicators() if self.algorithm is not None else []
        except Exception:  # noqa: BLE001 - an algorithm that can't tell its indicators needs no extra data
            pass
        return wanted_by(inds)

    def _ensure_market(self):
        """Fetch, in the background, the market data columns the chart's indicators need and the candles lack."""
        df, symbol, tf = self.df, self._loaded_symbol, self.tf.currentText()
        if df is None or not symbol:
            return
        cols = [c for c in self._market_wanted() if c not in df.columns and (symbol, tf, c) not in self._market_errors]
        if not cols or (symbol, tf) in self._market_loading:
            return
        self._market_loading.add((symbol, tf))
        market = self.engine.market

        def done(out):
            self._market_loading.discard((symbol, tf))
            for c, why in out.attrs.get("market_errors", {}).items():
                self._market_errors[(symbol, tf, c)] = why
            if self.df is None or self._loaded_symbol != symbol or self.tf.currentText() != tf:
                return
            got = [c for c in out.columns if c in MARKET_COLUMNS and c not in self.df.columns]
            if got:
                self.df = self.df.join(out[got])  # candles that arrived meanwhile stay NaN until the next load
            for c in cols:  # whatever didn't arrive is not asked for again
                if c not in self.df.columns:
                    self._market_errors.setdefault((symbol, tf, c), "not available")
            self.redraw()

        def failed(err):
            self._market_loading.discard((symbol, tf))
            for c in cols:
                self._market_errors[(symbol, tf, c)] = str(err)
            self.redraw()

        run_async(lambda: market.attach(df, symbol, cols), done, failed)

    def _named(self, symbol, tf, name):
        """Full instrument name for the price pane's title, if the chart still shows that symbol."""
        if name and self._title == f"{symbol} · {tf}":
            self._title = f"{name} · {tf}"
            self._show_values()
            self.news.set_symbol(symbol, name)
            self.info.set_symbol(symbol, name)

    # -- sidebar ---------------------------------------------------------------------------------
    # -- fundamentals and events (#165) ------------------------------------------------------------
    EVENTS_KEY = "chart.events"

    def _info_wanted(self) -> bool:
        panel = self.sidebar.is_open and self.sidebar.current == "info"
        return bool(self._loaded_symbol) and fundamentals.supported(self._loaded_symbol) and (panel or self.events_btn.isChecked())

    def _fetch_info(self):
        """Fetch the symbol's fundamentals and events off the GUI thread (cached for a day), when the info page or
        the event markers need them."""
        symbol = self._loaded_symbol
        if not self._info_wanted() or (self._info is not None and self._info.symbol == symbol):
            if self._info is not None and self._info.symbol == symbol:
                self.info.show_snapshot(self._info)
            return
        run_async(lambda: fundamentals.load(symbol), lambda snap: self._info_loaded(symbol, snap))

    def _info_loaded(self, symbol, snap):
        if symbol != self._loaded_symbol:
            return
        self._info = snap
        self.info.show_snapshot(snap)
        if self.events_btn.isChecked():
            self.redraw()

    def _events_toggled(self, on):
        if self.settings is not None:
            self.settings.set(self.EVENTS_KEY, bool(on))
        if on and self._info is None:
            self._fetch_info()
        else:
            self.redraw()
        if on:
            self._fetch_econ()
        if on and self._loaded_symbol and not fundamentals.supported(self._loaded_symbol):
            self.status.emit("no events for crypto, pairs or indices")

    def _fetch_econ(self):
        """The economic calendar of the loaded range, in the background (the feed is asked at most once an hour)."""
        df = self.df
        if not self.events_btn.isChecked() or df is None or len(df) < 2:
            return
        lo, hi = df.index[0], df.index[-1] + pd.Timedelta(days=1)
        symbol = self._loaded_symbol

        def done(events):
            if symbol == self._loaded_symbol:
                self._econ = events
                self.redraw()

        run_async(lambda: economic.load(lo, hi, min_impact="high"), done)

    def _draw_econ(self, x, h, ty, t):
        """R / C / J / ! marks above the candles of the high-impact economic events; one per candle."""
        if not self._econ or len(x) < 2:
            return
        step = float(np.median(np.diff(x)))
        spots: dict[int, list] = {}
        for e in self._econ:
            ts = e.time.timestamp()
            if x[0] - step <= ts <= x[-1] + step:
                spots.setdefault(nearest_bar(x, ts), []).append(e)
        for i, events in sorted(spots.items()):
            e = min(events, key=lambda e: economic.KINDS.index(e.kind))
            colour = qcolor(t.down if e.kind == "rates" else t.accent)
            y = ty(h[i] * 1.015) if h[i] > 0 else ty(h[i])
            item = pg.ScatterPlotItem(x=[x[i]], y=[y], symbol="s", size=15, brush=pg.mkBrush(qcolor(t.panel)),
                                      pen=pg.mkPen(colour, width=1.6))
            item.setToolTip(" · ".join(economic.format_event(ev) for ev in events))
            self.price.addItem(item)
            text = pg.TextItem(ECON_LETTERS[e.kind], color=colour, anchor=(0.5, 0.5))
            text.setPos(x[i], y)
            self.price.addItem(text)

    def _draw_events(self, x, h, l, ty, t):
        """E / S / D marks under the candles of the earnings, split and ex-dividend dates inside the loaded range,
        and the economic events above them (#167)."""
        if not self.events_btn.isChecked():
            return
        self._draw_econ(x, h, ty, t)
        if self._info is None or self._info.symbol != self._loaded_symbol or len(x) < 2:
            return
        step = float(np.median(np.diff(x)))
        spots = {}
        for e in self._info.events:
            ts = pd.Timestamp(e.day).tz_localize("UTC").timestamp()
            if x[0] - step <= ts <= x[-1] + step:
                spots.setdefault(nearest_bar(x, ts), []).append(e)
        for i, events in sorted(spots.items()):
            e = min(events, key=lambda e: fundamentals.KINDS.index(e.kind))  # one mark per candle; the page lists all
            colour = qcolor(event_colour(t, e.kind))
            y = ty(l[i] * 0.985) if l[i] > 0 else ty(l[i])
            self.price.addItem(pg.ScatterPlotItem(x=[x[i]], y=[y], symbol="o", size=15, brush=pg.mkBrush(qcolor(t.panel)),
                                                  pen=pg.mkPen(colour, width=1.6)))
            text = pg.TextItem(LETTERS[e.kind], color=colour, anchor=(0.5, 0.5))
            text.setPos(x[i], y)
            self.price.addItem(text)

    def _sidebar_shown(self, key: str):
        self._fetch_info()
        self.news.set_visible(key == "news")
        self.watchlists.set_visible(key == "watchlists")
        self.depth.set_visible(key == "depth")

    def _sidebar_toggled(self):
        if not self.sidebar.is_open:
            self.news.set_visible(False)  # collapsed: no fetching in the background
            self.watchlists.set_visible(False)
            self.depth.set_visible(False)

    def showEvent(self, event):
        super().showEvent(event)
        # the keyboard to the chart, not its symbol field (the first field gets it by default): typing then searches
        QTimer.singleShot(0, lambda: QApplication.focusWidget() in (None, self.symbol) and self.canvas.setFocus())
        if self.sidebar.is_open and self.sidebar.current:
            self._sidebar_shown(self.sidebar.current)

    def closeEvent(self, event):
        self.live.stop()
        super().closeEvent(event)

    def hideEvent(self, event):
        super().hideEvent(event)
        self.news.set_visible(False)  # another page of the app is open
        self.watchlists.set_visible(False)
        self.depth.set_visible(False)

    # -- indicators & algorithms -----------------------------------------------------------------
    def open_studies(self, initial: str = ""):
        """The indicator and signal dialog (#230) over this window."""
        from tradebot.gui.study_dialog import StudyDialog

        window = self.window()
        if self._studies is None:
            self._studies = StudyDialog(self, window, initial)
        else:
            self._studies.reset(window, initial)
        self._studies.show_palette()
        return self._studies

    def prewarm_studies(self):
        """Build and style the indicator dialog while the app is idle, so the first "/" opens it at once."""
        from tradebot.gui.study_dialog import StudyDialog

        if self._studies is None:
            self._studies = StudyDialog(self, self.window())
            self._studies.ensurePolished()
            self._studies.grab()  # lays out and paints it once, off screen

    def add_indicator(self, cls, parent=None):
        """Ask the parameters of indicator ``cls`` and add it; returns the new indicator, or None if cancelled."""
        values = ParamDialog.ask(cls.name, cls.params, parent=parent or self, timeframe=True)
        if values is None:
            return None
        ind = cls(**values)
        self.indicators.append(ind)
        self.redraw()
        return ind

    def edit_indicator(self, ind, parent=None):
        """Ask new parameters for ``ind`` on the chart; it keeps its place and colours. Returns the new indicator."""
        values = ParamDialog.ask(ind.name, type(ind).params, ind.values, parent=parent or self, timeframe=True,
                                 tf=ind.tf)
        if values is None or ind not in self.indicators:
            return None
        new = type(ind)(**values)
        self.indicators[self.indicators.index(ind)] = new
        if id(ind) in self._slots:
            self._slots[id(new)] = self._slots.pop(id(ind))
        self.redraw()
        return new

    def remove_indicator(self, ind):
        if ind in self.indicators:
            self.indicators.remove(ind)
            self.redraw()

    def styled_outputs(self, ind) -> list[str]:
        return self._styled_outputs(ind)

    @staticmethod
    def study_label(study) -> str:
        """An indicator or algorithm as the legend names it: ``rsi 14``, ``bbands 20 2 · 4h``."""
        tf = getattr(study, "tf", None)
        return _spaced_label(study.name, study.values) + (f" · {tf}" if tf else "")

    STYLES_KEY = "chart.styles"  #: settings: {indicator name: {output: {"color", "width", "dash"}}} (#52)

    def _styled_outputs(self, ind) -> list[str]:
        """Outputs of ``ind`` drawn as plain lines, whose look the user may change."""
        return [o for o in ind.outputs if o not in ind.labels and o not in ind.hidden and o != "hist"
                and not o.endswith("_hist")]

    def _style_of(self, ind, output: str) -> Style:
        """The indicator's own style for an output with the user's override on top."""
        base = ind.styles.get(output, Style())
        over = (self.settings.get(self.STYLES_KEY, {}) if self.settings else {}).get(ind.name, {}).get(output)
        return replace(base, **{k: v for k, v in over.items() if k in ("color", "width", "dash")}) if over else base

    def edit_styles(self, ind, parent=None):
        t = self.themes.current
        slot = self._palette_slot(ind)
        rows = []
        for k, out in enumerate(ind.outputs):
            if out not in self._styled_outputs(ind):
                continue
            st = self._style_of(ind, out)
            color = resolve_color(t, st.color) if st.color else t.series[(slot + k) % len(t.series)]
            rows.append((out, qcolor(color).name(), st.width or 1.6, st.dash if st.dash in ("dash", "dot") else None))
        dlg = StyleDialog(f"{ind.label}: line colours", rows, parent or self)
        if not dlg.exec():
            return
        self.set_style_overrides(ind.name, dlg.overrides(), merge=not dlg.reset_requested)

    def set_style_overrides(self, name: str, overrides: dict, merge: bool = True):
        """Remember ``overrides`` ({output: {color, width, dash}}) for indicator ``name`` and redraw."""
        if self.settings is not None:
            data = {k: dict(v) for k, v in self.settings.get(self.STYLES_KEY, {}).items()}
            mine = data.get(name, {}) if merge else {}
            for out, fields in overrides.items():
                mine[out] = {**mine.get(out, {}), **fields}
            if mine:
                data[name] = mine
            else:
                data.pop(name, None)
            self.settings.set(self.STYLES_KEY, data)
        self.redraw()

    def _volume_toggled(self, on: bool):
        """Show the volume pane (the ``volume`` indicator, first below the candles) or remove it."""
        has = [i for i in self.indicators if i.name == "volume"]
        if on and not has:
            self.indicators.insert(0, INDICATORS.get("volume")())
        elif not on and has:
            self.indicators = [i for i in self.indicators if i.name != "volume"]
        self.redraw()

    @property
    def volume_overlay(self) -> bool:
        """Whether semi-transparent volume bars run along the bottom of the price pane (setting, #116)."""
        return bool(self.settings.get("chart.volume_overlay", False)) if self.settings else getattr(self, "_overlay", False)

    def _toggle_volume_overlay(self):
        if self.settings:
            self.settings.set("chart.volume_overlay", not self.volume_overlay)
        else:
            self._overlay = not self.volume_overlay
        self.redraw()

    def _clear_volume_overlay(self):
        vb = getattr(self, "_overlay_vb", None)
        if vb is not None:
            self.price.scene().removeItem(vb)
            self._overlay_vb = None

    def _draw_volume_overlay(self, x, volume, o, c, width, t):
        """Volume bars in the bottom quarter of the price pane, on their own hidden y-axis."""
        vb = pg.ViewBox(enableMenu=False)
        vb.setMouseEnabled(False, False)
        vb.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.price.scene().addItem(vb)
        vb.setXLink(self.price.vb)
        sync = lambda *_: vb.setGeometry(self.price.vb.sceneBoundingRect())  # noqa: E731
        self.price.vb.sigResized.connect(sync)
        sync()
        ok = np.isfinite(volume)
        up, down = qcolor(t.up), qcolor(t.down)
        up.setAlpha(90)
        down.setAlpha(90)
        bars = VolumeBarsItem()
        bars.set_data(x[ok], volume[ok], o[ok], c[ok], width, up, down)
        vb.addItem(bars)
        vb.setYRange(0, max(float(np.nanmax(volume)), 1e-12) * 4, padding=0)
        vb.setLimits(yMin=0)
        self._overlay_vb = vb

    def choose_signal(self, name: str, parent=None) -> bool:
        """Ask the parameters of algorithm ``name`` and show its signals; False if cancelled (the current one stays)."""
        if name not in ALGORITHMS:
            return False
        algo = build_algorithm(name)
        values = ParamDialog.ask(f"{name} parameters", algo.param_space(), algo.values, parent or self)
        if values is None:
            return False
        self.algorithm = algo.with_params(**values)
        self.redraw()
        return True

    def remove_signal(self):
        self.algorithm = None
        self.redraw()

    def set_algorithm(self, algorithm):
        """Show an externally configured algorithm (e.g. from the backtest lab)."""
        self.algorithm = algorithm
        self.redraw()

    # -- rendering -------------------------------------------------------------------------------
    def _style(self, t: Theme):
        bg = None if t.glass else t.panel
        self.canvas.setBackground(qcolor(bg) if bg else None)
        self.canvas.setStyleSheet("background: transparent;" if t.glass else "")
        for plot in [self.price, *self.panes]:
            for name in ("left", "right", "bottom"):
                ax = plot.getAxis(name)
                ax.setPen(pg.mkPen(qcolor(t.panel_border)))
                ax.setTextPen(pg.mkPen(qcolor(t.muted)))
        cross = pg.mkPen(qcolor(t.muted), width=1, style=Qt.PenStyle.DashLine)
        for vline, hline in self._cross.values():
            vline.setPen(cross)
            hline.setPen(cross)
            hline.label.setColor(qcolor(t.text))
            hline.label.fill = pg.mkBrush(qcolor(t.panel_border))

    def _add_crosshair(self, plot):
        vline = pg.InfiniteLine(angle=90, movable=False)
        # the value under the cursor, at the right edge next to the axis
        hline = pg.InfiniteLine(angle=0, movable=False, label="{value:.6g}", labelOpts={"position": 0.98})
        hline.setVisible(False)  # until the cursor is over this pane
        for line in (vline, hline):
            plot.addItem(line, ignoreBounds=True)  # never part of auto-fit or view ranges
        self._cross[plot] = (vline, hline)
        return vline, hline

    # -- values under the cursor (#45) -----------------------------------------------------------
    def _add_legend(self, plot):
        """Text block in the plot's top-left corner, a child of its view box: never part of auto-fit or ranges."""
        label = pg.LabelItem(justify="left")
        label.setParentItem(plot.vb)
        label.anchor(itemPos=(0, 0), parentPos=(0, 0), offset=(6, 2))
        label.setAcceptedMouseButtons(Qt.MouseButton.NoButton)  # clicks and drags go to the chart below
        self._legend[plot] = label

    def _show_values(self):
        """Fill every pane's corner with its values at the hovered bar, or the latest one."""
        if self.df is None or self.df.empty:
            return
        t = self.themes.current
        i = self._hover if self._hover is not None and self._hover < len(self.df) else len(self.df) - 1
        muted = lambda text: f"<span style='color:{t.muted}'>{text}</span>"  # noqa: E731

        def colored(v, color, fmt=fmt_value):
            return f"<span style='color:{color}'>{fmt(v)}</span>"

        def line(label, values):
            return " ".join([muted(escape(label)), *(colored(series[i], *rest) for series, *rest in values)])

        row = self.df.iloc[i]
        direction = t.up if row.close >= row.open else t.down
        ohlc = " ".join(f"{muted(k)}{colored(row[c], direction)}" for k, c in
                        (("O", "open"), ("H", "high"), ("L", "low"), ("C", "close")))
        title = self._title + (f" · {self._type}" if self._type in ("Heikin-Ashi", "Renko", "Compare") else "")
        head = [f"<span style='color:{t.text}'>{escape(title)}</span>" if self._title else "", ohlc]
        if i > 0:
            prev = self.df["close"].iloc[i - 1]
            diff = row.close - prev
            sign_color = t.up if diff >= 0 else t.down
            head.append(f"<span style='color:{sign_color}'>{diff:+,.{_decimals(diff)}f} ({diff / prev * 100:+.2f}%)</span>")
        lines = {self.price: ["&nbsp;&nbsp;".join(p for p in head if p)]}
        for plot, rows in self._rows.items():
            lines.setdefault(plot, []).extend(line(label, values) for label, values in rows)
        if self._positions is not None:
            lines[self.price].append(self._signal_line(i, t, muted))
        for plot, label in self._legend.items():
            label.setText("<br>".join(lines.get(plot, [])), size="9pt")

    def _signal_line(self, i, t, muted) -> str:
        algo, pos = self.algorithm, self._positions
        state = {1: ("long", t.up), -1: ("short", t.down)}.get(int(np.sign(pos[i])), ("flat", t.muted))
        prev = pos[i - 1] if i > 0 else 0.0
        if pos[i] > prev:
            state = ("buy", t.up)
        elif pos[i] < prev:
            state = ("sell", t.down)
        return f"{muted(escape(_spaced_label(algo.name, algo.values)))} <span style='color:{state[1]}'>{state[0]}</span>"

    def redraw(self, autorange: bool = False):
        t = self.themes.current
        self._ensure_market()
        keep = {id(item) for item in (*self.drawings, *(line for pair in self._cross.values() for line in pair))}
        for plot in [self.price, *self.panes]:
            for item in list(plot.items):
                if id(item) not in keep:
                    plot.removeItem(item)
        self._clear_volume_overlay()
        spare, self.panes = self.panes, []  # indicator panes are reused in order: building one takes ~30 ms (#68)
        self._fit_series = {}
        self._fit_index = {}  # plot -> RangeIndex of its _fit_series, made on the first fit after a redraw
        self._rows = {self.price: []}
        self._positions = None
        self._style(t)
        self.vol_btn.blockSignals(True)
        self.vol_btn.setChecked(any(i.name == "volume" for i in self.indicators))
        self.vol_btn.blockSignals(False)
        if self.df is None or self.df.empty:
            self._drop_panes(spare)
            self._legend[self.price].setText("")
            return
        df = self.df
        x = self._x = epoch_seconds(df.index)
        width = (np.median(np.diff(x)) if len(x) > 1 else 86400) * 0.7
        o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
        positive = bool((l > 0).all())
        self._set_log(self.log_btn.isChecked() and positive and self._type != "Compare")  # percent change can be < 0
        if self.log_btn.isChecked() and not positive and self._type != "Compare":
            self.status.emit("log scale needs prices above 0: showing a linear scale")
        ty = self._ty
        kind = self._type
        compare = kind == "Compare"
        if compare:  # percent change lines instead of candles; overlays on the real price would not fit the scale
            self._draw_compare(df, x, t)
        else:
            # what is drawn: Heikin-Ashi/Renko replace the candles; indicators and signals below keep the real ones
            shown = heikin_ashi(df) if kind == "Heikin-Ashi" else df
            if kind == "Renko":
                try:
                    shown = renko(df)
                except ValueError:
                    shown = df.iloc[0:0]
            sx = epoch_seconds(shown.index) if len(shown) else np.empty(0)
            sw = (np.median(np.diff(sx)) if len(sx) > 1 else 86400) * 0.7
            so, sh, sl, sc = (shown[k].to_numpy() for k in ("open", "high", "low", "close"))
            if self.style_btn.isChecked() and kind != "Renko":
                self.price.addItem(Curve(sx, ty(sc), pen=pg.mkPen(qcolor(t.accent), width=2)))
                self._fit_series[self.price] = [ty(sc)]
            else:
                # bricks are not one per candle: the fit covers the real range (they follow the close)
                self._fit_series[self.price] = [ty(l), ty(h)] if kind == "Renko" else [ty(sl), ty(sh)]
                candles = CandlestickItem()
                candles.set_data(sx, ty(so), ty(sh), ty(sl), ty(sc), sw, qcolor(t.up), qcolor(t.down))
                self.price.addItem(candles)

        if self.volume_overlay and np.isfinite(volume_of(df)).any():
            self._draw_volume_overlay(x, volume_of(df), o, c, width, t)

        self._label_last = {True: -10**9, False: -10**9}  # pattern labels of all indicators share their spacing
        frames: dict = {}  # this redraw's indicator frames, kept for the next one
        for ind in self.indicators:
            if compare and ind.overlay:
                continue
            note = ""
            try:
                frame = self._frame_of(ind, df, frames)
            except ValueError as e:  # e.g. a higher timeframe (#161) shorter than these candles: the candles' own
                if not ind.tf:
                    raise
                frame, note = type(ind)(**ind.values)(df), f" · {e}"
            target = self.price
            if not ind.overlay:
                target = spare.pop(0) if spare else self._new_pane()
                for level in ind.levels:
                    ls = ind.level_styles.get(level, Style())
                    pen = pg.mkPen(qcolor(resolve_color(t, ls.color) if ls.color else t.muted), width=ls.width or 1,
                                   style=DASHES.get(ls.dash, Qt.PenStyle.DotLine))
                    target.addItem(pg.InfiniteLine(level, angle=0, pen=pen))
                self.panes.append(target)
            values = []  # this indicator's drawn outputs for the value line
            label = _spaced_label(ind.name, ind.values) + (f" · {ind.tf}" if ind.tf else "") + note
            if ind.uses_volume and not np.isfinite(volume_of(df)).any():
                label += f" · no volume data for {self._loaded_symbol or 'this symbol'}"
            if gone := [c for c in ind.market_data if c not in df.columns]:  # crypto market data (#166)
                why = self._market_errors.get((self._loaded_symbol, self.tf.currentText(), gone[0]))
                label += f" · no data: {why[:70]}" if why else " · loading market data …"
            self._rows.setdefault(target, []).append((label, values))
            slot = self._palette_slot(ind)
            curves = {}  # output -> (x, y) as drawn, for the fills
            for k, col in enumerate(frame.columns):
                if ind.output_of(col) in ind.zones:
                    self._draw_zone(target, ind, frame, col, x, width, t, values)
                    continue
                if not ind.is_line(col):
                    self._draw_labels(target, ind, frame, col, x, c, t)
                    continue
                raw = y = frame[col].to_numpy()
                if target is self.price:
                    y = ty(y)
                ok = np.isfinite(y)
                style = self._style_of(ind, ind.output_of(col))
                hist = col.endswith("_hist")
                default = t.muted if hist else t.series[(slot + k) % len(t.series)]
                color = qcolor(resolve_color(t, style.color) if style.color else default)
                amount = ind.units.get(ind.output_of(col)) in ("volume", "money")
                if style.bars:  # amounts like volume: up/down by candle (or the style's colour), merged bars add up
                    up, down = (color, color) if style.color else (qcolor(t.up), qcolor(t.down))
                    bars = VolumeBarsItem()
                    bars.set_data(x[ok], y[ok], o[ok], c[ok], width, up, down)
                    target.addItem(bars)
                    self._fit_series.setdefault(target, []).append(np.zeros_like(y))  # bars grow from 0
                elif hist:
                    bars = HistogramItem()
                    bars.set_data(x[ok], y[ok], width, color, pg.mkPen(pg.getConfigOption("foreground")))
                    target.addItem(bars)
                    self._fit_series.setdefault(target, []).append(np.zeros_like(y))  # bars grow from 0
                else:
                    pen = pg.mkPen(color, width=style.width or 1.6, style=DASHES.get(style.dash, Qt.PenStyle.SolidLine))
                    target.addItem(Curve(x[ok], y[ok], pen=pen, name=col))
                    curves[ind.output_of(col)] = y
                color = color.name()  # "#rrggbb" for the value line's rich text
                self._fit_series.setdefault(target, []).append(y)
                values.append((raw, color, fmt_amount if amount else fmt_value))  # prices also on the log scale
            for a, b, fill in ind.fills:  # behind the lines; not in _fit_series, so they never change the auto-fit
                if a in curves and b in curves:
                    item = Band(x, curves[a], curves[b], qcolor(resolve_color(t, fill.color or "muted")))
                    item.setZValue(-5)
                    target.addItem(item)
            self._draw_shades(ind, frame, x, width, t)
        self._drop_panes(spare)
        self._frames = frames
        if self.algorithm is not None and not compare:
            self._positions = self.algorithm.run(df).to_numpy()
            change = np.diff(np.r_[0.0, self._positions])
            for mask, symbol, color, anchor in ((change > 0, "t1", t.up, l * 0.995), (change < 0, "t", t.down, h * 1.005)):
                if mask.any():
                    self.price.addItem(pg.ScatterPlotItem(x=x[mask], y=ty(anchor[mask]), symbol=symbol, size=13,
                                                          brush=pg.mkBrush(color), pen=pg.mkPen(None)))
        if not compare:
            self._draw_events(x, h, l, ty, t)
        self._style(t)
        self._show_values()
        self._remember_session()
        self._lock_view()
        if autorange:
            self.reset_view()
        else:
            self.fit_y()

    def _frame_of(self, ind, df, frames: dict):
        """``ind(df)``, reused from the last redraw while the candles are the same object (#228): a theme switch,
        the line/candles toggle, the log scale or another indicator then don't recompute every indicator. New or
        merged candles (a load, a live tick, market data) are a new frame, so those recompute."""
        key = (type(ind), ind.tf, repr(sorted(ind.values.items())))
        cached = self._frames.get(key)
        frame = cached[1] if cached is not None and cached[0] is df else ind(df)
        frames[key] = (df, frame)
        return frame

    def _draw_compare(self, df, x, t):
        """The charted symbol and ``self.compare`` as percent change from the first time they all have (#163)."""
        name = self._loaded_symbol or "symbol"
        frame = percent_change({name: df["close"], **{k: v["close"] for k, v in self.compare.items()}})
        frame = frame.reindex(df.index)  # the bars of the chart, so the crosshair and values line up
        for k, col in enumerate(frame.columns):
            y = frame[col].to_numpy()
            ok = np.isfinite(y)
            color = qcolor(t.series[k % len(t.series)])
            self.price.addItem(Curve(x[ok], y[ok], pen=pg.mkPen(color, width=2), name=col))
            self._fit_series.setdefault(self.price, []).append(y)
            fmt = lambda v: "–" if not np.isfinite(v) else f"{v:+.2f}%"  # noqa: E731
            self._rows[self.price].append((col, [(y, color.name(), fmt)]))
        if not self.compare:
            self.status.emit("Compare: type the symbols to compare with in the 'vs' box and press Enter")

    def _new_pane(self):
        """An empty indicator pane below the last one, x-linked to the price pane."""
        pane = self.canvas.addPlot(row=len(self.panes) + 1, col=0, axisItems={"bottom": DateAxis()})
        pane.setXLink(self.price)
        pane.setMaximumHeight(170)
        pane.showGrid(x=True, y=True, alpha=0.15)
        pane.showAxis("right")
        pane.hideAxis("left")
        pane.getAxis("right").setWidth(70)
        pane.getAxis("right").setStyle(maxTickLevel=0)
        pane.setMenuEnabled(False)
        # no sigRangeChanged hook: the price pane is x-linked to every pane and fits them all when the time moves
        self._add_crosshair(pane)
        self._add_legend(pane)
        return pane

    def _drop_panes(self, panes):
        for pane in panes:
            self.canvas.removeItem(pane)
            self._cross.pop(pane, None)
            self._legend.pop(pane, None)

    def _palette_slot(self, ind) -> int:
        """Where an indicator's outputs start in the theme palette, fixed when it's first drawn: removing or
        adding other indicators doesn't recolour it."""
        if id(ind) not in self._slots:
            self._slots[id(ind)] = self._next_slot
            self._next_slot += len(ind.plot_outputs())
        return self._slots[id(ind)]

    def _draw_shades(self, ind, frame, x, width, t):
        """Background shades behind the candles (``ind.shades``, e.g. the market regime): one band per run of bars."""
        for output, colors in ind.shades.items():
            col = f"{ind.name}_{output}"
            if col not in frame:
                continue
            codes = frame[col].to_numpy()
            half = width / 0.7 / 2  # half a candle slot
            for code, css in colors.items():
                on = np.r_[False, codes == code, False]
                edges = np.flatnonzero(on[1:] != on[:-1])
                brush = pg.mkBrush(qcolor(resolve_color(t, css)))
                for a, b in zip(edges[::2], edges[1::2]):
                    band = pg.LinearRegionItem((x[a] - half, x[b - 1] + half), movable=False, brush=brush,
                                               pen=pg.mkPen(None))
                    band.setZValue(-10)
                    self.price.addItem(band, ignoreBounds=True)

    def _draw_zone(self, target, ind, frame, col, x, width, t, values):
        """An output drawn as horizontal bands, one per run of a level, plus its letter at the latest one."""
        colour, letter = ind.zones[ind.output_of(col)][:2]
        raw = frame[col].to_numpy(dtype=float)
        slot = width / 0.7  # one candle's space on the x-axis
        half = raw * float(ind.values.get("zone", 0.006)) / 2
        ty = self._ty if target is self.price else (lambda v: v)
        segs = zone_segments(x, raw, half, slot)
        # a level that held for a couple of candles is noise, not a zone; the current one is always kept
        segs = [g for k, g in enumerate(segs) if k == len(segs) - 1 or g[1] - g[0] >= 4 * slot]
        if len(ind.zones[ind.output_of(col)]) > 2:  # e.g. the point of control: only where it is now
            segs = segs[-1:]
        # the log scale bends prices, so the bands are placed by their transformed edges
        segs = [(x0, x1, float(ty(lo)), float(ty(hi)), float(ty(level))) for x0, x1, lo, hi, level in segs]
        color = qcolor(resolve_color(t, colour))
        item = ZonesItem(segs, color, letter, qcolor(t.text))
        target.addItem(item, ignoreBounds=True)
        if segs:  # the letter pairs the colour with a name; it sits at the right end of the latest band
            last = segs[-1]
            tag = pg.TextItem(letter, color=color, anchor=(0, 0.5))
            font = tag.textItem.font()
            font.setBold(True)
            tag.setFont(font)
            tag.setPos(last[1], last[4])
            tag.setZValue(5)
            target.addItem(tag, ignoreBounds=True)
        y = ty(raw)
        self._fit_series.setdefault(target, []).append(y)
        values.append((raw, color.name(), fmt_value))

    #: labels of one side closer than this many candles are thinned out (``label_signs`` indicators)
    LABEL_GAP = 8

    def _draw_labels(self, target, ind, frame, col, x, close, t):
        """Text labels of an indicator output (e.g. Elliott wave numbers) at its anchor values."""
        output = ind.output_of(col)
        texts = ind.labels.get(output)
        if not texts:
            return  # hidden output
        helps = ind.label_help.get(output, {})
        anchor_col = f"{ind.name}_{ind.label_anchor}" if ind.label_anchor else None
        anchor = frame[anchor_col].to_numpy() if anchor_col in frame else close
        codes = frame[col].to_numpy()
        last_on = self._label_last
        for i in np.flatnonzero(~np.isnan(codes)):
            # label above swing highs (local maximum of the anchor line), below swing lows
            neighbours = anchor[max(i - 1, 0): i + 2]
            is_high = anchor[i] >= np.nanmax(neighbours)
            code = float(codes[i])
            colour = t.text
            chip = {}
            if ind.label_signs:
                # bullish patterns sit under the candle, bearish ones over it, whatever the neighbours say
                is_high = code < 0 or (code >= 10 and is_high)
                if i - last_on[is_high] < self.LABEL_GAP:
                    continue  # too close to the previous label on this side: the alert still fires
                last_on[is_high] = i
                colour = t.down if code < 0 else t.up if code < 10 else t.muted
                # a chip in the panel colour keeps the text readable on top of same-coloured candles
                fill = qcolor(t.panel)
                fill.setAlpha(235)
                chip = {"fill": pg.mkBrush(fill), "border": pg.mkPen(qcolor(colour))}
            item = pg.TextItem(texts.get(code, "?"), color=colour,
                               anchor=(0.5, 1.45 if is_high else -0.45) if ind.label_signs
                               else (0.5, 1.15 if is_high else -0.15), **chip)
            font = item.textItem.font()
            font.setBold(True)
            item.setFont(font)
            if code in helps:
                item.setToolTip(helps[code])
            item.setPos(x[i], self._ty(anchor[i]) if target is self.price else anchor[i])
            item.setZValue(10)
            target.addItem(item)

    # -- crosshair & drawing tools ---------------------------------------------------------------
    def _mouse_moved(self, pos):
        """Crosshair through every pane at the hovered bar, from whichever pane the cursor is over (#46)."""
        plot = self._plot_at(pos)
        if self.df is None or self.df.empty or plot is None:
            return
        pt = plot.vb.mapSceneToView(pos)
        if plot is self.price:
            self._last_view_x = pt.x()
            if self._active_tool() is not None:
                pt, which = self.snap_view_point(pt)
                self._show_snap_marker(pt) if which else self._hide_snap_marker()
            else:
                self._hide_snap_marker()
        else:
            self._hide_snap_marker()
        self._update_preview(plot, pt)
        i = nearest_bar(self._x, pt.x())
        for p, (vline, hline) in self._cross.items():
            vline.setPos(self._x[i])  # panes are x-linked: the same time everywhere, snapped to the bar
            hline.setVisible(p is plot)
            if p is plot:
                hline.setPos(pt.y())
        if i != self._hover:
            self._hover = i
            self._show_values()
        row = self.df.iloc[i]
        chg = row.close / row.open - 1
        self.status.emit(f"{self.df.index[i]:%Y-%m-%d %H:%M}  O {row.open:.6g}  H {row.high:.6g}  "
                         f"L {row.low:.6g}  C {row.close:.6g}  ({chg * 100:+.2f}%)   "
                         f"cursor {self.price_at(pt.y()) if plot is self.price else pt.y():.6g}")

    # -- magnet (#197) ---------------------------------------------------------------------------
    MAGNET_KEY = "chart.magnet"

    def _magnet_toggled(self, on):
        if self.settings is not None:
            self.settings.set(self.MAGNET_KEY, on)
        if not on:
            self._hide_snap_marker()

    def magnet_active(self) -> bool:
        """The magnet button, flipped while Ctrl is held."""
        flip = bool(QApplication.keyboardModifiers() & Qt.KeyboardModifier.ControlModifier)
        return self.magnet_btn.isChecked() != flip

    def _snap_view_ohlc(self):
        # the cache holds the frame itself, so its identity can't be reused by a later frame
        key = (self._log, len(self._x))
        if self._snap_cache is None or self._snap_cache[0] != key or self._snap_cache[2] is not self.df:
            cols = tuple(np.asarray(self._ty(self.df[c].to_numpy(dtype=float))) for c in ("open", "high", "low", "close"))
            self._snap_cache = (key, cols, self.df)
        return self._snap_cache[1]

    def snap_view_point(self, pt):
        """``pt`` (view coordinates) snapped to the nearest candle O/H/L/C within the radius: (QPointF, which|None)."""
        if self.df is None or self.df.empty or not self.magnet_active():
            return pt, None
        dx, dy = self.price.vb.viewPixelSize()
        if not dx or not dy:
            return pt, None
        radius = SNAP_RADIUS_TOUCH if self.themes.touch else SNAP_RADIUS
        hit = snap_point(pt.x(), pt.y(), self._x, self._snap_view_ohlc(), 1 / dx, 1 / dy, radius)
        return (pt, None) if hit is None else (QPointF(hit[0], hit[1]), hit[2])

    def _show_snap_marker(self, pt):
        if self._snap_marker is None:
            self._snap_marker = pg.ScatterPlotItem(size=9, pxMode=True)
            self._snap_marker.setZValue(100)
            self._snap_marker.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            self.price.addItem(self._snap_marker, ignoreBounds=True)
        t = self.themes.current
        self._snap_marker.setData([pt.x()], [pt.y()], brush=pg.mkBrush(qcolor(t.accent)), pen=pg.mkPen(qcolor(t.text), width=1.5))
        self._snap_marker.setVisible(True)

    def _hide_snap_marker(self):
        if self._snap_marker is not None:
            self._snap_marker.setVisible(False)

    def _snap_handle_move(self, roi, start):
        """While a handle of ``roi`` is dragged: pull the moved one onto a candle value."""
        if self._snapping or not self.magnet_active():
            return
        handles = roi.getHandles()
        now = [roi.mapToParent(h.pos()) for h in handles]
        if len(start) != len(now):
            return
        moved = max(range(len(now)), key=lambda i: abs(now[i].x() - start[i].x()) + abs(now[i].y() - start[i].y()))
        if now[moved] == start[moved]:
            return
        pt, which = self.snap_view_point(now[moved])
        if which is not None:
            self._snapping = True
            try:
                roi.movePoint(handles[moved], pt, finish=False)
            finally:
                self._snapping = False

    def _hook_snapping(self, item):
        if isinstance(item, NoteItem):
            item.snap = lambda x, y: (lambda p: (p[0].x(), p[0].y()))(self.snap_view_point(QPointF(x, y)))
        elif isinstance(item, pg.InfiniteLine):
            item.sigPositionChanged.connect(lambda line: self._snap_line(line))
        else:
            start: list = []
            item.sigRegionChangeStarted.connect(
                lambda *_: start.__setitem__(slice(None), [item.mapToParent(h.pos()) for h in item.getHandles()]))
            item.sigRegionChanged.connect(lambda *_: self._snap_handle_move(item, start))

    def _snap_line(self, line):
        """A level (price, at the cursor's candle) or a time line (candle time) is being dragged."""
        if self._snapping or not self.magnet_active() or self.df is None or self.df.empty:
            return
        dx, dy = self.price.vb.viewPixelSize()
        radius = SNAP_RADIUS_TOUCH if self.themes.touch else SNAP_RADIUS
        if line.angle == 90:
            v = snap_time(line.value(), self._x, 1 / dx if dx else 0, radius)
        elif self._last_view_x is not None:
            hit = snap_point(self._last_view_x, line.value(), self._x, self._snap_view_ohlc(),
                             1 / dx if dx else 0, 1 / dy if dy else 0, radius)
            v = hit[1] if hit else None
        else:
            v = None
        if v is not None and v != line.value():
            self._snapping = True
            try:
                line.setValue(v)
            finally:
                self._snapping = False

    def _tool_toggled(self, key, on):
        if on:
            for k, btn in self.tools.items():
                if k != key:
                    btn.setChecked(False)
        self._pending_point = None
        self._clear_preview()
        self._hide_snap_marker()

    def _clear_preview(self):
        if self._preview is not None:
            self.price.removeItem(self._preview)
            self._preview = None

    def _make_shape(self, tool, p1, pt):
        """The drawing for ``tool`` between two view points (also used, inert, as the preview before the 2nd click)."""
        pen = pg.mkPen(qcolor(self.themes.current.accent), width=2)
        if tool == "trend":
            return pg.LineSegmentROI([[p1.x(), p1.y()], [pt.x(), pt.y()]], pen=pen)
        if tool in ("fib", "measure"):
            return self._two_point(tool, [p1.x(), p1.y()], [pt.x(), pt.y()], self._accent_hex())
        return pg.RectROI([min(p1.x(), pt.x()), min(p1.y(), pt.y())], [abs(pt.x() - p1.x()), abs(pt.y() - p1.y())],
                          pen=pen)

    def _update_preview(self, plot, pt):
        """After the first click: show what the second one would create, following the cursor."""
        tool = self._active_tool()
        if self._pending_point is None or tool is None or plot is not self.price:
            return
        self._clear_preview()
        item = self._make_shape(tool, self._pending_point, pt)
        item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        for h in item.getHandles():
            h.hide()
        item.setOpacity(0.6)
        self.price.addItem(item, ignoreBounds=True)
        self._preview = item

    def _active_tool(self):
        return next((k for k, b in self.tools.items() if b.isChecked()), None)

    def _mouse_clicked(self, ev):
        if ev.button() == Qt.MouseButton.RightButton:
            self._context_menu(ev.scenePos(), ev.screenPos().toPoint())
        elif ev.button() == Qt.MouseButton.LeftButton:
            self._click(ev.scenePos())

    def _tap(self, scene_pos):
        """Touch tap: move the crosshair / OHLC readout there, then act like a click."""
        self._mouse_moved(scene_pos)
        self._click(scene_pos)

    def _plot_at(self, scene_pos):
        return next((p for p in [self.price, *self.panes] if p.sceneBoundingRect().contains(scene_pos)), None)

    def _click(self, scene_pos):
        tool = self._active_tool()
        if tool is None or not self.price.sceneBoundingRect().contains(scene_pos):
            return
        pt = self.price.vb.mapSceneToView(scene_pos)
        raw_x = pt.x()
        pt, _ = self.snap_view_point(pt)
        if tool == "hline":
            self._add_level(self.price_at(pt.y()))
            return
        if tool == "vline":
            dx = self.price.vb.viewPixelSize()[0]
            t = snap_time(raw_x, self._x, 1 / dx if dx else 0,
                          SNAP_RADIUS_TOUCH if self.themes.touch else SNAP_RADIUS, self.magnet_active())
            self._add_vline(raw_x if t is None else t)
            return
        if tool == "text":
            text = self._ask_text("")
            if text:
                self._add_note(pt.x(), self.price_at(pt.y()), text)
            else:
                self.tools["text"].setChecked(False)
            return
        if self._pending_point is None:
            self._pending_point = pt
            self.status.emit("tap/click the second point")
            return
        p1, self._pending_point = self._pending_point, None
        self._clear_preview()
        self._add_drawing(self._make_shape(tool, p1, pt), tool)

    def _accent_hex(self) -> str:
        return qcolor(self.themes.current.accent).name()

    def _add_level(self, price: float, color: str | None = None, save: bool = True):
        color = color or self._accent_hex()
        item = pg.InfiniteLine(self._ty(price), angle=0, movable=True, pen=pg.mkPen(qcolor(color), width=2),
                               label="{value:.6g}", labelOpts={"position": 0.97, "color": qcolor(color)})
        item.label.format = _PriceFormat(self)
        item.label.valueChanged()
        self._add_drawing(item, "hline", save)

    def _add_vline(self, x: float, color: str | None = None, save: bool = True):
        item = pg.InfiniteLine(x, angle=90, movable=True, pen=pg.mkPen(qcolor(color or self._accent_hex()), width=2))
        self._add_drawing(item, "vline", save)

    def _add_note(self, x: float, price: float, text: str, color: str | None = None, save: bool = True):
        item = NoteItem(x, self._ty(price), text, qcolor(color or self._accent_hex()))
        self._add_drawing(item, "text", save)

    def _two_point(self, kind, p1, p2, color):
        cls = FibItem if kind == "fib" else MeasureItem
        return cls(p1, p2, self._ty, self.price_at, qcolor(color))

    def _ask_text(self, current: str) -> str:
        text, ok = QInputDialog.getText(self, "Note", "Text:", text=current)
        return text.strip() if ok else ""

    def reset_view(self):
        self._adjusting = True
        try:
            for plot in [self.price, *self.panes]:
                plot.autoRange()  # autoRange() disables auto range in pyqtgraph, so re-enable afterwards
                plot.enableAutoRange()
        finally:
            self._adjusting = False
        self.fit_y()

    # -- y-axis auto-fit -------------------------------------------------------------------------
    @property
    def autofit(self) -> bool:
        return self.fit_btn.isChecked()

    def fit_y(self, force: bool = False):
        """Fit each plot's y-axis to its data inside the visible time range (while auto-fit is on, or ``force``)."""
        if not (self.autofit or force):
            return
        x0, x1 = self.price.vb.viewRange()[0]
        self._adjusting = True
        try:
            for plot in [self.price, *self.panes]:
                plot.enableAutoRange(y=False)  # pyqtgraph would otherwise fit y to all data
                index = self._fit_index.get(plot)
                if index is None:  # built once per redraw, then every scroll step only looks it up
                    index = self._fit_index[plot] = RangeIndex(self._x, self._fit_series.get(plot, []))
                bounds = index.y_range(x0, x1)
                if bounds is not None:
                    plot.setYRange(*bounds, padding=0)
        finally:
            self._adjusting = False

    def _range_changed(self, _vb, _ranges, changed):
        if changed[0] and self.autofit and not self._adjusting:
            self.fit_y()  # scrolled or zoomed in time

    def _lock_view(self):
        """While auto-fit is on, mouse and touch move only the time axis, and not past the data."""
        on = self.autofit and len(self._x) > 0
        margin = float(np.median(np.diff(self._x))) if len(self._x) > 1 else 86400.0  # keep the edge candles whole
        limits = dict(xMin=self._x[0] - margin, xMax=self._x[-1] + margin) if on else dict(xMin=None, xMax=None)
        for plot in [self.price, *self.panes]:
            plot.vb.setMouseEnabled(x=True, y=not on)
        self.price.vb.setLimits(**limits)
        if not on:
            for pane in self.panes:
                pane.vb.setLimits(**limits)
            return
        # A pane redraw() just created still has pyqtgraph's default 0..1 range: limits would clamp that to a
        # 1-second window at the first candle and the x-link would push it to every pane (#36). So give each pane
        # the price range first, with the price pane not following back: before layout, the link's pixel mapping
        # would shrink the price range a little on every redraw.
        x_range = self.price.vb.viewRange()[0]
        self.price.vb.blockLink(True)
        try:
            for pane in self.panes:
                pane.vb.setXRange(*x_range, padding=0)
                pane.vb.setLimits(**limits)
        finally:
            self.price.vb.blockLink(False)

    def _autofit_toggled(self, on):
        if self.settings:
            self.settings.set("chart.autofit", on)
        self._lock_view()
        if on:
            self.fit_y()
        else:
            for plot in [self.price, *self.panes]:
                plot.enableAutoRange(y=False)  # keep the current y-range until the user changes it

    # -- logarithmic price scale (#60) -----------------------------------------------------------
    def _ty(self, price):
        """Price -> the price pane's view y: log10(price) on the log scale."""
        if not self._log:
            return price
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log10(price)

    def price_at(self, y: float) -> float:
        """The price pane's view y -> price."""
        return 10.0 ** y if self._log else y

    def _set_log(self, on: bool):
        """Switch the price pane's view y between price and log10(price), moving the drawings along."""
        if on == self._log:
            return
        self._log = self.price_axis.log = on  # first: the level labels read it while their lines move

        def to_view(y):
            return np.log10(max(y, 1e-12)) if on else 10.0 ** y

        for item in self.drawings:
            if isinstance(item, NoteItem):
                item.setPos(item.pos().x(), to_view(item.pos().y()))
            elif isinstance(item, pg.InfiniteLine):
                if item.angle % 180 == 0:  # a vertical line is a time, not a price
                    item.setValue(to_view(item.value()))
            elif isinstance(item, pg.RectROI):
                (x0, y0), (w, h) = item.pos(), item.size()
                y0, y1 = to_view(y0), to_view(y0 + h)
                item.setPos([x0, y0], finish=False)
                item.setSize([w, y1 - y0], finish=False)
            else:  # trendline: move both ends
                for handle in item.getHandles():
                    p = item.mapToParent(handle.pos())
                    item.movePoint(handle, QPointF(p.x(), to_view(p.y())), finish=False)
        self.price_axis.picture = None  # re-generate the ticks
        self.price_axis.update()

    def _log_toggled(self, on):
        if self.settings:
            self.settings.set("chart.log", on)
        self.redraw()
        self.fit_y(force=True)  # the old y-range means nothing on the other scale

    def eventFilter(self, obj, event):
        if obj is self.canvas and event.type() == QEvent.Type.KeyPress and (
                forward_key(self.symbols.dialog, event) or forward_key(self._studies, event)):
            return True  # typed while a picker opens: it gets the keys (#230)
        if obj is self.canvas and event.type() == QEvent.Type.KeyPress and self._type_to_search(event):
            return True  # typing on the chart opens the symbol dialog (#220), "/" the indicator dialog (#230)
        if obj is self.canvas and event.type() == QEvent.Type.Resize:
            bottom = self.canvas.height() - self.fit_btn.height() - 8
            self.fit_btn.move(8, bottom)
            self.log_btn.move(8 + self.fit_btn.width() + 4, bottom)
            self.vol_btn.move(8 + self.fit_btn.width() + self.log_btn.width() + 8, bottom)
            self.state.setGeometry(self.canvas.rect())
        elif obj is self.canvas and event.type() == QEvent.Type.Leave and self._hover is not None:
            self._hover = None  # cursor left the chart: show the latest bar again
            self._hide_snap_marker()
            self._show_values()
        return super().eventFilter(obj, event)

    def _type_to_search(self, event) -> bool:
        """A letter, digit, ``^`` or ``.`` typed with the cursor over the chart starts a symbol search with it, and
        ``/`` opens the indicator and signal dialog (#230), unless a text field has the keyboard or a dialog or menu
        is open. Keys with Ctrl/Alt/Meta keep their meaning; AltGr (Ctrl+Alt on Windows) still types ``/`` on layouts
        that need it."""
        text = event.text()
        mods = event.modifiers() & ~(Qt.KeyboardModifier.ShiftModifier | Qt.KeyboardModifier.KeypadModifier
                                     | Qt.KeyboardModifier.GroupSwitchModifier)
        altgr = Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.AltModifier
        if text == "/" and mods == altgr:
            mods = Qt.KeyboardModifier.NoModifier
        if len(text) != 1 or not (text.isalnum() or text in "^./") or event.isAutoRepeat() or mods:
            return False
        app = QApplication.instance()
        focus = app.focusWidget()
        if (not self.isVisible() or not self.canvas.underMouse() or app.activeModalWidget() is not None
                or app.activePopupWidget() is not None or not self.window().isActiveWindow()
                or isinstance(focus, (QLineEdit, QAbstractSpinBox, QTextEdit, QPlainTextEdit))):
            return False
        if text == "/":  # the character, so any layout's slash works (Shift+7 on a German keyboard)
            self.open_studies()
        else:
            self.symbols.open_dialog(text.upper())
        return True

    def _context_menu(self, scene_pos, global_pos):
        """Secondary actions: right click, or long press on a touchscreen."""
        plot = self._plot_at(scene_pos)
        menu = QMenu(self)
        menu.addAction("Reset zoom", self.reset_view)
        menu.addAction("Indicators and signals …  /", lambda: self.open_studies())
        if plot is self.price and self.df is not None:
            price = self.price_at(self.price.vb.mapSceneToView(scene_pos).y())
            menu.addAction(f"Add level at {price:.6g}", lambda: self._add_level(price))
            if self._loaded_symbol:
                menu.addAction(f"Alert me at {price:.6g}", lambda: self.add_price_alert(price))
        for key, btn in self.tools.items():
            if key != "hline":
                menu.addAction(f"Draw {btn.text().split(' ', 1)[-1].lower()}", lambda k=key: self.tools[k].setChecked(True))
        hit = self.drawing_at(scene_pos)
        if hit is not None:
            menu.addSeparator()
            menu.addAction("Change colour …", lambda: self._pick_color(hit))
            if isinstance(hit, NoteItem):
                menu.addAction("Edit text …", lambda: self._edit_note(hit))
            menu.addAction("Delete drawing", lambda: self.delete_drawing(hit))
            menu.addSeparator()
        if self.drawings:
            menu.addAction("Clear drawings", self.clear_drawings)
        if self._loaded_symbol:
            lists = menu.addMenu("Add to watchlist")
            for name in self.watchlists.lists:
                lists.addAction(name, lambda n=name, s=self._loaded_symbol: self.watchlists.add_symbol(s, n))
        menu.addSeparator()
        menu.addAction("Line chart" if not self.style_btn.isChecked() else "Candles", self.style_btn.toggle)
        menu.addAction("Volume overlay off" if self.volume_overlay else "Volume overlay on", self._toggle_volume_overlay)
        menu.addAction("Hide volume" if self.vol_btn.isChecked() else "Show volume", self.vol_btn.toggle)
        menu.popup(global_pos)
        self._menu = menu  # keep a reference while open (and for tests)

    def add_price_alert(self, price: float):
        """A background price alert (#156): fires when the price reaches ``price`` from the side it is on now."""
        from tradebot import alerts

        last = float(self.df["close"].iloc[-1])
        alert = alerts.Alert("price", [self._loaded_symbol], "1m", op=">=" if price >= last else "<=", value=price)
        try:
            alerts.add(alert)
        except (ValueError, OSError) as exc:
            self.status.emit(f"alert not saved: {exc}")
            return
        self.status.emit(f"alert added: {alert.describe()} (checked while the app or `tradebot watch` runs)")

    def _add_drawing(self, item, tool, save: bool = True):
        item._kind = tool
        self._hook_snapping(item)
        self.price.addItem(item)
        self.drawings.append(item)
        self.tools[tool].setChecked(False)
        if isinstance(item, NoteItem):
            item.on_moved = self._save_drawings
        elif isinstance(item, pg.InfiniteLine):
            item.sigPositionChangeFinished.connect(self._save_drawings)
        else:
            item.sigRegionChangeFinished.connect(self._save_drawings)
        if save:
            self._save_drawings()
            self.status.emit(f"{tool} added — drag to adjust, right-click to recolour or delete")

    # -- drawings stored per symbol (#160) --------------------------------------------------------
    DRAWINGS_KEY = "chart.drawings"

    def drawing_at(self, scene_pos):
        """The topmost drawing under a scene position, or None."""
        for item in reversed(self.drawings):
            if item.shape().contains(item.mapFromScene(scene_pos)):
                return item
        return None

    def drawing_specs(self) -> list[dict]:
        return [s for s in (to_spec(i, getattr(i, "_kind", ""), self.price_at) for i in self.drawings) if s]

    def _save_drawings(self, *_):
        """Remember the drawings of the shown symbol; they come back with it, on any timeframe."""
        if self.settings is None or not self._loaded_symbol:
            return
        stored = dict(self.settings.get(self.DRAWINGS_KEY, {}))
        specs = self.drawing_specs()
        if specs:
            stored[self._loaded_symbol] = specs
        else:
            stored.pop(self._loaded_symbol, None)
        self.settings.set(self.DRAWINGS_KEY, stored)

    def _switch_drawings(self, symbol: str):
        """Another symbol is shown: swap in its drawings (the old ones were saved as they changed)."""
        if self._loaded_symbol:
            self._remove_drawings()
        for spec in (self.settings.get(self.DRAWINGS_KEY, {}).get(symbol, []) if self.settings else []):
            try:
                self._restore(spec)
            except (KeyError, TypeError, ValueError, IndexError):
                continue  # a broken entry never stops the others

    def _restore(self, spec: dict):
        kind, color = spec["kind"], spec.get("color")
        if kind == "hline":
            self._add_level(spec["price"], color, save=False)
        elif kind == "vline":
            self._add_vline(spec["x"], color, save=False)
        elif kind == "text":
            self._add_note(spec["x"], spec["price"], spec["text"], color, save=False)
        elif kind == "rect":
            y0, y1 = self._ty(spec["p0"]), self._ty(spec["p1"])
            item = pg.RectROI([spec["x0"], min(y0, y1)], [spec["x1"] - spec["x0"], abs(y1 - y0)],
                              pen=pg.mkPen(qcolor(color or self._accent_hex()), width=2))
            self._add_drawing(item, "rect", save=False)
        elif kind in ("trend", "fib", "measure"):
            (x1, p1), (x2, p2) = spec["points"]
            pts = [[x1, float(self._ty(p1))], [x2, float(self._ty(p2))]]
            color = color or self._accent_hex()
            if kind == "trend":
                item = pg.LineSegmentROI(pts, pen=pg.mkPen(qcolor(color), width=2))
            else:
                item = self._two_point(kind, *pts, color)
            self._add_drawing(item, kind, save=False)

    def set_drawing_color(self, item, color: str):
        set_color(item, color)
        self._save_drawings()

    def _pick_color(self, item):
        color = QColorDialog.getColor(QColor(color_of(item)), self, "Drawing colour")
        if color.isValid():
            self.set_drawing_color(item, color.name())

    def _edit_note(self, item):
        text = self._ask_text(item.toPlainText())
        if text:
            item.setText(text)
            self._save_drawings()

    def delete_drawing(self, item):
        if item in self.drawings:
            self.price.removeItem(item)
            self.drawings.remove(item)
            self._save_drawings()

    def _remove_drawings(self):
        for item in self.drawings:
            self.price.removeItem(item)
        self.drawings.clear()

    # -- loadouts and the remembered session (#54) ------------------------------------------------
    SESSION_KEY = "chart.session"

    def _fill_loadout_menu(self):
        """One row per saved loadout (click: apply, ✕: delete after confirming), then "Save Current"."""
        menu = self.loadout_menu
        menu.clear()
        saved = self.loadouts.all()
        if not saved:
            menu.addAction("No saved loadouts").setEnabled(False)
        for name in saved:
            row = QWidget()
            lay = QHBoxLayout(row)
            lay.setContentsMargins(0, 0, 4, 0)
            lay.setSpacing(0)
            apply_btn = QPushButton(name, objectName="LoadoutName", toolTip=_describe(saved[name]))
            apply_btn.clicked.connect(lambda _, n=name: (menu.close(), self.apply_loadout(n)))
            delete = QToolButton(text="✕", objectName="LoadoutDelete", toolTip=f"Delete \u201c{name}\u201d")
            delete.clicked.connect(lambda _, n=name: (menu.close(), self.delete_loadout(n)))
            lay.addWidget(apply_btn, 1)
            lay.addWidget(delete)
            action = QWidgetAction(menu)
            action.setDefaultWidget(row)
            menu.addAction(action)
        menu.addSeparator()
        menu.addAction("Save Current", self.save_current_loadout)

    def save_current_loadout(self):
        saved = self.loadouts.all()
        suggestion = next(f"Loadout {i}" for i in range(1, len(saved) + 2) if f"Loadout {i}" not in saved)
        name, ok = QInputDialog.getText(self, "Save loadout", "Name for the current indicators and signal:",
                                        QLineEdit.EchoMode.Normal, suggestion)
        name = name.strip()
        if not ok or not name:
            return
        if name in self.loadouts.all() and not _confirm(self, "Replace loadout",
                                                        f"A loadout \u201c{name}\u201d exists. Replace it?"):
            return
        self.loadouts.save(name, capture(self.indicators, self.algorithm, self.style_btn.isChecked()))
        self.status.emit(f"loadout \u201c{name}\u201d saved")

    def delete_loadout(self, name: str):
        if _confirm(self, "Delete loadout", f"Delete the loadout \u201c{name}\u201d?"):
            self.loadouts.delete(name)
            self.status.emit(f"loadout \u201c{name}\u201d deleted")

    def apply_loadout(self, name: str):
        data = self.loadouts.all().get(name)
        if data is not None:
            note = self._apply_setup(data)
            self.status.emit(f"loadout \u201c{name}\u201d applied{note}")

    def _apply_setup(self, data: dict) -> str:
        """Replace the indicators, signal and chart style in one redraw; returns a note on what was skipped."""
        self.indicators, self.algorithm, line, skipped = restore(data)
        self._slots.clear()  # colours as when the loadout was made: palette from the start, in its order
        self._next_slot = 0
        self.style_btn.blockSignals(True)  # no extra redraw
        self.style_btn.setChecked(line)
        self.style_btn.blockSignals(False)
        self.redraw()
        return f" (skipped, not installed: {', '.join(skipped)})" if skipped else ""

    def _remember_session(self):
        """The current symbol and setup become the default for the next start, saved or not."""
        if self.settings is None or not self._loaded_symbol:
            return
        session = capture(self.indicators, self.algorithm, self.style_btn.isChecked())
        session.update(symbol=self._loaded_symbol, tf=self.tf.currentText(), bars=self.bars.currentText())
        if self.settings.get(self.SESSION_KEY) != session:
            self.settings.set(self.SESSION_KEY, session)

    def _restore_session(self):
        session = self.settings.get(self.SESSION_KEY) if self.settings is not None else None
        if not session:
            return
        for widget, value in ((self.tf, session.get("tf")), (self.bars, session.get("bars"))):
            if value and widget.findText(value) >= 0:
                widget.blockSignals(True)  # no load yet: the app loads once everything is set
                widget.setCurrentText(value)
                widget.blockSignals(False)
        if session.get("symbol"):
            self.symbol.setText(session["symbol"])
        kind = self.settings.get("chart.type", "Standard")
        if kind in CHART_TYPES and kind not in ("Compare", "Ratio"):
            self.type_box.blockSignals(True)
            self.type_box.setCurrentText(kind)
            self.type_box.blockSignals(False)
            self._type = kind
        self._pending_note = self._apply_setup(session)

    def clear_drawings(self):
        self._remove_drawings()
        self._save_drawings()


def _confirm(parent, title: str, question: str) -> bool:
    answer = QMessageBox.question(parent, title, question,
                                  QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                  QMessageBox.StandardButton.No)
    return answer == QMessageBox.StandardButton.Yes


def _describe(loadout: dict) -> str:
    """Tooltip: what a loadout contains."""
    parts = [" ".join([i["name"], *(f"{v:g}" for v in i.get("params", {}).values())]) for i in loadout["indicators"]]
    if loadout.get("signal"):
        parts.append(f"signal: {loadout['signal']['name']}")
    return ", ".join(parts) or "no indicators"
