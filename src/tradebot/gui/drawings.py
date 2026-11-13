# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Chart drawings beyond the basic ROIs: Fibonacci retracement, price/time measure, text notes, and the
conversion of every drawing to and from plain dicts so they can be stored per symbol (#160).

x is epoch seconds and y the price pane's view y; the chart passes ``ty`` (price -> view y) and ``price_at``
(view y -> price) so the items stay at their prices on the logarithmic scale."""

from __future__ import annotations

import pyqtgraph as pg
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QPainterPath

FIB_LEVELS = (0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618)


def _fmt(v: float) -> str:
    return f"{v:.6g}"


class _TwoPoint(pg.LineSegmentROI):
    """A segment ROI with price-aware labels; subclasses say what to draw around the two ends."""

    def __init__(self, p1, p2, ty, price_at, color):
        super().__init__([list(p1), list(p2)], pen=pg.mkPen(color, width=2))
        self.ty, self.price_at = ty, price_at
        self._texts: list[pg.TextItem] = []
        self.sigRegionChanged.connect(self.refresh)
        self.refresh()

    def ends(self) -> list[QPointF]:
        """The two ends in the parent (view) coordinates."""
        return [self.mapToParent(h.pos()) for h in self.getHandles()]

    def prices(self) -> tuple[float, float]:
        a, b = self.ends()
        return self.price_at(a.y()), self.price_at(b.y())

    def _text(self, i: int, text: str, view_x: float, view_y: float):
        while len(self._texts) <= i:
            t = pg.TextItem(anchor=(0, 0.5), color=self.pen.color())
            t.setParentItem(self)
            t.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            self._texts.append(t)
        t = self._texts[i]
        t.setColor(self.pen.color())
        t.setText(text)
        t.setPos(self.mapFromParent(QPointF(view_x, view_y)))
        t.setVisible(True)
        return t

    def _hide_from(self, i: int):
        for t in self._texts[i:]:
            t.setVisible(False)

    def lines(self) -> list[tuple[float, float, float, float]]:
        """Extra segments (parent coordinates) drawn besides the ROI's own line."""
        return []

    def refresh(self, *_):
        self.prepareGeometryChange()
        self._relabel()
        self.update()

    def _relabel(self):
        self._hide_from(0)

    def shape(self):
        path = QPainterPath()
        for x0, y0, x1, y1 in self.lines():
            a, b = self.mapFromParent(QPointF(x0, y0)), self.mapFromParent(QPointF(x1, y1))
            path.addRect(QRectF(a, b).normalized().adjusted(0, 0, 0, 0))
        return path.united(super().shape())

    def boundingRect(self):
        return self.shape().boundingRect().adjusted(-1, -1, 1, 1)

    def paint(self, p, *args):
        super().paint(p, *args)
        pen = pg.mkPen(self.pen.color(), width=1, style=Qt.PenStyle.DashLine)
        pen.setCosmetic(True)
        p.setPen(pen)
        for x0, y0, x1, y1 in self.lines():
            p.drawLine(self.mapFromParent(QPointF(x0, y0)), self.mapFromParent(QPointF(x1, y1)))


class FibItem(_TwoPoint):
    """Fibonacci retracement/extension between two prices: horizontal levels from the first end's price (0)
    to the second's (1) and beyond."""
    kind = "fib"

    def _levels(self):
        (a, b), (p1, p2) = self.ends(), self.prices()
        x0, x1 = min(a.x(), b.x()), max(a.x(), b.x())
        return x0, x1, [(r, p1 + (p2 - p1) * r) for r in FIB_LEVELS]

    def lines(self):
        x0, x1, levels = self._levels()
        return [(x0, self.ty(price), x1, self.ty(price)) for _, price in levels]

    def _relabel(self):
        x0, x1, levels = self._levels()
        for i, (r, price) in enumerate(levels):
            self._text(i, f"{r:g}  ({_fmt(price)})", x1, self.ty(price))
        self._hide_from(len(levels))


class MeasureItem(_TwoPoint):
    """Price range measure: the price and percent change between two points and the time between them."""
    kind = "measure"

    def lines(self):
        a, b = self.ends()
        y0, y1 = a.y(), b.y()
        return [(a.x(), y0, b.x(), y0), (b.x(), y0, b.x(), y1)]

    def _relabel(self):
        a, b = self.ends()
        p1, p2 = self.prices()
        secs = abs(b.x() - a.x())
        days, rest = divmod(secs, 86400)
        span = f"{int(days)}d {int(rest // 3600)}h" if days else f"{int(rest // 3600)}h {int(rest % 3600 // 60)}m"
        pct = (p2 / p1 - 1) * 100 if p1 else float("nan")
        self._text(0, f"{p2 - p1:+.6g} ({pct:+.2f}%) · {span}", b.x(), b.y())


class NoteItem(pg.TextItem):
    """A text note anchored at a time and price; drag to move."""
    kind = "text"

    def __init__(self, x, y, text, color, on_moved=None):
        super().__init__(text, color=color, anchor=(0, 1), border=pg.mkPen(color), fill=pg.mkBrush(0, 0, 0, 90))
        self.setPos(x, y)
        self.on_moved = on_moved
        self._grab = QPointF()
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)

    def mouseDragEvent(self, ev):
        vb = self.getViewBox()
        if ev.button() != Qt.MouseButton.LeftButton or vb is None:
            ev.ignore()
            return
        ev.accept()
        if ev.isStart():
            self._grab = self.pos() - vb.mapSceneToView(ev.buttonDownScenePos())
        self.setPos(vb.mapSceneToView(ev.scenePos()) + self._grab)
        if ev.isFinish() and self.on_moved:
            self.on_moved()


def set_color(item, color: str):
    """Recolour any drawing."""
    c = QColor(color)
    if isinstance(item, NoteItem):
        item.setColor(c)
        item.border = pg.mkPen(c)
        item.update()
        return
    pen = pg.mkPen(c, width=2)
    if isinstance(item, pg.InfiniteLine):
        item.setPen(pen)
        item.currentPen = pen
        if item.label is not None:
            item.label.setColor(c)
        item.update()
    else:
        item.setPen(pen)
        if isinstance(item, _TwoPoint):
            item.refresh()


def color_of(item) -> str:
    if isinstance(item, NoteItem):
        return pg.mkColor(item.color).name()
    pen = item.pen
    return pen.color().name()


def to_spec(item, kind: str, price_at) -> dict | None:
    """The drawing as a JSON-able dict (times in epoch seconds, prices as prices)."""
    spec = {"kind": kind, "color": color_of(item)}
    if kind == "hline":
        spec["price"] = price_at(item.value())
    elif kind == "vline":
        spec["x"] = item.value()
    elif kind == "rect":
        (x0, y0), (w, h) = item.pos(), item.size()
        spec.update(x0=x0, x1=x0 + w, p0=price_at(y0), p1=price_at(y0 + h))
    elif kind in ("trend", "fib", "measure"):
        pts = [item.mapToParent(h.pos()) for h in item.getHandles()]
        spec["points"] = [[p.x(), price_at(p.y())] for p in pts]
    elif kind == "text":
        spec.update(x=item.pos().x(), price=price_at(item.pos().y()), text=item.toPlainText())
    else:
        return None
    return spec
