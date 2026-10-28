# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Interactive OHLC chart: candles/line, indicator overlays and panes,
algorithm buy/sell markers, crosshair, and drawing tools
(trendline, horizontal support/resistance line, rectangle)."""

from __future__ import annotations

from html import escape

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt6.QtCore import QEvent, QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QPainter, QPicture
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLineEdit,
    QMenu,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from tradebot.algorithms import ALGORITHMS, build_algorithm
from tradebot.config import Settings
from tradebot.core.timeframes import TIMEFRAMES
from tradebot.gui.common import DateRange, SymbolCompleter, epoch_seconds, run_async, shortfall
from tradebot.gui.params import ParamDialog
from tradebot.gui.theme import Theme, ThemeManager, qcolor, resolve_color
from tradebot.gui.touch import PlotTouchHandler, TouchController
from tradebot.indicators import INDICATORS, Style
from tradebot.symbols import SymbolSearch, symbol_name

pg.setConfigOptions(antialias=True)
DASHES = {"dash": Qt.PenStyle.DashLine, "dot": Qt.PenStyle.DotLine}  #: Style.dash -> Qt pen style
FIT_PADDING = 0.05  #: auto-fit margin above and below the visible data, as a fraction of its span


def _decimals(v: float) -> int:
    """About five significant digits: 93,564 · 184.20 · 0.12345."""
    v = abs(float(v))
    return 2 if v == 0 or not np.isfinite(v) else int(np.clip(4 - np.floor(np.log10(v)), 0, 8))


def fmt_value(v) -> str:
    """A value for the cursor readout; NaN (e.g. an indicator's warm-up) shows as a dash."""
    v = float(v)
    return "–" if not np.isfinite(v) else f"{v:,.{_decimals(v)}f}"


def _spaced_label(name: str, values: dict) -> str:
    """``rsi 14`` / ``macd 12 26 9``: the name followed by the parameter values, as in the readout."""
    return " ".join([name, *(f"{v:g}" if isinstance(v, float) else str(v) for v in values.values())])


def nearest_bar(x, t: float) -> int:
    """Index of the bar in the sorted ``x`` closest to time ``t``."""
    i = int(np.clip(np.searchsorted(x, t), 1, len(x) - 1)) if len(x) > 1 else 0
    return i - 1 if len(x) > 1 and t - x[i - 1] < x[i] - t else i


def visible_y_range(x, series, x0, x1, padding=FIT_PADDING):
    """Padded (low, high) of all ``series`` (arrays aligned with the sorted ``x``) over ``x0 <= x <= x1``.

    NaNs are ignored; returns None if nothing is visible.
    """
    i0, i1 = np.searchsorted(x, x0, "left"), np.searchsorted(x, x1, "right")
    visible = [np.asarray(s, dtype=float)[i0:i1] for s in series]
    visible = np.concatenate(visible) if visible else np.empty(0)
    visible = visible[~np.isnan(visible)]
    if not visible.size:
        return None
    lo, hi = float(visible.min()), float(visible.max())
    span = (hi - lo) or abs(hi) * 0.01 or 1.0  # flat data: open up a small band around it
    return lo - padding * span, hi + padding * span


class CandlestickItem(pg.GraphicsObject):
    """Candles pre-rendered into a QPicture (fast pan/zoom for thousands of bars)."""

    def __init__(self):
        super().__init__()
        self.picture = QPicture()

    def set_data(self, x, o, h, l, c, width, up, down):
        self.picture = QPicture()
        p = QPainter(self.picture)
        for color, mask in ((up, c >= o), (down, c < o)):
            pen = pg.mkPen(color, width=1)
            pen.setCosmetic(True)
            p.setPen(pen)
            p.setBrush(pg.mkBrush(color))
            for xi, oi, hi, li, ci in zip(x[mask], o[mask], h[mask], l[mask], c[mask]):
                p.drawLine(QPointF(xi, li), QPointF(xi, hi))
                p.drawRect(QRectF(xi - width / 2, oi, width, (ci - oi) or 1e-12))
        p.end()
        self.prepareGeometryChange()
        self.update()

    def paint(self, p, *args):
        p.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        return QRectF(self.picture.boundingRect())


class ChartView(QWidget):
    status = pyqtSignal(str)

    def __init__(self, engine, themes: ThemeManager, watchlist: list[str], settings: Settings | None = None,
                 symbol_search: SymbolSearch | None = None):
        super().__init__()
        self.engine, self.themes, self.settings = engine, themes, settings
        self.df: pd.DataFrame | None = None
        self.indicators: list = []
        self.algorithm = None
        self.drawings: list = []
        self._pending_point = None
        self._x = np.empty(0)
        self._fit_series: dict = {}  # plot -> the arrays its y-axis auto-fit covers
        self._adjusting = False  # set while the chart changes view ranges itself, as opposed to the user
        self._title = ""  # "SYMBOL · 1d" of the loaded data
        self._hover: int | None = None  # bar under the cursor; None = show the latest bar
        self._rows: dict = {}  # plot -> [(label, [(values, colour), …])] for the value lines (#45)
        self._legend: dict = {}  # plot -> LabelItem in its top-left corner
        self._positions = None  # the signal's target position per bar
        self._slots: dict[int, int] = {}  # id(indicator) -> first palette index of its outputs
        self._next_slot = 0

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
        self.style_btn = QToolButton(text="Line", checkable=True)
        self.style_btn.toggled.connect(lambda _: self.redraw())

        self.ind_btn = QToolButton(text="Indicators ▾")
        self.ind_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.ind_menu = QMenu(self)
        self.ind_menu.aboutToShow.connect(self._fill_indicator_menu)
        self.ind_btn.setMenu(self.ind_menu)

        self.algo_box = QComboBox()
        self.algo_box.addItem("No signals")
        self.algo_box.addItems(ALGORITHMS.names())
        self.algo_box.currentTextChanged.connect(self._choose_algorithm)

        self.tools: dict[str, QToolButton] = {}
        tool_group = []
        for key, label, tip in (("trend", "╱ Trend", "Trendline: click two points"),
                                ("hline", "― Level", "Support/resistance: click a price"),
                                ("rect", "▭ Zone", "Rectangle: click two corners")):
            btn = QToolButton(text=label, checkable=True, toolTip=tip)
            btn.toggled.connect(lambda on, k=key: self._tool_toggled(k, on))
            self.tools[key] = btn
            tool_group.append(btn)
        clear = QToolButton(text="✕ Clear", toolTip="Remove all drawings")
        clear.clicked.connect(self.clear_drawings)

        bar = QHBoxLayout()
        for w in (self.symbol, self.tf, self.bars, self.dates, load, self.style_btn, self.ind_btn, self.algo_box):
            bar.addWidget(w)
        bar.addSpacing(12)
        for w in (*tool_group, clear):
            bar.addWidget(w)
        bar.addStretch()

        # -- plot area ---------------------------------------------------------------------------
        self.canvas = pg.GraphicsLayoutWidget()
        self.price = self.canvas.addPlot(row=0, col=0, axisItems={"bottom": pg.DateAxisItem()})
        self.price.showGrid(x=True, y=True, alpha=0.15)
        self.price.setMenuEnabled(False)
        self.price.getAxis("right").setWidth(70)
        self.price.showAxis("right")
        self.price.hideAxis("left")
        self.panes: list[pg.PlotItem] = []
        # crosshair: per pane a vertical line (shown in every pane) and a horizontal one (hovered pane only)
        self._cross: dict = {}
        self.vline, self.hline = self._add_crosshair(self.price)
        self._add_legend(self.price)
        self.canvas.scene().sigMouseMoved.connect(self._mouse_moved)
        self.canvas.scene().sigMouseClicked.connect(self._mouse_clicked)
        self.price.vb.sigRangeChanged.connect(self._range_changed)

        self.fit_btn = QToolButton(self.canvas, text="↕", checkable=True, toolTip="Auto-fit price axis to visible candles")
        # on by default (a checked= keyword is applied before checkable and lost)
        self.fit_btn.setChecked(bool(settings.get("chart.autofit", True)) if settings else True)
        size = 40 if themes.touch else 28  # finger-sized in touch mode
        self.fit_btn.setFixedSize(size, size)
        self.fit_btn.toggled.connect(self._autofit_toggled)
        self.canvas.installEventFilter(self)  # keeps the button in the bottom-left corner

        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.addLayout(bar)
        lay.addWidget(self.canvas, 1)
        self.touch = TouchController(self.canvas.viewport(), PlotTouchHandler(
            self.canvas, self._plot_at, on_tap=self._tap, on_long_press=self._context_menu, on_reset=self.reset_view))
        themes.changed.connect(lambda _: self.redraw())

    # -- data ------------------------------------------------------------------------------------
    def open_symbol(self, symbol: str):
        self.symbol.setText(symbol)
        self.load()

    def load(self):
        symbol, tf, bars = self.symbol.text().strip(), self.tf.currentText(), int(self.bars.currentText())
        if not symbol:
            return
        self.status.emit(f"loading {symbol} {tf} …")
        start, end = self.dates.range()
        run_async(lambda: self.engine.get_ohlc(symbol, tf, start=start, end=end, bars=bars),
                  lambda df: self._loaded(symbol, df, shortfall(df, bars, start)),
                  lambda err: self.status.emit(f"{symbol}: {err}"))

    def _loaded(self, symbol, df, note=""):
        self.df = df
        tf = self.tf.currentText()
        self._title = f"{symbol} · {tf}"  # the ticker until the full name arrives
        self._hover = None
        run_async(lambda: symbol_name(symbol), lambda name: self._named(symbol, tf, name))
        self.status.emit(f"{symbol}: {len(df)} candles {df.index[0]:%Y-%m-%d} → {df.index[-1]:%Y-%m-%d %H:%M}{note}")
        self.redraw(autorange=True)

    def _named(self, symbol, tf, name):
        """Full instrument name for the price pane's title, if the chart still shows that symbol."""
        if name and self._title == f"{symbol} · {tf}":
            self._title = f"{name} · {tf}"
            self._show_values()

    # -- indicators & algorithms -----------------------------------------------------------------
    def _fill_indicator_menu(self):
        self.ind_menu.clear()
        for ind in self.indicators:
            act = self.ind_menu.addAction(f"✓ {ind.label}  (remove)")
            act.triggered.connect(lambda _, i=ind: self._remove_indicator(i))
        if self.indicators:
            self.ind_menu.addSeparator()
        for cls in INDICATORS:
            act = self.ind_menu.addAction(f"{cls.name} — {cls.description}")
            act.triggered.connect(lambda _, c=cls: self._add_indicator(c))

    def _add_indicator(self, cls):
        values = ParamDialog.ask(cls.name, cls.params, parent=self)
        if values is not None:
            self.indicators.append(cls(**values))
            self.redraw()

    def _remove_indicator(self, ind):
        self.indicators.remove(ind)
        self.redraw()

    def _choose_algorithm(self, name):
        if name not in ALGORITHMS:
            self.algorithm = None
        else:
            algo = build_algorithm(name)
            values = ParamDialog.ask(f"{name} parameters", algo.param_space(), algo.values, self)
            self.algorithm = algo.with_params(**values) if values is not None else None
        self.redraw()

    def set_algorithm(self, algorithm):
        """Show an externally configured algorithm (e.g. from the backtest lab)."""
        self.algo_box.blockSignals(True)
        self.algo_box.setCurrentText(algorithm.name)
        self.algo_box.blockSignals(False)
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

        def colored(v, color):
            return f"<span style='color:{color}'>{fmt_value(v)}</span>"

        def line(label, values):
            return " ".join([muted(escape(label)), *(colored(series[i], color) for series, color in values)])

        row = self.df.iloc[i]
        direction = t.up if row.close >= row.open else t.down
        ohlc = " ".join(f"{muted(k)}{colored(row[c], direction)}" for k, c in
                        (("O", "open"), ("H", "high"), ("L", "low"), ("C", "close")))
        head = [f"<span style='color:{t.text}'>{escape(self._title)}</span>" if self._title else "", ohlc]
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
        for item in list(self.price.items):
            if item not in (self.vline, self.hline, *self.drawings):
                self.price.removeItem(item)
        for pane in self.panes:
            self.canvas.removeItem(pane)
            self._cross.pop(pane, None)
            self._legend.pop(pane, None)
        self.panes.clear()
        self._fit_series = {}
        self._rows = {self.price: []}
        self._positions = None
        self._style(t)
        if self.df is None or self.df.empty:
            self._legend[self.price].setText("")
            return
        df = self.df
        x = self._x = epoch_seconds(df.index)
        width = (np.median(np.diff(x)) if len(x) > 1 else 86400) * 0.7
        o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
        if self.style_btn.isChecked():
            self.price.plot(x, c, pen=pg.mkPen(qcolor(t.accent), width=2))
            self._fit_series[self.price] = [c]
        else:
            self._fit_series[self.price] = [l, h]
            candles = CandlestickItem()
            candles.set_data(x, o, h, l, c, width, qcolor(t.up), qcolor(t.down))
            self.price.addItem(candles)

        for ind in self.indicators:
            frame = ind(df)
            target = self.price
            if not ind.overlay:
                target = self.canvas.addPlot(row=len(self.panes) + 1, col=0, axisItems={"bottom": pg.DateAxisItem()})
                target.setXLink(self.price)
                target.setMaximumHeight(170)
                target.showGrid(x=True, y=True, alpha=0.15)
                target.showAxis("right")
                target.hideAxis("left")
                target.getAxis("right").setWidth(70)
                target.setMenuEnabled(False)
                for level in ind.levels:
                    target.addItem(pg.InfiniteLine(level, angle=0, pen=pg.mkPen(qcolor(t.muted), style=Qt.PenStyle.DotLine)))
                target.vb.sigRangeChanged.connect(self._range_changed)
                self._add_crosshair(target)
                self._add_legend(target)
                self.panes.append(target)
            values = []  # this indicator's drawn outputs for the value line
            self._rows.setdefault(target, []).append((_spaced_label(ind.name, ind.values), values))
            slot = self._palette_slot(ind)
            for k, col in enumerate(frame.columns):
                if not ind.is_line(col):
                    self._draw_labels(target, ind, frame, col, x, c, t)
                    continue
                y = frame[col].to_numpy()
                ok = ~np.isnan(y)
                style = ind.styles.get(ind.output_of(col), Style())
                hist = col.endswith("_hist")
                default = t.muted if hist else t.series[(slot + k) % len(t.series)]
                color = qcolor(resolve_color(t, style.color) if style.color else default)
                if hist:
                    target.addItem(pg.BarGraphItem(x=x[ok], height=y[ok], width=width, brush=pg.mkBrush(color)))
                    self._fit_series.setdefault(target, []).append(np.zeros_like(y))  # bars grow from 0
                else:
                    pen = pg.mkPen(color, width=style.width or 1.6, style=DASHES.get(style.dash, Qt.PenStyle.SolidLine))
                    target.plot(x[ok], y[ok], pen=pen, name=col)
                color = color.name()  # "#rrggbb" for the value line's rich text
                self._fit_series.setdefault(target, []).append(y)
                values.append((y, color))
        if self.algorithm is not None:
            self._positions = self.algorithm.run(df).to_numpy()
            change = np.diff(np.r_[0.0, self._positions])
            for mask, symbol, color, anchor in ((change > 0, "t1", t.up, l * 0.995), (change < 0, "t", t.down, h * 1.005)):
                if mask.any():
                    self.price.addItem(pg.ScatterPlotItem(x=x[mask], y=anchor[mask], symbol=symbol, size=13,
                                                          brush=pg.mkBrush(color), pen=pg.mkPen(None)))
        self._style(t)
        self._show_values()
        self._lock_view()
        if autorange:
            self.reset_view()
        else:
            self.fit_y()

    def _palette_slot(self, ind) -> int:
        """Where an indicator's outputs start in the theme palette, fixed when it's first drawn: removing or
        adding other indicators doesn't recolour it."""
        if id(ind) not in self._slots:
            self._slots[id(ind)] = self._next_slot
            self._next_slot += len(ind.outputs)
        return self._slots[id(ind)]

    def _draw_labels(self, target, ind, frame, col, x, close, t):
        """Text labels of an indicator output (e.g. Elliott wave numbers) at its anchor values."""
        texts = ind.labels.get(ind.output_of(col))
        if not texts:
            return  # hidden output
        anchor_col = f"{ind.name}_{ind.label_anchor}" if ind.label_anchor else None
        anchor = frame[anchor_col].to_numpy() if anchor_col in frame else close
        codes = frame[col].to_numpy()
        for i in np.flatnonzero(~np.isnan(codes)):
            # label above swing highs (local maximum of the anchor line), below swing lows
            neighbours = anchor[max(i - 1, 0): i + 2]
            is_high = anchor[i] >= np.nanmax(neighbours)
            item = pg.TextItem(texts.get(float(codes[i]), "?"), color=t.text, anchor=(0.5, 1.15 if is_high else -0.15))
            font = item.textItem.font()
            font.setBold(True)
            item.setFont(font)
            item.setPos(x[i], anchor[i])
            target.addItem(item)

    # -- crosshair & drawing tools ---------------------------------------------------------------
    def _mouse_moved(self, pos):
        """Crosshair through every pane at the hovered bar, from whichever pane the cursor is over (#46)."""
        plot = self._plot_at(pos)
        if self.df is None or self.df.empty or plot is None:
            return
        pt = plot.vb.mapSceneToView(pos)
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
                         f"L {row.low:.6g}  C {row.close:.6g}  ({chg * 100:+.2f}%)   cursor {pt.y():.6g}")

    def _tool_toggled(self, key, on):
        if on:
            for k, btn in self.tools.items():
                if k != key:
                    btn.setChecked(False)
        self._pending_point = None

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
        if tool == "hline":
            self._add_level(pt.y())
            return
        if self._pending_point is None:
            self._pending_point = pt
            self.status.emit("tap/click the second point")
            return
        p1, self._pending_point = self._pending_point, None
        pen = pg.mkPen(qcolor(self.themes.current.accent), width=2)
        if tool == "trend":
            item = pg.LineSegmentROI([[p1.x(), p1.y()], [pt.x(), pt.y()]], pen=pen)
        else:
            item = pg.RectROI([min(p1.x(), pt.x()), min(p1.y(), pt.y())], [abs(pt.x() - p1.x()), abs(pt.y() - p1.y())],
                              pen=pen)
        self._add_drawing(item, tool)

    def _add_level(self, price: float):
        t = self.themes.current
        item = pg.InfiniteLine(price, angle=0, movable=True, pen=pg.mkPen(qcolor(t.accent), width=2),
                               label="{value:.6g}", labelOpts={"position": 0.97, "color": t.accent})
        self._add_drawing(item, "hline")

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

    def fit_y(self):
        """Fit each plot's y-axis to its data inside the visible time range (while auto-fit is on)."""
        if not self.autofit:
            return
        x0, x1 = self.price.vb.viewRange()[0]
        self._adjusting = True
        try:
            for plot in [self.price, *self.panes]:
                plot.enableAutoRange(y=False)  # pyqtgraph would otherwise fit y to all data
                bounds = visible_y_range(self._x, self._fit_series.get(plot, []), x0, x1)
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

    def eventFilter(self, obj, event):
        if obj is self.canvas and event.type() == QEvent.Type.Resize:
            self.fit_btn.move(8, self.canvas.height() - self.fit_btn.height() - 8)
        elif obj is self.canvas and event.type() == QEvent.Type.Leave and self._hover is not None:
            self._hover = None  # cursor left the chart: show the latest bar again
            self._show_values()
        return super().eventFilter(obj, event)

    def _context_menu(self, scene_pos, global_pos):
        """Secondary actions: right click, or long press on a touchscreen."""
        plot = self._plot_at(scene_pos)
        menu = QMenu(self)
        menu.addAction("Reset zoom", self.reset_view)
        if plot is self.price and self.df is not None:
            price = self.price.vb.mapSceneToView(scene_pos).y()
            menu.addAction(f"Add level at {price:.6g}", lambda: self._add_level(price))
        for key, btn in self.tools.items():
            if key != "hline":
                menu.addAction(f"Draw {btn.text().split(' ', 1)[-1].lower()}", lambda k=key: self.tools[k].setChecked(True))
        if self.drawings:
            menu.addAction("Clear drawings", self.clear_drawings)
        menu.addSeparator()
        menu.addAction("Line chart" if not self.style_btn.isChecked() else "Candles", self.style_btn.toggle)
        menu.popup(global_pos)
        self._menu = menu  # keep a reference while open (and for tests)

    def _add_drawing(self, item, tool):
        self.price.addItem(item)
        self.drawings.append(item)
        self.tools[tool].setChecked(False)
        self.status.emit(f"{tool} added — drag to adjust, ✕ Clear to remove")

    def clear_drawings(self):
        for item in self.drawings:
            self.price.removeItem(item)
        self.drawings.clear()
