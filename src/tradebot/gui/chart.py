# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Interactive OHLC chart: candles/line, indicator overlays and panes,
algorithm buy/sell markers, crosshair, and drawing tools
(trendline, horizontal support/resistance line, rectangle)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QPainter, QPicture
from PyQt6.QtWidgets import (
    QComboBox,
    QCompleter,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from tradebot.algorithms import ALGORITHMS, build_algorithm
from tradebot.core.timeframes import TIMEFRAMES, to_timedelta
from tradebot.gui.common import epoch_seconds, run_async
from tradebot.gui.params import ParamDialog
from tradebot.gui.theme import Theme, ThemeManager, qcolor
from tradebot.gui.touch import PlotTouchHandler, TouchController
from tradebot.indicators import INDICATORS

pg.setConfigOptions(antialias=True)


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

    def __init__(self, engine, themes: ThemeManager, watchlist: list[str]):
        super().__init__()
        self.engine, self.themes = engine, themes
        self.df: pd.DataFrame | None = None
        self.indicators: list = []
        self.algorithm = None
        self.drawings: list = []
        self._pending_point = None

        # -- toolbar -----------------------------------------------------------------------------
        self.symbol = QLineEdit(watchlist[0] if watchlist else "BTC/USDT")
        self.symbol.setCompleter(QCompleter(watchlist))
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
        for w in (self.symbol, self.tf, self.bars, load, self.style_btn, self.ind_btn, self.algo_box):
            bar.addWidget(w)
        bar.addSpacing(12)
        for w in (*tool_group, clear):
            bar.addWidget(w)
        bar.addStretch()
        self.info = QLabel()
        self.info.setObjectName("Muted")
        bar.addWidget(self.info)

        # -- plot area ---------------------------------------------------------------------------
        self.canvas = pg.GraphicsLayoutWidget()
        self.price = self.canvas.addPlot(row=0, col=0, axisItems={"bottom": pg.DateAxisItem()})
        self.price.showGrid(x=True, y=True, alpha=0.15)
        self.price.setMenuEnabled(False)
        self.price.getAxis("right").setWidth(70)
        self.price.showAxis("right")
        self.price.hideAxis("left")
        self.panes: list[pg.PlotItem] = []
        self.vline = pg.InfiniteLine(angle=90, movable=False)
        self.hline = pg.InfiniteLine(angle=0, movable=False)
        self.price.addItem(self.vline, ignoreBounds=True)
        self.price.addItem(self.hline, ignoreBounds=True)
        self.canvas.scene().sigMouseMoved.connect(self._mouse_moved)
        self.canvas.scene().sigMouseClicked.connect(self._mouse_clicked)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.addLayout(bar)
        lay.addWidget(self.canvas, 1)
        self.touch = TouchController(self.canvas.viewport(), PlotTouchHandler(
            self.canvas, self._plot_at, on_tap=self._tap, on_long_press=self._context_menu))
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
        end = pd.Timestamp.now(tz="UTC")
        run_async(lambda: self.engine.get_ohlc(symbol, tf, start=end - bars * to_timedelta(tf), end=end),
                  lambda df: self._loaded(symbol, df), lambda err: self.status.emit(f"{symbol}: {err}"))

    def _loaded(self, symbol, df):
        self.df = df
        self.status.emit(f"{symbol}: {len(df)} candles {df.index[0]:%Y-%m-%d} → {df.index[-1]:%Y-%m-%d %H:%M}")
        self.redraw(autorange=True)

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
        self.vline.setPen(cross)
        self.hline.setPen(cross)

    def redraw(self, autorange: bool = False):
        t = self.themes.current
        for item in list(self.price.items):
            if item not in (self.vline, self.hline, *self.drawings):
                self.price.removeItem(item)
        for pane in self.panes:
            self.canvas.removeItem(pane)
        self.panes.clear()
        self._style(t)
        if self.df is None or self.df.empty:
            return
        df = self.df
        x = epoch_seconds(df.index)
        width = (np.median(np.diff(x)) if len(x) > 1 else 86400) * 0.7
        o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
        if self.style_btn.isChecked():
            self.price.plot(x, c, pen=pg.mkPen(qcolor(t.accent), width=2))
        else:
            candles = CandlestickItem()
            candles.set_data(x, o, h, l, c, width, qcolor(t.up), qcolor(t.down))
            self.price.addItem(candles)

        series_colors = iter(t.series * 4)
        legend_items = []
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
                target.setTitle(ind.label, color=t.muted, size="9pt")
                for level in ind.levels:
                    target.addItem(pg.InfiniteLine(level, angle=0, pen=pg.mkPen(qcolor(t.muted), style=Qt.PenStyle.DotLine)))
                self.panes.append(target)
            for col in frame.columns:
                if not ind.is_line(col):
                    self._draw_labels(target, ind, frame, col, x, c, t)
                    continue
                y = frame[col].to_numpy()
                ok = ~np.isnan(y)
                color = next(series_colors)
                if col.endswith("_hist"):
                    target.addItem(pg.BarGraphItem(x=x[ok], height=y[ok], width=width, brush=pg.mkBrush(qcolor(t.muted))))
                else:
                    target.plot(x[ok], y[ok], pen=pg.mkPen(color, width=1.6), name=col)
                legend_items.append((color, col))
        if self.algorithm is not None:
            change = np.diff(np.r_[0.0, self.algorithm.run(df).to_numpy()])
            for mask, symbol, color, anchor in ((change > 0, "t1", t.up, l * 0.995), (change < 0, "t", t.down, h * 1.005)):
                if mask.any():
                    self.price.addItem(pg.ScatterPlotItem(x=x[mask], y=anchor[mask], symbol=symbol, size=13,
                                                          brush=pg.mkBrush(color), pen=pg.mkPen(None)))
        self._style(t)
        self.info.setText("   ".join(f"<span style='color:{c}'>■</span> {n}" for c, n in legend_items))
        if autorange:
            self.reset_view()

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
        if self.df is None or not self.price.sceneBoundingRect().contains(pos):
            return
        pt = self.price.vb.mapSceneToView(pos)
        self.vline.setPos(pt.x())
        self.hline.setPos(pt.y())
        idx = epoch_seconds(self.df.index)
        i = int(np.clip(np.searchsorted(idx, pt.x()), 0, len(idx) - 1))
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
        for plot in [self.price, *self.panes]:
            plot.autoRange()  # autoRange() disables auto range in pyqtgraph, so re-enable afterwards
            plot.enableAutoRange()

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
