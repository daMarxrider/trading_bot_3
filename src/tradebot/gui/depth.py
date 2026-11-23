# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Order-book depth of the charted crypto pair, in the chart's sidebar (#166).

A depth chart: the cumulative amount that can be bought (asks, right) or sold (bids, left) as the price moves
away from the middle, as two step areas, with the share of bids in the book near the price as a figure. Fetched
only while the panel is visible, refreshed every few seconds, and a failure (offline, rate limited) is said in
words instead of leaving a stale picture.
"""

from __future__ import annotations

import html

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QStackedWidget, QToolButton, QVBoxLayout, QWidget

from tradebot.data.market import OrderBook, is_crypto
from tradebot.gui.common import run_async
from tradebot.gui.theme import ThemeManager, qcolor

REFRESH_MS = 15_000
RANGES = {"±1 %": 0.01, "±2 %": 0.02, "±5 %": 0.05, "±10 %": 0.10}  #: how far from the mid price the chart reaches


def imbalance_text(value: float, within_label: str) -> tuple[str, str]:
    """``(text, side)`` for the book's imbalance: a triangle and a word, so colour is never the only cue.
    ``side`` is ``"up"`` (more bids), ``"down"`` (more asks) or ``"flat"``."""
    if abs(value) < 0.05:
        return f"balanced within {within_label}", "flat"
    if value > 0:
        return f"▲ bids {value * 100:.0f} % heavier within {within_label}", "up"
    return f"▼ asks {-value * 100:.0f} % heavier within {within_label}", "down"


def outward(prices: np.ndarray, cum: np.ndarray, edge: float) -> tuple[np.ndarray, np.ndarray]:
    """A step line from the best price outward: the cumulative amount ``cum[k]`` holds from ``prices[k]`` to the next
    price (the last one to ``edge``), starting at 0 at the best price."""
    nxt = np.r_[prices[1:], edge]
    x = np.empty(2 * len(prices) + 1)
    y = np.empty_like(x)
    x[0], y[0] = prices[0], 0.0
    x[1::2], y[1::2] = prices, cum
    x[2::2], y[2::2] = nxt, cum
    return x, y


def _amount(x: float) -> str:
    return f"{x:,.0f}" if x >= 100 else f"{x:,.2f}"


class DepthPanel(QWidget):
    """The chart sidebar's order-book page."""

    def __init__(self, engine, themes: ThemeManager, parent=None):
        super().__init__(parent)
        self.engine, self.themes = engine, themes
        self.symbol = ""
        self.book: OrderBook | None = None
        self._stats = ""
        self._visible = False
        self._token = 0  # a result for an older request or another symbol is dropped
        self.header = QLabel(objectName="Muted", wordWrap=True)
        self.range = QComboBox(toolTip="How far from the middle price the chart reaches")
        self.range.addItems(RANGES)
        self.range.setCurrentText("±2 %")
        self.range.currentIndexChanged.connect(lambda _: self._render())
        self.refresh_btn = QToolButton(text="⟳", toolTip="Refresh the order book")
        self.refresh_btn.clicked.connect(self.refresh)
        self.plot = pg.PlotWidget()
        self.plot.setMenuEnabled(False)
        self.plot.setMouseEnabled(False, False)
        self.plot.hideButtons()
        self.plot.getPlotItem().showGrid(x=False, y=True, alpha=0.12)
        self.plot.setMinimumHeight(180)
        self.stats = QLabel(objectName="Muted", wordWrap=True)
        self.stats.setTextFormat(Qt.TextFormat.RichText)
        self.message = QLabel(objectName="Muted", wordWrap=True)
        self.message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.body = QStackedWidget()
        self.body.addWidget(self.message)
        page = QWidget()
        col = QVBoxLayout(page)
        col.setContentsMargins(0, 0, 0, 0)
        col.addWidget(self.plot, 1)
        col.addWidget(self.stats)
        self.body.addWidget(page)
        self._page = page
        head = QHBoxLayout()
        head.addWidget(self.header, 1)
        head.addWidget(self.range)
        head.addWidget(self.refresh_btn)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addLayout(head)
        lay.addWidget(self.body, 1)
        self.timer = QTimer(self, interval=REFRESH_MS, timeout=self.refresh)
        themes.changed.connect(lambda _: self._render())
        self._show_message("Load a crypto chart to see its order book.")

    # -- following the chart ---------------------------------------------------------------------
    def set_symbol(self, symbol: str):
        self.header.setText(html.escape(symbol))
        if symbol == self.symbol:
            return
        self.symbol, self.book = symbol, None
        self._token += 1
        if not is_crypto(symbol):
            self.timer.stop()
            self._show_message("Order-book depth is available for crypto pairs such as BTC/USDT.")
        elif self._visible:
            self.refresh()
        else:
            self._show_message("Open this page to load the order book.")

    def set_visible(self, on: bool):
        """Whether the sidebar shows this page: the book is only fetched (and refreshed) while it does."""
        self._visible = on
        if not on:
            self.timer.stop()
            return
        if is_crypto(self.symbol):
            self.timer.start()
            if self.book is None:
                self.refresh()

    def refresh(self):
        symbol = self.symbol
        if not is_crypto(symbol):
            return
        if self.book is None:
            self._show_message(f"Loading the order book of {symbol} …")
        self._token += 1
        token, fetch = self._token, self.engine.market.order_book  # bound now, not when the worker gets to it
        run_async(lambda: fetch(symbol, 500), lambda book: self._loaded(token, symbol, book),
                  lambda err: self._failed(token, symbol, err))

    def _loaded(self, token, symbol, book):
        if token != self._token or symbol != self.symbol:
            return
        self.book = book
        self._render()

    def _failed(self, token, symbol, err):
        if token != self._token or symbol != self.symbol:
            return
        what = f"The order book of {symbol} is unavailable right now (offline or rate limited).\n{err}"
        if self.book is None:
            self._show_message(what)
        else:
            self.stats.setText(f"<span style='color:{self.themes.current.warn}'>Couldn't refresh: {html.escape(str(err))}"
                               "</span><br>" + self._stats)

    # -- drawing ---------------------------------------------------------------------------------
    def _render(self):
        book = self.book
        if book is None:
            return
        t = self.themes.current
        within = RANGES[self.range.currentText()]
        bp, bc, ap, ac = book.depth(within)
        bp, bc = bp[::-1], bc[::-1]  # from the best bid outward
        if not len(bp) or not len(ap):
            self._show_message("The order book is too thin for this range; pick a wider one.")
            return
        mid = book.mid
        self.body.setCurrentWidget(self._page)
        plot = self.plot
        plot.clear()
        plot.setBackground(None)
        item = plot.getPlotItem()
        for axis in ("bottom", "left"):
            ax = item.getAxis(axis)
            ax.setPen(pg.mkPen(qcolor(t.panel_border)))
            ax.setTextPen(pg.mkPen(qcolor(t.muted)))
        # step areas that start at the best price and end at the edge of the range
        top = float(max(bc.max(), ac.max()))
        for (x, y), color in ((outward(bp, bc, mid * (1 - within)), t.up),
                              (outward(ap, ac, mid * (1 + within)), t.down)):
            fill = qcolor(color)
            fill.setAlpha(70)
            curve = pg.PlotDataItem(x, y, pen=pg.mkPen(qcolor(color), width=1.6), fillLevel=0,
                                    brush=pg.mkBrush(fill))
            plot.addItem(curve)
        plot.addItem(pg.InfiniteLine(mid, angle=90, pen=pg.mkPen(qcolor(t.muted), width=1,
                                                                  style=Qt.PenStyle.DotLine)))
        for text, x, anchor, color in (("bids", mid * (1 - within * 0.97), (0, 0), t.up),
                                       ("asks", mid * (1 + within * 0.97), (1, 0), t.down)):
            label = pg.TextItem(text, color=qcolor(color), anchor=anchor)
            label.setPos(x, top * 1.0)
            plot.addItem(label)
        plot.setXRange(mid * (1 - within), mid * (1 + within), padding=0.01)
        plot.setYRange(0, top * 1.12, padding=0)
        label = self.range.currentText()
        text, side = imbalance_text(book.imbalance(within), label)
        color = {"up": t.up, "down": t.down}.get(side, t.muted)
        self._stats = (
            f"Mid <b>{mid:,.2f}</b> · spread <b>{book.spread * 100:.3f} %</b><br>"
            f"Bids {_amount(bc[-1])} · asks {_amount(ac[-1])} within {html.escape(label)}<br>"
            f"<span style='color:{color}'>{html.escape(text)}</span>")
        self.stats.setText(self._stats)

    def _show_message(self, text: str):
        self.message.setText(text)
        self.body.setCurrentWidget(self.message)
