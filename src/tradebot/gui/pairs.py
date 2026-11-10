# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Pairs tab (#117): two assets normalised on one chart, their gap as a z-score, and the numbers behind it."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QLineEdit, QPushButton, QSpinBox, QVBoxLayout, QWidget

from tradebot.analysis.pairs import cointegration, pair_health, pair_stats, rolling_pair
from tradebot.core.models import split_pair
from tradebot.core.timeframes import TIMEFRAMES
from tradebot.gui.common import card, run_async
from tradebot.gui.theme import ThemeManager, qcolor


class PairView(QWidget):
    status = pyqtSignal(str)

    def __init__(self, engine, themes: ThemeManager, watchlist: list[str]):
        super().__init__()
        self.engine, self.themes = engine, themes
        self.symbol_a = QLineEdit(watchlist[0] if watchlist else "KO", placeholderText="first, e.g. KO")
        self.symbol_b = QLineEdit(watchlist[1] if len(watchlist) > 1 else "PEP", placeholderText="second, e.g. PEP")
        self.tf = QComboBox()
        self.tf.addItems([t for t in TIMEFRAMES if t not in ("1m", "5m", "15m", "30m", "1h", "4h")])  # #120
        self.tf.setCurrentText("1d")
        self.bars = QSpinBox(minimum=100, maximum=20000, value=750)
        self.window = QSpinBox(minimum=10, maximum=1000, value=60,
                               toolTip="Candles for the rolling hedge ratio and z-score: short follows the pair "
                                       "but absorbs the gap, long reacts slowly to a changed relationship")
        run = QPushButton("Analyse", objectName="Primary")
        run.clicked.connect(self.run)
        row = QHBoxLayout()
        for label, w in (("A", self.symbol_a), ("B", self.symbol_b), ("Timeframe", self.tf), ("Candles", self.bars),
                         ("Window", self.window)):
            row.addWidget(QLabel(label))
            row.addWidget(w)
        row.addWidget(run)
        self.prices = pg.PlotWidget(axisItems={"bottom": pg.DateAxisItem()})
        self.zplot = pg.PlotWidget(axisItems={"bottom": pg.DateAxisItem()})
        for plot in (self.prices, self.zplot):
            plot.showGrid(x=True, y=True, alpha=0.15)
        self.prices.addLegend()
        self.zplot.setXLink(self.prices)
        self.numbers = QLabel("Pick two symbols and press Analyse. Prices in different currencies are "
                              "converted into A's currency (daily and longer).")
        self.numbers.setWordWrap(True)
        self.numbers.setTextFormat(pg.QtCore.Qt.TextFormat.RichText)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 16)
        title = QLabel("Pairs")
        title.setObjectName("Title")
        lay.addWidget(title)
        lay.addWidget(card(_row(row)))
        lay.addWidget(self.numbers)
        lay.addWidget(self.prices, 3)
        lay.addWidget(self.zplot, 2)
        themes.changed.connect(self._style)
        self._style(themes.current)

    def _style(self, t):
        for plot in (self.prices, self.zplot):
            plot.setBackground(None if t.glass else qcolor(t.panel))
            for name in ("left", "bottom"):
                plot.getAxis(name).setTextPen(pg.mkPen(qcolor(t.muted)))
                plot.getAxis(name).setPen(pg.mkPen(qcolor(t.panel_border)))

    def run(self):
        a, b, tf = self.symbol_a.text().strip(), self.symbol_b.text().strip(), self.tf.currentText()
        if not a or not b:
            self.status.emit("pairs: enter both symbols")
            return
        bars, window = self.bars.value(), self.window.value()
        self.status.emit(f"pairs: loading {a} and {b} …")

        def work():
            ca, cb = (df["close"] for df in self.engine.pair_legs((a, b), tf, bars=bars))  # same currency
            both = ca.to_frame("a").join(cb.rename("b"), how="inner")
            return both["a"], both["b"]

        run_async(work, lambda r: self._show(a, b, window, *r), lambda err: self.status.emit(f"pairs failed: {err}"))

    def _show(self, a, b, window, ca, cb):
        t = self.themes.current
        x = ca.index.as_unit("ns").asi8 / 1e9
        self.prices.clear()
        self.zplot.clear()
        for series, name, color in ((ca, a, t.series[0]), (cb, b, t.series[1])):
            self.prices.plot(x, (series / series.iloc[0] * 100).to_numpy(), pen=pg.mkPen(color, width=2), name=name)
        roll = rolling_pair(ca, cb, window)
        z = roll.zscore.reindex(ca.index).to_numpy()
        self.zplot.plot(x, z, pen=pg.mkPen(qcolor(t.accent), width=2))
        for level, color in ((0, t.muted), (2, t.down), (-2, t.up)):
            self.zplot.addItem(pg.InfiniteLine(level, angle=0, pen=pg.mkPen(qcolor(color), width=1,
                                                                            style=pg.QtCore.Qt.PenStyle.DashLine)))
        text = []
        try:
            st = pair_stats(ca, cb)
            co = cointegration(ca, cb)
            text.append(f"correlation <b>{st.correlation:.2f}</b> · hedge ratio β <b>{st.beta:.2f}</b> (rolling "
                        f"{np.nanmean(roll.beta.to_numpy()[-window:]):.2f}) · gap now <b>{st.zscore:+.2f} σ</b> · "
                        f"half-life <b>{'does not revert' if st.half_life is None else f'{st.half_life:.1f} candles'}</b> · "
                        f"cointegration t = <b>{co.stat:.2f}</b>, {co.p_text}{' ✓' if co.passed else ''}")
            health = pair_health(ca, cb)
            if health.broken:
                text.append(f"<span style='color:{t.down}'><b>this pair may have stopped working:</b> "
                            + "; ".join(health.reasons) + "</span>")
        except ValueError as exc:
            text.append(str(exc))
        self.numbers.setText("<br>".join(text))
        self.status.emit(f"pairs: {a} vs {b}, {len(ca)} common candles. The chart symbol {a}:{b} shows the gap as candles")


def _row(layout) -> QWidget:
    w = QWidget()
    w.setLayout(layout)
    return w
