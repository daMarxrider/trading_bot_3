# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Pairs tab (#117): two assets normalised on one chart, their gap as a z-score, the chance that a trade on the gap
closes before its stop (#223), and the numbers behind it."""

from __future__ import annotations

from html import escape

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel, QLineEdit, QPushButton, QSpinBox,
                             QVBoxLayout, QWidget)

from tradebot.analysis.pairs import cointegration, pair_health, kalman_pair, pair_stats, rolling_pair
from tradebot.analysis.reversion import reversion
from tradebot.core.models import split_pair
from tradebot.core.timeframes import TIMEFRAMES
from tradebot.gui.common import SymbolCompleter, card, run_async
from tradebot.gui.theme import ThemeManager, qcolor
from tradebot.symbols import SymbolSearch


class PairView(QWidget):
    status = pyqtSignal(str)

    def __init__(self, engine, themes: ThemeManager, watchlist: list[str], symbol_search: SymbolSearch | None = None):
        super().__init__()
        self.engine, self.themes = engine, themes
        self.symbol_a = QLineEdit(watchlist[0] if watchlist else "KO", placeholderText="first, e.g. KO")
        self.symbol_b = QLineEdit(watchlist[1] if len(watchlist) > 1 else "PEP", placeholderText="second, e.g. PEP")
        # the chart's symbol search (#220): completion while typing, the symbol dialog on a click or the search icon
        search = symbol_search or SymbolSearch(lambda: watchlist)
        self.completers = (SymbolCompleter(self.symbol_a, search), SymbolCompleter(self.symbol_b, search))
        for completer in self.completers:
            completer.chosen.connect(self._chosen)
        self.tf = QComboBox()
        self.tf.addItems(list(TIMEFRAMES))
        self.tf.setCurrentText("1d")
        self.bars = QSpinBox(minimum=100, maximum=20000, value=750)
        self.window_box = QSpinBox(minimum=10, maximum=1000, value=60,
                                   toolTip="Candles for the rolling hedge ratio and z-score: short follows the pair "
                                           "but absorbs the gap, long reacts slowly to a changed relationship")
        self.method = QComboBox(toolTip="Rolling: OLS over the window. Kalman: an adaptive filter, no window, "
                                        "the hedge ratio follows the pair smoothly")
        self.method.addItems(["rolling", "kalman"])
        run = QPushButton("Analyse", objectName="Primary")
        run.clicked.connect(self.run)
        for field in (self.symbol_a, self.symbol_b):
            field.setMinimumWidth(150)  # a long symbol (BTC/USDT:USDT) next to the search icon
        # what to load on the first row, how to read it on the second: each fits a narrow window
        row = QHBoxLayout()
        for label, w in (("A", self.symbol_a), ("B", self.symbol_b), ("Timeframe", self.tf), ("Candles", self.bars)):
            row.addWidget(QLabel(label))
            row.addWidget(w, 1 if w in (self.symbol_a, self.symbol_b) else 0)
        row.addWidget(run)
        # the trade the chance is about: closed inside ±exit, given up beyond ±stop, within the horizon (#223)
        self.exit = QDoubleSpinBox(minimum=0.0, maximum=10.0, singleStep=0.1, value=0.5, suffix=" σ",
                                   toolTip="The trade closes when the gap is back inside this many σ")
        self.stop = QDoubleSpinBox(minimum=0.1, maximum=20.0, singleStep=0.5, value=4.0, suffix=" σ",
                                   toolTip="The trade gives up when the gap widens beyond this many σ")
        self.horizon = QSpinBox(minimum=1, maximum=5000, value=30, suffix=" candles",
                                toolTip="How long the gap has to close in. The lowest chart shows the chance that a "
                                        "trade opened at each candle closes to the exit before the stop within this "
                                        "time, from how fast the gap reverted over the last 2 × window candles")
        trade = QHBoxLayout()
        for label, w in (("Window", self.window_box), ("Hedge ratio", self.method), ("Exit", self.exit),
                         ("Stop", self.stop), ("Within", self.horizon)):
            trade.addWidget(QLabel(label))
            trade.addWidget(w)
        trade.addStretch(1)
        self.prices = pg.PlotWidget(axisItems={"bottom": pg.DateAxisItem()})
        self.zplot = pg.PlotWidget(axisItems={"bottom": pg.DateAxisItem()})
        self.chance = pg.PlotWidget(axisItems={"bottom": pg.DateAxisItem()})
        for plot in (self.prices, self.zplot, self.chance):
            plot.showGrid(x=True, y=True, alpha=0.15)
        self.prices.addLegend()
        self.zplot.setXLink(self.prices)
        self.chance.setXLink(self.prices)
        # always the whole 0-100 %, with room for the 0 and 100 labels: labels that don't fit inside are not drawn
        self.chance.enableAutoRange(axis="y", enable=False)
        self.chance.setMouseEnabled(y=False)
        self.chance.setYRange(-20, 120, padding=0)
        self.chance.setToolTip("Chance in % that a trade opened at this candle closes to the exit before the stop "
                               "within the horizon")
        for plot in (self.prices, self.zplot, self.chance):
            plot.getAxis("left").setWidth(44)  # one width: the three time axes line up
        self.chance.getAxis("left").setTicks([[(0, "0"), (50, "50"), (100, "100")]])  # readable at any height
        self.chance.addLegend()
        self.numbers = QLabel("Pick two symbols and press Analyse. Prices in different currencies are "
                              "converted into A's currency (intraday: only candles with a fresh exchange rate).")
        self.numbers.setWordWrap(True)
        self.numbers.setTextFormat(pg.QtCore.Qt.TextFormat.RichText)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 16)
        title = QLabel("Pairs")
        title.setObjectName("Title")
        lay.addWidget(title)
        lay.addWidget(card(_row(row), _row(trade)))
        lay.addWidget(self.numbers)
        lay.addWidget(self.prices, 3)
        lay.addWidget(self.zplot, 2)
        lay.addWidget(self.chance, 1)
        themes.changed.connect(self._style)
        self._style(themes.current)

    def _chosen(self, symbol: str):
        """A ratio picked in the dialog (its ÷ button returns ``A:B``) is a whole pair: it fills both fields."""
        legs = split_pair(symbol)
        if legs is not None:
            self.symbol_a.setText(legs[0])
            self.symbol_b.setText(legs[1])

    def _style(self, t):
        for plot in (self.prices, self.zplot, self.chance):
            plot.setBackground(None if t.glass else qcolor(t.panel))
            for name in ("left", "bottom"):
                plot.getAxis(name).setTextPen(pg.mkPen(qcolor(t.muted)))
                plot.getAxis(name).setPen(pg.mkPen(qcolor(t.panel_border)))

    def run(self):
        a, b, tf = self.symbol_a.text().strip(), self.symbol_b.text().strip(), self.tf.currentText()
        if not a or not b:
            self.status.emit("pairs: enter both symbols")
            return
        bars, window, method = self.bars.value(), self.window_box.value(), self.method.currentText()
        self.status.emit(f"pairs: loading {a} and {b} …")

        def work():
            ca, cb = (df["close"] for df in self.engine.pair_legs((a, b), tf, bars=bars))
            both = ca.to_frame("a").join(cb.rename("b"), how="inner")
            return both["a"], both["b"]

        run_async(work, lambda r: self._show(a, b, window, method, *r), lambda err: self.status.emit(f"pairs failed: {err}"))

    def _show(self, a, b, window, method, ca, cb):
        t = self.themes.current
        x = ca.index.as_unit("ns").asi8 / 1e9
        self.prices.clear()
        self.zplot.clear()
        self.chance.clear()
        for series, name, color in ((ca, a, t.series[0]), (cb, b, t.series[1])):
            self.prices.plot(x, (series / series.iloc[0] * 100).to_numpy(), pen=pg.mkPen(color, width=2), name=name)
        roll = kalman_pair(ca, cb) if method == "kalman" else rolling_pair(ca, cb, window)
        z = roll.zscore.reindex(ca.index).to_numpy()
        self.zplot.plot(x, z, pen=pg.mkPen(qcolor(t.accent), width=2))
        exit, stop, horizon = self.exit.value(), self.stop.value(), self.horizon.value()
        style = pg.QtCore.Qt.PenStyle
        levels = [(0, t.muted, style.DashLine), (2, t.down, style.DashLine), (-2, t.up, style.DashLine)]  # the entries
        levels += [(v, t.muted, style.DotLine) for v in (exit, -exit)] \
            + [(v, t.muted, style.DashDotLine) for v in (stop, -stop)]
        for level, color, style in levels:
            self.zplot.addItem(pg.InfiniteLine(level, angle=0, pen=pg.mkPen(qcolor(color), width=1, style=style)))
        chance_text = []
        if exit < stop:
            rev = reversion(roll.zscore, horizon, exit, stop, 2 * window).reindex(ca.index)
            chance = rev["probability"].to_numpy() * 100
            self.chance.plot(x, chance, pen=pg.mkPen(qcolor(t.accent), width=2), connect="finite",
                             name="chance to close before the stop, %")
            last = rev.iloc[-1]
            if last.half_life == last.half_life:  # not NaN
                gap = roll.zscore.iloc[-1]
                now = ("the gap is inside the exit band" if abs(gap) <= exit else
                       f"a trade now closes to ±{exit:g} σ before ±{stop:g} σ within {horizon} candles with "
                       f"<b>{last.probability:.0%}</b> chance")
                chance_text.append(f"half-life now <b>{last.half_life:.1f} candles</b> (last {2 * window}) · {now}")
            else:
                chance_text.append(f"chance to close: unknown, no reversion seen over the last {2 * window} "
                                   f"candles (or fewer than {3 * window + 2} candles)")
        else:
            chance_text.append("the exit must be inside the stop: lower Exit or raise Stop")
        text = []
        try:
            st = pair_stats(ca, cb)
            co = cointegration(ca, cb)
            text.append(f"correlation <b>{st.correlation:.2f}</b> · hedge ratio β <b>{st.beta:.2f}</b> ({method} "
                        f"{np.nanmean(roll.beta.to_numpy()[-(1 if method == 'kalman' else window):]):.2f}) · gap now <b>{st.zscore:+.2f} σ</b> · "
                        f"half-life <b>{'does not revert' if st.half_life is None else f'{st.half_life:.1f} candles'}</b> · "
                        f"cointegration t = <b>{co.stat:.2f}</b>, {escape(co.p_text)}{' ✓' if co.passed else ''}")
            health = pair_health(ca, cb)
            if health.broken:
                text.append(f"<span style='color:{t.down}'><b>this pair may have stopped working:</b> "
                            + escape("; ".join(health.reasons)) + "</span>")
        except ValueError as exc:
            text.append(escape(str(exc)))
        self.numbers.setText("<br>".join(text + chance_text))
        self.status.emit(f"pairs: {a} vs {b}, {len(ca)} common candles. The chart symbol {a}:{b} shows the gap as candles")


def _row(layout) -> QWidget:
    w = QWidget()
    w.setLayout(layout)
    return w
