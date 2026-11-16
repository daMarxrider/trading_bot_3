# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Portfolio backtest tab (#148): one shared account trading a strategy on several symbols, next to buy & hold
of the same basket."""

from __future__ import annotations

import pandas as pd
import pyqtgraph as pg
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from tradebot.algorithms import algorithm_names, build_algorithm
from tradebot.backtest import metrics as m
from tradebot.backtest.portfolio import ALLOCATIONS, BASKET, REBALANCE, portfolio_backtest
from tradebot.core.timeframes import TIMEFRAMES
from tradebot.gui.common import DataFrameModel, card, run_async
from tradebot.gui.theme import ThemeManager, qcolor


class PortfolioBacktestView(QWidget):
    status = pyqtSignal(str)

    def __init__(self, engine, themes: ThemeManager, watchlist: list[str]):
        super().__init__()
        self.engine, self.themes = engine, themes
        self.result = None
        self.symbols = QLineEdit(", ".join(watchlist[:5]) or "AAPL, MSFT, KO, JNJ",
                                 placeholderText="symbols, comma-separated")
        self.algo = QComboBox()
        self.algo.addItems([n for n in algorithm_names() if n not in ("vote", "pool", "combo")])
        if self.algo.findText("sma_cross") >= 0:
            self.algo.setCurrentText("sma_cross")
        self.tf = QComboBox()
        self.tf.addItems(list(TIMEFRAMES))
        self.tf.setCurrentText("1d")
        self.bars = QSpinBox(minimum=100, maximum=20000, value=1000)
        self.max_positions = QSpinBox(minimum=0, maximum=100, value=0, specialValueText="all",
                                      toolTip="Most positions held at once; each slot is that share of the account, "
                                              "empty slots stay in cash. 'all' = one slot per symbol")
        self.allocation = QComboBox(toolTip="\n".join(f"{k}: {v}" for k, v in ALLOCATIONS.items()))
        self.allocation.addItems(list(ALLOCATIONS))
        self.rebalance = QComboBox(toolTip="Set the held positions back to their target weights. "
                                           + "; ".join(f"{k}: {v}" for k, v in REBALANCE.items()))
        self.rebalance.addItems(list(REBALANCE))
        self.rebalance.setCurrentText("month")
        self.fee = QDoubleSpinBox(minimum=0.0, maximum=2.0, value=0.1, decimals=3, singleStep=0.05, suffix=" %",
                                  toolTip="Fee per side")
        self.capital = QDoubleSpinBox(minimum=100.0, maximum=1e9, value=10_000.0, decimals=0, singleStep=1000.0)
        run = QPushButton("Run", objectName="Primary")
        run.clicked.connect(self.run)
        grid = QGridLayout()
        grid.addWidget(QLabel("Symbols"), 0, 0)
        grid.addWidget(self.symbols, 0, 1, 1, 9)
        for col, (label, w) in enumerate((("Strategy", self.algo), ("Timeframe", self.tf), ("Candles", self.bars),
                                          ("Max positions", self.max_positions), ("Allocation", self.allocation),
                                          ("Rebalance", self.rebalance), ("Fee", self.fee),
                                          ("Capital", self.capital))):
            grid.addWidget(QLabel(label), 1 + col // 4, (col % 4) * 2)
            grid.addWidget(w, 1 + col // 4, (col % 4) * 2 + 1)
        grid.addWidget(run, 2, 9)
        box = QWidget()
        box.setLayout(grid)

        self.plot = pg.PlotWidget(axisItems={"bottom": pg.DateAxisItem()})
        self.plot.showGrid(x=True, y=True, alpha=0.15)
        self.plot.addLegend()
        pct = lambda v: m.fmt("exposure", v)  # noqa: E731
        self.metrics_model = DataFrameModel()
        self.symbols_model = DataFrameModel(formats={"P&L": lambda v: f"{v:+,.2f}", "time held": pct,
                                                     "avg weight": pct, "trades": lambda v: str(int(v))},
                                            colors=self._pnl_color)
        self.metrics_table, self.symbols_table = QTableView(), QTableView()
        for table, model in ((self.metrics_table, self.metrics_model), (self.symbols_table, self.symbols_model)):
            table.setModel(model)
            table.verticalHeader().hide()
            table.horizontalHeader().setStretchLastSection(True)
        self.symbols_table.setSortingEnabled(True)
        self.summary = QLabel("Pick the symbols and a strategy and press Run. The account trades all of them at "
                              "once, next to buy & hold of the same basket.")
        self.summary.setWordWrap(True)
        self.summary.setTextFormat(pg.QtCore.Qt.TextFormat.RichText)

        tables = QSplitter()
        tables.addWidget(self.metrics_table)
        tables.addWidget(self.symbols_table)
        body = QSplitter(pg.QtCore.Qt.Orientation.Vertical)
        body.addWidget(self.plot)
        body.addWidget(tables)
        body.setStretchFactor(0, 3)
        body.setStretchFactor(1, 2)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 16)
        title = QLabel("Portfolio backtest")
        title.setObjectName("Title")
        lay.addWidget(title)
        lay.addWidget(card(box))
        lay.addWidget(self.summary)
        lay.addWidget(body, 1)
        themes.changed.connect(self._style)
        self._style(themes.current)

    def _style(self, t):
        self.plot.setBackground(None if t.glass else qcolor(t.panel))
        for name in ("left", "bottom"):
            self.plot.getAxis(name).setTextPen(pg.mkPen(qcolor(t.muted)))
            self.plot.getAxis(name).setPen(pg.mkPen(qcolor(t.panel_border)))

    def _pnl_color(self, col, value):
        if col != "P&L" or not isinstance(value, (int, float)):
            return None
        t = self.themes.current
        return qcolor(t.up if value > 0 else t.down)

    def settings(self) -> dict:
        return {"max_positions": self.max_positions.value() or None, "allocation": self.allocation.currentText(),
                "rebalance": self.rebalance.currentText(), "fee": self.fee.value() / 100,
                "initial_capital": self.capital.value()}

    def run(self):
        symbols = list(dict.fromkeys(s.strip() for s in self.symbols.text().split(",") if s.strip()))
        if not symbols:
            self.status.emit("portfolio backtest: enter some symbols")
            return
        tf, bars, algo, kw = self.tf.currentText(), self.bars.value(), self.algo.currentText(), self.settings()
        self.status.emit(f"portfolio backtest: loading {len(symbols)} symbols …")

        def work():
            frames, skipped = {}, []
            for s in symbols:
                try:
                    frames[s] = self.engine.get_ohlc(s, tf, bars=bars)
                except Exception:  # unknown symbol, offline without cache
                    skipped.append(s)
            if not frames:
                raise ValueError("none of the symbols could be loaded")
            return portfolio_backtest(frames, build_algorithm(algo, {}), **kw), skipped

        run_async(work, lambda r: self.show_result(*r),
                  lambda err: self.status.emit(f"portfolio backtest failed: {err}"))

    def show_result(self, pf, skipped=()):
        self.result = pf
        t = self.themes.current
        res, bench = pf.result, pf.benchmark
        x = res.equity.index.as_unit("ns").asi8 / 1e9
        self.plot.clear()
        self.plot.plot(x, res.equity.to_numpy(), pen=pg.mkPen(qcolor(t.accent), width=2), name="portfolio")
        self.plot.plot(x, bench.equity.to_numpy(), pen=pg.mkPen(qcolor(t.muted), width=1.5), name=BASKET)
        cols = pf.columns()
        keys = list(dict.fromkeys(k for c in cols.values() for k in c))
        self.metrics_model.set_frame(pd.DataFrame(
            {"metric": [k.replace("_", " ") for k in keys],
             **{name: [m.fmt(k, c[k]) if k in c else "–" for k in keys] for name, c in cols.items()}}))
        per = pf.per_symbol().rename(columns={"contribution": "P&L"}).rename_axis("symbol").reset_index()
        self.symbols_model.set_frame(per)
        for table in (self.metrics_table, self.symbols_table):
            table.resizeColumnsToContents()
        s = pf.settings
        lead = pf.excess
        color = t.up if lead > 0 else t.down
        text = (f"{res.equity.index[0]:%Y-%m-%d} → {res.equity.index[-1]:%Y-%m-%d} · max {s['max_positions']} "
                f"positions · {s['allocation']} allocation · rebalance {s['rebalance']} · final equity "
                f"<b>{res.equity.iloc[-1]:,.2f}</b> (basket {bench.equity.iloc[-1]:,.2f}) · "
                f"<span style='color:{color}'>{m.fmt('vs buy & hold', lead)} XIRR vs the basket</span>")
        if skipped:
            text += f"<br>skipped (could not be loaded): {', '.join(skipped)}"
        self.summary.setText(text)
        self.status.emit(f"portfolio backtest: {len(pf.weights.columns)} symbols, {len(res.trades)} trades")
