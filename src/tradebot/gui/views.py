# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Screener, news, portfolio and backtest-lab pages."""

from __future__ import annotations

import html
import math

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt6.QtCore import QRectF, QSize, Qt, QTimer, QUrl, pyqtSignal
from PyQt6.QtGui import QColor, QDesktopServices, QPainter, QPen
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from tradebot.algorithms import ALGORITHMS, build_algorithm
from tradebot.backtest.metrics import fmt
from tradebot.brokers import BROKERS, create_broker
from tradebot.core.timeframes import TIMEFRAMES
from tradebot.gui.common import (
    DataFrameModel,
    DateRange,
    Kpi,
    SymbolCompleter,
    card,
    epoch_seconds,
    pct,
    run_async,
    shortfall,
)
from tradebot.gui.params import ParamForm
from tradebot.gui.theme import ThemeManager, qcolor
from tradebot.gui.touch import PlotTouchHandler, TouchController, enable_kinetic_scroll
from tradebot.symbols import SymbolSearch


def _title(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("Title")
    return label


def _table(model) -> QTableView:
    view = QTableView()
    view.setModel(model)
    view.setSortingEnabled(True)
    view.setAlternatingRowColors(True)
    view.setShowGrid(False)
    view.verticalHeader().hide()
    view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    enable_kinetic_scroll(view)
    return view


def _signed_color(themes: ThemeManager, cols: set[str]):
    def color(col, value):
        if col in cols and isinstance(value, (int, float)) and not (isinstance(value, float) and math.isnan(value)):
            return qcolor(themes.current.up if value >= 0 else themes.current.down)
        return None

    return color


# -- screener -----------------------------------------------------------------------------------------
class ScreenerView(QWidget):
    open_symbol = pyqtSignal(str)
    status = pyqtSignal(str)

    PRESETS = {
        "Custom": "",
        "Oversold (RSI < 30)": "rsi < 30",
        "Overbought (RSI > 70)": "rsi > 70",
        "Uptrend (close > SMA50 > SMA200)": "close > sma50; sma50 > sma200",
        "Golden zone pullback": "close > sma200; rsi < 45",
        "Momentum leaders (+10% 1M)": "change_1m > 0.10",
        "Large caps (> $100B)": "market_cap > 1e11",
        "High volatility (ATR > 4%)": "atr_pct > 0.04",
    }

    def __init__(self, engine, themes: ThemeManager, watchlist: list[str]):
        super().__init__()
        self.engine = engine
        self.symbols = QPlainTextEdit(", ".join(watchlist))
        self.symbols.setMaximumHeight(70)
        self.preset = QComboBox()
        self.preset.addItems(list(self.PRESETS))
        self.preset.currentTextChanged.connect(lambda k: self.filters.setText(self.PRESETS[k]) if self.PRESETS[k] else None)
        self.filters = QLineEdit()
        self.filters.setPlaceholderText("filters separated by ';'  e.g.  rsi < 35; close > sma200; market_cap > 1e10")
        run = QPushButton("Run screen")
        run.setObjectName("Primary")
        run.clicked.connect(self.run)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.hide()
        self.model = DataFrameModel(formats={
            "change_1d": pct, "change_1m": pct, "change_1y": pct, "atr_pct": pct,
            "market_cap": lambda v: f"{v / 1e9:,.1f}B", "avg_volume": lambda v: f"{v:,.0f}",
        }, colors=_signed_color(themes, {"change_1d", "change_1m", "change_1y"}))
        table = _table(self.model)
        table.doubleClicked.connect(lambda idx: self.open_symbol.emit(self.model.df.iat[idx.row(), 0]))
        hint = QLabel("Columns: close, change_1d/1m/1y, rsi, sma50, sma200, atr_pct, avg_volume, market_cap · "
                      "double-click a row to chart it")
        hint.setObjectName("Muted")

        top = QGridLayout()
        top.addWidget(QLabel("Universe"), 0, 0)
        top.addWidget(self.symbols, 0, 1, 1, 3)
        top.addWidget(QLabel("Preset"), 1, 0)
        top.addWidget(self.preset, 1, 1)
        top.addWidget(self.filters, 1, 2)
        top.addWidget(run, 1, 3)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.addWidget(_title("Screener"))
        box = card()
        box.layout().addLayout(top)
        lay.addWidget(box)
        lay.addWidget(self.progress)
        lay.addWidget(table, 1)
        lay.addWidget(hint)

    def run(self):
        from tradebot.screener import run_screen

        symbols = [s.strip() for s in self.symbols.toPlainText().replace("\n", ",").split(",") if s.strip()]
        filters = [f.strip() for f in self.filters.text().split(";") if f.strip()]
        start = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=420)
        self.progress.show()
        self.status.emit(f"screening {len(symbols)} symbols …")

        def done(table):
            self.progress.hide()
            self.model.set_frame(table)
            self.status.emit(f"screener: {len(table)} of {len(symbols)} match")

        def failed(err):
            self.progress.hide()
            self.status.emit(f"screener failed: {err}")

        run_async(lambda: run_screen(symbols, lambda s: self.engine.get_ohlc(s, "1d", start=start), filters), done, failed)


# -- news -------------------------------------------------------------------------------------------------
class NewsView(QWidget):
    status = pyqtSignal(str)
    REFRESH_MS = 5 * 60 * 1000

    def __init__(self, themes: ThemeManager, symbols_fn):
        super().__init__()
        self.themes, self.symbols_fn = themes, symbols_fn
        self.list = QListWidget()
        self.list.setWordWrap(True)
        self.list.setSpacing(4)
        enable_kinetic_scroll(self.list)
        self.list.itemActivated.connect(lambda it: QDesktopServices.openUrl(QUrl(it.data(Qt.ItemDataRole.UserRole))))
        self.filter = QLineEdit()
        self.filter.setPlaceholderText("filter headlines …")
        self.filter.textChanged.connect(self._render)
        refresh = QPushButton("Refresh")
        refresh.setObjectName("Primary")
        refresh.clicked.connect(self.refresh)
        self.auto = QCheckBox("auto-refresh every 5 min")
        self.auto.setChecked(True)
        self.timer = QTimer(self, interval=self.REFRESH_MS, timeout=self.refresh)
        self.auto.toggled.connect(lambda on: self.timer.start() if on else self.timer.stop())
        self.items = []
        head = QHBoxLayout()
        head.addWidget(_title("News"))
        head.addStretch()
        head.addWidget(self.filter)
        head.addWidget(self.auto)
        head.addWidget(refresh)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.addLayout(head)
        lay.addWidget(self.list, 1)
        hint = QLabel("Watchlist and portfolio symbols · double-click a headline to open it in the browser")
        hint.setObjectName("Muted")
        lay.addWidget(hint)
        themes.changed.connect(lambda _: self._render())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._render()  # re-wrap headlines to the new width

    def showEvent(self, event):
        super().showEvent(event)
        if not self.items:
            self.refresh()
            self.timer.start()

    def refresh(self):
        from tradebot.news import fetch_news

        symbols = self.symbols_fn()
        self.status.emit(f"fetching news for {len(symbols)} symbols …")
        run_async(lambda: fetch_news(symbols, 100), self._loaded, lambda e: self.status.emit(f"news failed: {e}"))

    def _loaded(self, items):
        self.items = items
        self._render()
        self.status.emit(f"{len(items)} headlines")

    def _render(self):
        t = self.themes.current
        needle = self.filter.text().lower()
        self.list.clear()
        for n in self.items:
            if needle and needle not in (n.title + n.summary + n.symbol).lower():
                continue
            label = QLabel(
                f"<span style='color:{t.accent}; font-weight:600'>{html.escape(n.symbol)}</span>"
                f"&nbsp;&nbsp;<span style='color:{t.muted}'>{n.published:%Y-%m-%d %H:%M} · {html.escape(n.publisher)}</span>"
                f"<br><span style='font-size:15px; font-weight:600'>{html.escape(n.title)}</span>"
                + (f"<br><span style='color:{t.muted}'>{html.escape(n.summary[:240])}</span>" if n.summary else "")
            )
            label.setWordWrap(True)
            label.setContentsMargins(10, 8, 10, 8)
            label.setStyleSheet("background: transparent;")
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, n.url)
            width = max(200, self.list.viewport().width() - 12)
            item.setSizeHint(QSize(width, label.heightForWidth(width) + 4))
            self.list.addItem(item)
            self.list.setItemWidget(item, label)


# -- portfolio --------------------------------------------------------------------------------------------
class DonutChart(QWidget):
    """Asset-allocation donut with legend."""

    def __init__(self, themes: ThemeManager):
        super().__init__()
        self.themes = themes
        self.weights = pd.Series(dtype=float)
        self.setMinimumSize(280, 240)
        themes.changed.connect(lambda _: self.update())

    def set_weights(self, weights: pd.Series, top: int = 7):
        if len(weights) > top:
            weights = pd.concat([weights.iloc[:top], pd.Series({"other": weights.iloc[top:].sum()})])
        self.weights = weights
        self.update()

    def paintEvent(self, event):
        if self.weights.empty:
            return
        t = self.themes.current
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        side = min(self.height() - 20, self.width() * 0.5)
        rect = QRectF(10, (self.height() - side) / 2, side, side)
        start = 90 * 16
        colors = [QColor(c) for c in t.series * 3]
        for i, w in enumerate(self.weights):
            span = -int(round(w * 360 * 16))
            pen = QPen(colors[i], side * 0.16)
            pen.setCapStyle(Qt.PenCapStyle.FlatCap)
            p.setPen(pen)
            inset = side * 0.08
            p.drawArc(rect.adjusted(inset, inset, -inset, -inset), start, span)
            start += span
        p.setPen(qcolor(t.text))
        x0, y = rect.right() + 24, rect.top() + 10
        for i, (name, w) in enumerate(self.weights.items()):
            p.fillRect(QRectF(x0, y + 3, 10, 10), colors[i])
            p.drawText(int(x0 + 18), int(y + 13), f"{name}  {w * 100:.1f}%")
            y += 22
        p.end()


class PortfolioView(QWidget):
    open_symbol = pyqtSignal(str)
    status = pyqtSignal(str)
    loaded = pyqtSignal(list)

    def __init__(self, engine, themes: ThemeManager):
        super().__init__()
        self.engine, self.themes = engine, themes
        self.checks = {name: QCheckBox(name) for name in BROKERS.names()}
        self.checks.get("scalable", next(iter(self.checks.values()))).setChecked(True)
        load = QPushButton("Load portfolio")
        load.setObjectName("Primary")
        load.clicked.connect(self.load)
        brokers = QHBoxLayout()
        brokers.addWidget(QLabel("Brokers:"))
        for cb in self.checks.values():
            brokers.addWidget(cb)
        brokers.addStretch()
        brokers.addWidget(load)

        self.kpis = {k: Kpi(label) for k, label in (
            ("value", "Total equity"), ("income", "Annual dividend income"), ("current", "Current dividend yield"),
            ("annual", "Annual (trailing) yield"), ("yoc", "Yield on cost"))}
        kpi_row = QHBoxLayout()
        for k in self.kpis.values():
            kpi_row.addWidget(k)
        self.donut = DonutChart(themes)
        self.by_class = DonutChart(themes)
        self.model = DataFrameModel(formats={
            "weight": pct, "current_yield": pct, "annual_yield": pct, "yield_on_cost": pct,
            "value": lambda v: f"{v:,.2f}", "price": lambda v: f"{v:,.4g}", "annual_income": lambda v: f"{v:,.2f}",
        })
        table = _table(self.model)
        table.doubleClicked.connect(lambda idx: self.open_symbol.emit(self.model.df.iat[idx.row(), 0]))

        charts = QHBoxLayout()
        charts.addWidget(card(QLabel("Allocation by holding"), self.donut), 2)
        charts.addWidget(card(QLabel("Allocation by asset class"), self.by_class), 1)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.addWidget(_title("Portfolio & Dividends"))
        box = card()
        box.layout().addLayout(brokers)
        lay.addWidget(box)
        lay.addLayout(kpi_row)
        lay.addLayout(charts)
        lay.addWidget(table, 1)

    def load(self):
        from tradebot.portfolio import build_portfolio

        names = [n for n, cb in self.checks.items() if cb.isChecked()]
        self.status.emit(f"loading portfolio from {', '.join(names)} …")

        def work():
            positions = [p for n in names for p in create_broker(n, data=self.engine).get_positions()]
            return build_portfolio(positions, self.engine.latest_price)

        run_async(work, self._loaded, lambda e: self.status.emit(f"portfolio failed: {e}"))

    def _loaded(self, pf):
        self.kpis["value"].set(f"{pf.total_value:,.2f}")
        self.kpis["income"].set(f"{pf.annual_income:,.2f}")
        self.kpis["current"].set(pct(pf.current_yield))
        self.kpis["annual"].set(pct(pf.annual_yield))
        self.kpis["yoc"].set(pct(pf.yield_on_cost))
        self.donut.set_weights(pf.allocation())
        self.by_class.set_weights(pf.allocation("asset_class"))
        self.model.set_frame(pf.to_frame().sort_values("value", ascending=False))
        self.loaded.emit([h.symbol if h.asset_class.value == "stock" else f"{h.symbol}/USDT" for h in pf.holdings])
        self.status.emit(f"portfolio: {len(pf.holdings)} holdings")


# -- backtest lab -----------------------------------------------------------------------------------------
class BacktestView(QWidget):
    status = pyqtSignal(str)
    show_on_chart = pyqtSignal(str, object)
    generation = pyqtSignal(int, float)  # emitted from the worker thread, delivered queued

    def __init__(self, engine, themes: ThemeManager, watchlist: list[str], symbol_search: SymbolSearch | None = None):
        super().__init__()
        self.engine, self.themes = engine, themes
        self.symbol = QLineEdit(watchlist[0] if watchlist else "BTC/USDT")
        self.symbols = SymbolCompleter(self.symbol, symbol_search or SymbolSearch(lambda: watchlist))
        self.tf = QComboBox()
        self.tf.addItems(list(TIMEFRAMES))
        self.tf.setCurrentText("1d")
        self.bars = QSpinBox()
        self.bars.setRange(100, 20000)
        self.bars.setValue(1500)
        self.dates = DateRange()
        self.fee = QDoubleSpinBox()
        self.fee.setDecimals(4)
        self.fee.setRange(0, 0.05)
        self.fee.setValue(0.001)
        self.algo = QComboBox()
        self.algo.addItems(ALGORITHMS.names())
        self.algo.currentTextChanged.connect(self._algo_changed)
        self.params = ParamForm({})
        self.population = QSpinBox()
        self.population.setRange(16, 20000)
        self.population.setValue(512)
        self.generations = QSpinBox()
        self.generations.setRange(1, 500)
        self.generations.setValue(20)
        self.objective = QComboBox()
        self.objective.addItems(["sharpe", "calmar", "return"])
        self.method = QComboBox()
        for label, key in (("Genetic algorithm (GA)", "ga"), ("Particle swarm (PSO)", "pso"),
                           ("Both + compare", "both")):
            self.method.addItem(label, key)
        self.method.setToolTip("'Both' runs GA, then PSO on the same data and budget and reports how much they "
                               "disagree. Big disagreement means a fragile optimum.")
        run = QPushButton("Run backtest")
        run.setObjectName("Primary")
        run.clicked.connect(self.run_backtest)
        opt = QPushButton("Optimize")
        opt.clicked.connect(self.run_optimizer)
        chart = QPushButton("Show on chart")
        chart.clicked.connect(lambda: self.show_on_chart.emit(self.symbol.text(), self._algorithm()))
        self.progress = QProgressBar()
        self.progress.hide()

        form = QFormLayout()
        for label, w in (("Symbol", self.symbol), ("Timeframe", self.tf), ("Candles", self.bars), ("Dates", self.dates),
                         ("Fee / side", self.fee), ("Algorithm", self.algo)):
            form.addRow(label, w)
        opt_form = QFormLayout()
        for label, w in (("Method", self.method), ("Population / swarm", self.population),
                         ("Generations / iterations", self.generations), ("Objective", self.objective)):
            opt_form.addRow(label, w)
        from tradebot.optimize import select_backend

        backend = QLabel(f"Compute backend: {select_backend().description}")
        backend.setObjectName("Muted")
        backend.setWordWrap(True)
        left = QVBoxLayout()
        left.addWidget(card(_title("Backtest lab")))
        setup = card()
        setup.layout().addLayout(form)
        left.addWidget(setup)
        params_card = card(QLabel("Parameters"), self.params)
        left.addWidget(params_card)
        opt_card = card(QLabel("Optimizer"))
        opt_card.layout().addLayout(opt_form)
        opt_card.layout().addWidget(backend)
        left.addWidget(opt_card)
        buttons = QHBoxLayout()
        for b in (run, opt, chart):
            buttons.addWidget(b)
        left.addLayout(buttons)
        left.addWidget(self.progress)
        left.addStretch()
        left_w = QWidget()
        left_w.setLayout(left)
        scroll = QScrollArea()
        scroll.setWidget(left_w)
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(380)
        enable_kinetic_scroll(scroll)
        scroll.setStyleSheet("QScrollArea { background: transparent; border: none; } QScrollArea > QWidget > QWidget { background: transparent; }")

        self.plot = pg.PlotWidget(axisItems={"bottom": pg.DateAxisItem()})
        self.plot.showGrid(x=True, y=True, alpha=0.15)
        self.plot.addLegend()
        self.touch = TouchController(self.plot.viewport(), PlotTouchHandler(self.plot, lambda _: self.plot.getPlotItem()))
        self.metrics = DataFrameModel()
        metrics_view = _table(self.metrics)
        metrics_view.setMaximumHeight(300)
        right = QVBoxLayout()
        right.addWidget(card(QLabel("Equity curve vs. buy & hold"), self.plot), 3)
        right.addWidget(metrics_view, 2)
        self.verdict = QLabel()
        self.verdict.setWordWrap(True)
        self.verdict.setTextFormat(Qt.TextFormat.RichText)
        self.verdict_card = card(self.verdict)
        self.verdict_card.hide()
        right.addWidget(self.verdict_card)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.addWidget(scroll)
        lay.addLayout(right, 1)
        self._algo_changed(self.algo.currentText())
        self.generation.connect(self._progress)
        themes.changed.connect(self._style)
        self._style(themes.current)

    def _style(self, t):
        self.plot.setBackground(None if t.glass else qcolor(t.panel))
        self.plot.setStyleSheet("background: transparent;" if t.glass else "")
        for name in ("left", "bottom"):
            self.plot.getAxis(name).setTextPen(pg.mkPen(qcolor(t.muted)))
            self.plot.getAxis(name).setPen(pg.mkPen(qcolor(t.panel_border)))

    def _algo_changed(self, name):
        algo = build_algorithm(name)
        self.params.set_space(algo.param_space(), algo.values)

    def _algorithm(self):
        return build_algorithm(self.algo.currentText()).with_params(**self.params.values())

    def _data(self):
        """Runs in the worker thread: the last N candles, or the From/To range when set."""
        symbol, bars = self.symbol.text().strip(), self.bars.value()
        start, end = self.dates.range()
        df = self.engine.get_ohlc(symbol, self.tf.currentText(), start=start, end=end, bars=bars)
        if note := shortfall(df, bars, start):
            self.status.emit(f"{symbol}:{note}")
        return df

    def _busy(self, on: bool, maximum: int = 0):
        self.progress.setRange(0, maximum)
        self.progress.setVisible(on)

    def run_backtest(self):
        from tradebot.backtest import backtest

        algo, fee = self._algorithm(), self.fee.value()
        self._busy(True)

        def work():
            df = self._data()
            return df, {"backtest": backtest(df, algo, fee=fee)}

        run_async(work, self._show, self._fail)

    def run_optimizer(self):
        from tradebot.optimize import OPTIMIZERS, run_comparison

        algo, fee = self._algorithm(), self.fee.value()
        method = self.method.currentData()
        kwargs = dict(population=self.population.value(), generations=self.generations.value(),
                      objective=self.objective.currentText(), fee=fee, min_trades=4)
        gens = kwargs["generations"]
        self._busy(True, gens * (2 if method == "both" else 1))
        self._phase = {"name": method.upper(), "offset": 0}
        self.status.emit("optimizing …")

        def phase(name):  # called in the worker thread; only touches plain data
            self._phase = {"name": name.upper(), "offset": 0 if name == "ga" else gens}

        def report(step, best, _params):
            self.generation.emit(self._phase["offset"] + step, best)

        def work():
            df = self._data()
            if method == "both":
                cmp = run_comparison(algo, df, progress=report, on_phase=phase, **kwargs)
                return df, {"GA in-sample": cmp.ga.train, "GA out-of-sample": cmp.ga.test,
                            "PSO in-sample": cmp.pso.train, "PSO out-of-sample": cmp.pso.test,
                            "_params": cmp.better.best_params, "_comparison": cmp}
            res = OPTIMIZERS[method](algo, df, **kwargs).run(report)
            return df, {"in-sample": res.train, "out-of-sample": res.test, "_params": res.best_params}

        run_async(work, self._show, self._fail)

    def _progress(self, step, best):
        self.progress.setValue(step + 1)
        self.status.emit(f"{self._phase['name']} step {step + 1}: best {self.objective.currentText()} = {best:.3f}")

    def _fail(self, err):
        self._busy(False)
        self.status.emit(f"backtest failed: {err}")

    def _show(self, payload):
        df, results = payload
        self._busy(False)
        params = results.pop("_params", None)
        comparison = results.pop("_comparison", None)
        self._show_comparison(comparison)
        if params:
            self.params.set_space(self._algorithm().param_space(), params)
        t = self.themes.current
        self.plot.clear()
        first = next(r for r in results.values() if r is not None)
        x_all = epoch_seconds(df.index)
        hold = first.equity.iloc[0] * df["close"] / df["close"].iloc[0]
        self.plot.plot(x_all, hold.to_numpy(), pen=pg.mkPen(qcolor(t.muted), width=1, style=Qt.PenStyle.DashLine), name="buy & hold")
        rows = {}
        for (name, res), color in zip(results.items(), t.series):
            if res is None:
                continue
            eq = res.equity
            if name.endswith("out-of-sample"):  # continue from where the matching in-sample curve ended
                eq = eq / eq.iloc[0] * results[name.replace("out-of-sample", "in-sample")].equity.iloc[-1]
            self.plot.plot(epoch_seconds(eq.index), eq.to_numpy(), pen=pg.mkPen(color, width=2), name=name)
            rows[name] = {k: fmt(k, v) for k, v in res.metrics.items()}
        self.metrics.set_frame(pd.DataFrame(rows).rename_axis("metric").reset_index())
        self.status.emit("done" + (f" · best params {params}" if params else ""))

    def _show_comparison(self, cmp):
        """Verdict card for GA-vs-PSO runs: parameter differences, fitness gap and interpretation."""
        if cmp is None:
            self.verdict_card.hide()
            return
        t = self.themes.current
        color = {"consistent": t.up, "fragile": "#f5a524", "divergent": t.down}[cmp.verdict]
        rows = "".join(
            f"<tr><td>{k}</td><td align='right'>{cmp.ga.best_params[k]:.4g}</td>"
            f"<td align='right'>{cmp.pso.best_params[k]:.4g}</td><td align='right'>{d:.0%}</td></tr>"
            for k, d in cmp.param_diff.items())
        self.verdict.setText(
            f"<b style='color:{color}'>{cmp.verdict.upper()}</b> · agreement {cmp.agreement:.0%} · "
            f"in-sample {cmp.objective} gap {cmp.fitness_gap:.1%}<br>{cmp.explanation()}"
            f"<table cellspacing='6'><tr><th align='left'>param</th><th>GA</th><th>PSO</th>"
            f"<th>difference (% of range)</th></tr>{rows}</table>"
            f"Parameters on the left were set to the better in-sample result ({cmp.better.method.upper()}).")
        self.verdict_card.show()
