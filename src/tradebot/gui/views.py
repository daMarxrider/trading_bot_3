# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Screener, news, portfolio and backtest-lab pages."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt6.QtCore import QEvent, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen
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
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableView,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from tradebot.algorithms import ALGORITHMS, build_algorithm
from tradebot.algorithms.builtin import VotingEnsemble
from tradebot.backtest.baselines import BUY_HOLD, MONTHLY, baselines, buy_and_hold, monthly_investing
from tradebot.backtest.metrics import fmt
from tradebot.config import Settings
from tradebot.brokers import BROKERS, create_broker
from tradebot.core.timeframes import TIMEFRAMES
from tradebot.glossary import METHODS, OBJECTIVES, STRATEGIES, explain
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
    tooltip,
)
from tradebot.gui.chart import PriceAxis, visible_y_range
from tradebot.gui.news import fill_news_list, open_link
from tradebot.gui.params import ParamForm
from tradebot.gui.theme import ThemeManager, qcolor
from tradebot.gui.touch import LongPressTooltips, PlotTouchHandler, TouchController, enable_kinetic_scroll
from tradebot.symbols import SymbolSearch


def _title(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("Title")
    return label


def _explain_choices(combo: QComboBox, intro: str, texts: dict):
    """A tooltip per entry (keyed by the entry's data, or its text) and one for the box that explains all of
    them, since the open list's tooltips can't be reached by touch."""
    lines = [intro] if intro else []
    for i in range(combo.count()):
        key = combo.itemData(i) or combo.itemText(i)
        if texts.get(key):
            combo.setItemData(i, tooltip(texts[key]), Qt.ItemDataRole.ToolTipRole)
            lines.append(f"<b>{combo.itemText(i)}</b>: {texts[key]}")
    combo.setToolTip("<qt>" + "<br>".join(lines) + "</qt>")


def _explained_row(form: QFormLayout, label: str, widget: QWidget, term: str):
    """A form row whose label and field both explain ``term`` on hover."""
    form.addRow(label, widget)
    text = tooltip(explain(term))
    form.labelForField(widget).setToolTip(text)
    widget.setToolTip(widget.toolTip() or text)  # a combo box keeps its list of choices


def _results_tip(column: str) -> str:
    """Results-table header: "GA out-of-sample" -> the out-of-sample explanation."""
    for term in (BUY_HOLD, MONTHLY, "out-of-sample", "in-sample", "backtest"):
        if term in column:
            return tooltip(explain(term))
    return tooltip("Point at a row to see what that number means.") if column == "metric" else ""


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
class MemberPicker(QWidget):
    """Checkboxes for the vote ensemble's members (#78): every registered algorithm but the vote itself, plus All.

    Starts with the ensemble's default members; unchecking All goes back to them. One member always stays checked.
    """

    changed = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.default = [m.name for m in VotingEnsemble().members]
        self.all = QCheckBox("All")
        self.all.setToolTip(tooltip("Use every known strategy, plugins included."))
        self.all.clicked.connect(self._all_clicked)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.all)
        self.boxes: dict[str, QCheckBox] = {}
        for name in ALGORITHMS.names():
            if issubclass(ALGORITHMS.get(name), VotingEnsemble):  # vote and pool can't be members
                continue
            box = self.boxes[name] = QCheckBox(name)
            box.setToolTip(tooltip(STRATEGIES.get(name) or ALGORITHMS.get(name).description))
            box.toggled.connect(lambda on, b=box: self._toggled(b, on))
            lay.addWidget(box)
        self.set_selected(self.default)

    def selected(self) -> list[str]:
        return [name for name, box in self.boxes.items() if box.isChecked()]

    def set_selected(self, names):
        for name, box in self.boxes.items():
            box.blockSignals(True)
            box.setChecked(name in names)
            box.blockSignals(False)
        self._sync_all()
        self.changed.emit()

    def _all_clicked(self, on):
        self.set_selected(list(self.boxes) if on else self.default)

    def _toggled(self, box, on):
        if not on and not self.selected():
            box.setChecked(True)  # the vote needs at least one member
            return
        self._sync_all()
        self.changed.emit()

    def _sync_all(self):
        self.all.setChecked(all(box.isChecked() for box in self.boxes.values()))


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
        self.list.itemActivated.connect(open_link)
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
        needle = self.filter.text().lower()
        items = [n for n in self.items if not needle or needle in (n.title + n.summary + n.symbol).lower()]
        fill_news_list(self.list, items, self.themes)


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

    def __init__(self, engine, themes: ThemeManager, watchlist: list[str], symbol_search: SymbolSearch | None = None,
                 settings: Settings | None = None):
        super().__init__()
        self.engine, self.themes, self.settings = engine, themes, settings
        self._x = np.empty(0)
        self._fit_series: list = []  # the plotted curves on self._x (NaN where a curve has no value), for auto-fit
        self._curves: list = []  # (name, x, values, pen) of the last run, redrawn when the scale changes
        self._log = False  # the plot's view y is log10(value) (#76)
        self._adjusting = False  # set while the lab changes the view range itself, as opposed to the user
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
        self.dca_amount = QDoubleSpinBox()  # the monthly-investing baseline (#19)
        self.dca_amount.setRange(0, 1e9)
        self.dca_amount.setDecimals(2)
        self.dca_amount.setSpecialValueText("auto")
        self.algo = QComboBox()
        self.algo.addItems(ALGORITHMS.names())
        self.algo.currentTextChanged.connect(self._algo_changed)
        self.params = ParamForm({})
        self.members = MemberPicker()
        self.members.changed.connect(self._members_changed)
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
        _explain_choices(self.method, explain("method"), METHODS)
        _explain_choices(self.objective, explain("objective"), OBJECTIVES)
        strategies = {name: STRATEGIES.get(name) or ALGORITHMS.get(name).description for name in ALGORITHMS.names()}
        _explain_choices(self.algo, explain("algorithm"), strategies)
        run = QPushButton("Run backtest")
        run.setObjectName("Primary")
        run.clicked.connect(self.run_backtest)
        opt = QPushButton("Optimize")
        opt.clicked.connect(self.run_optimizer)
        chart = QPushButton("Show on chart")
        chart.clicked.connect(lambda: self.show_on_chart.emit(self.symbol.text(), self._algorithm()))
        for button, term in ((run, "backtest"), (opt, "optimize"), (chart, "show_on_chart")):
            button.setToolTip(tooltip(explain(term)))
        self.progress = QProgressBar()
        self.progress.hide()

        form = QFormLayout()
        for label, w, term in (("Symbol", self.symbol, "symbol"), ("Timeframe", self.tf, "timeframe"),
                               ("Candles", self.bars, "candles"), ("Dates", self.dates, "dates"),
                               ("Fee / side", self.fee, "fee"), ("Monthly amount", self.dca_amount, "dca_amount"),
                               ("Algorithm", self.algo, "algorithm")):
            _explained_row(form, label, w, term)
        opt_form = QFormLayout()
        for label, w, term in (("Method", self.method, "method"), ("Population / swarm", self.population, "population"),
                               ("Generations / iterations", self.generations, "generations"),
                               ("Objective", self.objective, "objective")):
            _explained_row(opt_form, label, w, term)
        from tradebot.optimize import select_backend

        backend = QLabel(f"Compute backend: {select_backend().description}")
        backend.setObjectName("Muted")
        backend.setWordWrap(True)
        backend.setToolTip(tooltip(explain("backend")))
        left = QVBoxLayout()
        left.addWidget(card(_title("Backtest lab")))
        setup = card()
        setup.layout().addLayout(form)
        left.addWidget(setup)
        members_title = QLabel("Members")
        members_title.setToolTip(tooltip(explain("members")))
        self.members_card = card(members_title, self.members)
        left.addWidget(self.members_card)
        params_title = QLabel("Parameters")
        params_title.setToolTip(tooltip(explain("parameters")))
        params_card = card(params_title, self.params)
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

        self.value_axis = PriceAxis("left")
        self.plot = pg.PlotWidget(axisItems={"bottom": pg.DateAxisItem(), "left": self.value_axis})
        self.plot.showGrid(x=True, y=True, alpha=0.15)
        self.plot.addLegend()
        self.touch = TouchController(self.plot.viewport(), PlotTouchHandler(self.plot, lambda _: self.plot.getPlotItem(),
                                                                            on_reset=self.reset_view))
        self.plot.getPlotItem().vb.sigRangeChanged.connect(self._range_changed)
        # the same auto-fit toggle as in the chart's bottom-left corner
        self.fit_btn = QToolButton(self.plot, text="↕", checkable=True, toolTip="Auto-fit value axis to visible curves",
                                   objectName="ChartCorner")
        self.fit_btn.setChecked(bool(settings.get("backtest.autofit", True)) if settings else True)
        size = 40 if themes.touch else 28  # finger-sized in touch mode
        self.fit_btn.setFixedSize(size, size)
        self.fit_btn.toggled.connect(self._autofit_toggled)
        self.log_btn = QToolButton(self.plot, text="log", checkable=True, toolTip="Logarithmic value scale",
                                   objectName="ChartCorner")
        self.log_btn.setChecked(bool(settings.get("backtest.log", False)) if settings else False)
        self.log_btn.setFixedSize(size, size)
        self.log_btn.toggled.connect(self._log_toggled)
        self.plot.installEventFilter(self)  # keeps the button in the bottom-left corner
        self.metrics = DataFrameModel(row_tips=lambda row: tooltip(explain(row["metric"])), header_tips=_results_tip)
        self.metrics_view = metrics_view = _table(self.metrics)
        metrics_view.setMaximumHeight(300)
        right = QVBoxLayout()
        equity_title = QLabel("Equity curve vs. buy & hold and monthly investing")
        equity_title.setToolTip(tooltip(explain("equity")))
        right.addWidget(card(equity_title, self.plot), 3)
        self.baseline_note = QLabel()  # shown when the strategy loses to a baseline (#19)
        self.baseline_note.setWordWrap(True)
        self.baseline_note.hide()
        right.addWidget(self.baseline_note)
        right.addWidget(metrics_view, 2)
        self.verdict = QLabel()
        self.verdict.setWordWrap(True)
        self.verdict.setTextFormat(Qt.TextFormat.RichText)
        self.verdict.setToolTip("<qt>" + "<br>".join(f"<b>{name}</b>: {explain(term)}" for name, term in (
            ("Verdict", "verdict"), ("Agreement", "agreement"), ("Gap", "gap"),
            ("Difference (% of range)", "param_diff"))) + "</qt>")
        self.verdict_card = card(self.verdict)
        self.verdict_card.hide()
        right.addWidget(self.verdict_card)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.addWidget(scroll)
        lay.addLayout(right, 1)
        self._algo_changed(self.algo.currentText())
        self.generation.connect(self._progress)
        self._tips = LongPressTooltips(self)  # touch: press and hold anything to read its explanation
        themes.changed.connect(self._style)
        self._style(themes.current)

    def _style(self, t):
        self.plot.setBackground(None if t.glass else qcolor(t.panel))
        self.plot.setStyleSheet("background: transparent;" if t.glass else "")
        for name in ("left", "bottom"):
            self.plot.getAxis(name).setTextPen(pg.mkPen(qcolor(t.muted)))
            self.plot.getAxis(name).setPen(pg.mkPen(qcolor(t.panel_border)))

    def _algo_changed(self, name):
        self.members_card.setVisible(ALGORITHMS.get(name) is VotingEnsemble)
        algo = self._build()
        self.params.set_space(algo.param_space(), algo.values)

    def _members_changed(self):
        algo = self._build()
        self.params.set_space(algo.param_space(), {**algo.values, **self.params.values()})  # keep what was set

    def _build(self):
        """The selected algorithm with default parameters; a vote with the checked members."""
        name = self.algo.currentText()
        return build_algorithm(name, members=self.members.selected() if ALGORITHMS.get(name) is VotingEnsemble else None)

    def _algorithm(self):
        return self._build().with_params(**self.params.values())

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

    @property
    def _amount(self) -> float | None:
        return self.dca_amount.value() or None  # 0 shows as "auto": capital / months

    def _plot_baselines(self, df, fee, amount) -> dict:
        """Runs in the worker thread: both baselines over all candles, for the equity chart."""
        return {BUY_HOLD: buy_and_hold(df, fee), MONTHLY: monthly_investing(df, fee, amount=amount)}

    def run_backtest(self):
        from tradebot.backtest import backtest

        algo, fee, amount = self._algorithm(), self.fee.value(), self._amount
        self._busy(True)

        def work():
            df = self._data()
            res = backtest(df, algo, fee=fee)
            return df, {"backtest": res, "_baselines": {"backtest": baselines(df, res, fee, amount=amount)},
                        "_plot_baselines": self._plot_baselines(df, fee, amount)}

        run_async(work, self._show, self._fail)

    def run_optimizer(self):
        from tradebot.optimize import OPTIMIZERS, run_comparison

        algo, fee = self._algorithm(), self.fee.value()
        method = self.method.currentData()
        kwargs = dict(population=self.population.value(), generations=self.generations.value(),
                      objective=self.objective.currentText(), fee=fee, min_trades=4, dca_amount=self._amount)
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
            plot = self._plot_baselines(df, fee, kwargs["dca_amount"])
            if method == "both":
                cmp = run_comparison(algo, df, progress=report, on_phase=phase, **kwargs)
                return df, {"GA in-sample": cmp.ga.train, "GA out-of-sample": cmp.ga.test,
                            "PSO in-sample": cmp.pso.train, "PSO out-of-sample": cmp.pso.test,
                            "_params": cmp.better.best_params, "_comparison": cmp, "_plot_baselines": plot,
                            "_baselines": {f"{r.method.upper()} {period}": getattr(r, f"{attr}_baselines")
                                           for period, attr in (("in-sample", "train"), ("out-of-sample", "test"))
                                           for r in (cmp.ga, cmp.pso)}}
            res = OPTIMIZERS[method](algo, df, **kwargs).run(report)
            return df, {"in-sample": res.train, "out-of-sample": res.test, "_params": res.best_params,
                        "_baselines": {"in-sample": res.train_baselines, "out-of-sample": res.test_baselines},
                        "_plot_baselines": plot}

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
        bases = results.pop("_baselines", {})
        plot_bases = results.pop("_plot_baselines", None) or self._plot_baselines(df, self.fee.value(), self._amount)
        self._show_comparison(comparison)
        self._show_not_beaten(bases)
        if params:
            self.params.set_space(self._algorithm().param_space(), params)
        t = self.themes.current
        self._x = x_all = epoch_seconds(df.index)
        self._fit_series, self._curves = [], []
        for name, style in ((BUY_HOLD, Qt.PenStyle.DashLine), (MONTHLY, Qt.PenStyle.DotLine)):
            if name in plot_bases:
                values = plot_bases[name].equity.to_numpy(dtype=float)
                self._fit_series.append(values)
                self._curves.append((name, x_all, values, pg.mkPen(qcolor(t.muted), width=1, style=style)))
        rows, base_rows = {}, {}
        for name, res in results.items():  # the strategy's columns first, then one per baseline and period
            if res is not None and bases.get(name) is not None:
                period = name.removeprefix("GA ").removeprefix("PSO ")
                for col, metrics in bases[name].columns(name, res).items():
                    target = rows if col == name else base_rows
                    target[col if col == name or period == "backtest" else f"{col} {period}"] = metrics
        for (name, res), color in zip(results.items(), t.series):
            if res is None:
                continue
            eq = res.equity
            if name.endswith("out-of-sample"):  # continue from where the matching in-sample curve ended
                eq = eq / eq.iloc[0] * results[name.replace("out-of-sample", "in-sample")].equity.iloc[-1]
            self._curves.append((name, epoch_seconds(eq.index), eq.to_numpy(dtype=float), pg.mkPen(color, width=2)))
            rows.setdefault(name, res.metrics)
            self._fit_series.append(eq.reindex(df.index).to_numpy(dtype=float))
        columns = {name: {k: fmt(k, v) for k, v in metrics.items()} for name, metrics in {**rows, **base_rows}.items()}
        self.metrics.set_frame(pd.DataFrame(columns).fillna("–").rename_axis("metric").reset_index())
        self._draw()
        self._lock_view()
        self.reset_view()
        self.status.emit("done" + (f" · best params {params}" if params else ""))

    def _draw(self):
        """Plot the curves of the last run: log10 of the values while the log scale is on and all are above 0."""
        positive = all((y > 0).all() for _, _, y, _ in self._curves)
        self._log = self.value_axis.log = self.log_btn.isChecked() and positive
        if self.log_btn.isChecked() and not positive:
            self.status.emit("log scale needs values above 0: showing a linear scale")
        self.value_axis.picture = None  # re-generate the ticks
        self.plot.clear()
        for name, x, y, pen in self._curves:
            self.plot.plot(x, self._ty(y), pen=pen, name=name)

    def _ty(self, values):
        """Value -> the plot's view y: log10(value) on the log scale."""
        if not self._log:
            return values
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log10(values)

    def _log_toggled(self, on):
        if self.settings:
            self.settings.set("backtest.log", on)
        x_range = self.plot.getPlotItem().vb.viewRange()[0]
        self._adjusting = True
        try:
            self._draw()
            if self._curves:
                self.plot.setXRange(*x_range, padding=0)  # same time range on the other scale
        finally:
            self._adjusting = False
        self.fit_y(force=True)  # the old y-range means nothing on the other scale

    def reset_view(self):
        """Everything in view, still fitted while auto-fit is on (double tap, a new run)."""
        plot = self.plot.getPlotItem()
        self._adjusting = True
        try:
            plot.autoRange()  # autoRange() disables auto range in pyqtgraph, so re-enable afterwards
            plot.enableAutoRange()
        finally:
            self._adjusting = False
        self.fit_y()

    # -- y-axis auto-fit, as in the chart -------------------------------------------------------
    @property
    def autofit(self) -> bool:
        return self.fit_btn.isChecked()

    def fit_y(self, force: bool = False):
        """Fit the y-axis to the curves inside the visible time range (while auto-fit is on, or ``force``)."""
        if not (self.autofit or force):
            return
        plot = self.plot.getPlotItem()
        x0, x1 = plot.vb.viewRange()[0]
        self._adjusting = True
        try:
            plot.enableAutoRange(y=False)  # pyqtgraph would otherwise fit y to all data
            bounds = visible_y_range(self._x, [self._ty(s) for s in self._fit_series], x0, x1)
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
        margin = float(np.median(np.diff(self._x))) if len(self._x) > 1 else 86400.0
        vb = self.plot.getPlotItem().vb
        vb.setMouseEnabled(x=True, y=not on)
        vb.setLimits(**(dict(xMin=self._x[0] - margin, xMax=self._x[-1] + margin) if on else dict(xMin=None, xMax=None)))

    def _autofit_toggled(self, on):
        if self.settings:
            self.settings.set("backtest.autofit", on)
        self._lock_view()
        if on:
            self.fit_y()
        else:
            self.plot.getPlotItem().enableAutoRange(y=False)  # keep the current y-range until the user changes it

    def eventFilter(self, obj, event):
        if obj is self.plot and event.type() == QEvent.Type.Resize:
            bottom = self.plot.height() - self.fit_btn.height() - 8
            self.fit_btn.move(8, bottom)
            self.log_btn.move(8 + self.fit_btn.width() + 4, bottom)
        return super().eventFilter(obj, event)

    def _show_not_beaten(self, bases: dict):
        """A plain warning when the strategy loses to a baseline on unseen data (or in a plain backtest)."""
        lines = []
        for name, bl in bases.items():
            if bl is None or not bl.not_beaten or not (name == "backtest" or "out-of-sample" in name):
                continue
            where = "Over this period" if name == "backtest" else "Out-of-sample"
            who = "the strategy" if name in ("backtest", "out-of-sample") else f"the {name.split()[0]} result"
            lines.append(f"{where}, {who} does not beat {' or '.join(bl.not_beaten)}.")
        self.baseline_note.setText("<b>" + "<br>".join(lines) + "</b>")
        self.baseline_note.setStyleSheet(f"color: {self.themes.current.down};")
        self.baseline_note.setVisible(bool(lines))
        self.baseline_note.setToolTip(tooltip(explain("xirr")))

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
