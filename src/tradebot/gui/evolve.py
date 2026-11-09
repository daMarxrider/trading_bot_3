# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""The Evolve page (#83): evolve a whole strategy with genetic programming, see how honest the result is,
and keep it as a plugin algorithm."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from tradebot.algorithms import ALGORITHMS
from tradebot.backtest.baselines import BUY_HOLD, MONTHLY, buy_and_hold, monthly_investing
from tradebot.backtest.metrics import fmt
from tradebot.config import Settings, watchlists
from tradebot.evolve.search import LIMITS, SCRATCH_GUARDS, SCRATCH_ISLANDS, SCRATCH_LIMITS
from tradebot.core.timeframes import TIMEFRAMES
from tradebot.glossary import AGGREGATES, OBJECTIVES, explain
from tradebot.gui.chart import PlotAutoFit
from tradebot.gui.common import (
    DataFrameModel,
    DateRange,
    Panes,
    SymbolCompleter,
    card,
    epoch_seconds,
    run_async,
    shortfall,
    tooltip,
)
from tradebot.gui.theme import ThemeManager, qcolor
from tradebot.gui.touch import LongPressTooltips, enable_kinetic_scroll
from tradebot.gui.views import MemberPicker, _explain_choices, _explained_row, _signed_color, _table, _title
from tradebot.symbols import SymbolSearch

PARTS = ("train", "validation", "test")
LONG = {"vs monthly": "vs monthly investing"}  #: short column headers -> glossary terms


class Stopped(Exception):
    """Raised from the progress callback to end a run early."""


def indicator_picker() -> MemberPicker:
    """Every indicator the rules can use (those with at least one usable output), all checked."""
    from tradebot.evolve.primitives import indicator_outputs

    usable = list(dict.fromkeys(cls for cls, _, _ in indicator_outputs()))
    return MemberPicker({cls.name: (cls.name, cls.description) for cls in usable}, [cls.name for cls in usable],
                        all_tip="Let the rules use every indicator, plugins included.")


def plugin_path(name: str) -> Path:
    from tradebot.core.registry import default_plugin_dirs

    return default_plugin_dirs()[0] / f"{name}.py"


def on_common_x(curves) -> tuple[np.ndarray, list[np.ndarray]]:
    """The sorted times of all ``curves`` (series on a DatetimeIndex) and each curve's values on them, NaN where
    it has none: what the equity plot's auto-fit covers (train, validation and test cover different times)."""
    xs = [epoch_seconds(c.index) for c in curves]
    x = np.unique(np.concatenate(xs)) if xs else np.empty(0)
    out = []
    for cx, c in zip(xs, curves):
        values = np.full(len(x), np.nan)
        values[np.searchsorted(x, cx)] = c.to_numpy(dtype=float)
        out.append(values)
    return x, out


class EvolveView(QWidget):
    status = pyqtSignal(str)
    show_on_chart = pyqtSignal(str, object)
    saved = pyqtSignal(str)  #: an algorithm saved as a plugin (and registered), by name
    generation = pyqtSignal(int, float)  # emitted from the worker thread, delivered queued

    def __init__(self, engine, themes: ThemeManager, watchlist: list[str], symbol_search: SymbolSearch | None = None,
                 settings: Settings | None = None, watchlists_fn=None):
        super().__init__()
        self.engine, self.themes, self.settings = engine, themes, settings
        self.watchlists_fn = watchlists_fn or watchlists
        self.result = None  # the last EvolutionResult
        self._frames: dict = {}  # symbol -> candles of the last run
        self._shown = ""
        self._stop = False
        self._phase = {"name": "evolving", "offset": 0, "runs": 0}

        self.symbol = QLineEdit(watchlist[0] if watchlist else "BTC/USDT", placeholderText="AAPL, MSFT, BTC/USDT …")
        self.symbols = SymbolCompleter(self.symbol, symbol_search or SymbolSearch(lambda: watchlist), multi=True)
        self.lists_btn = QToolButton(text="☰", toolTip="Use a watchlist's symbols, or those of all watchlists")
        self.lists_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.lists_menu = QMenu(self.lists_btn)
        self.lists_menu.aboutToShow.connect(self._fill_lists_menu)
        self.lists_btn.setMenu(self.lists_menu)
        symbol_row = QWidget()
        row = QHBoxLayout(symbol_row)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.symbol, 1)
        row.addWidget(self.lists_btn)
        self.tf = QComboBox()
        self.tf.addItems(list(TIMEFRAMES))
        self.tf.setCurrentText("1d")
        self.bars = self._spin(200, 50000, 3000)
        self.dates = DateRange()
        self.fee = self._double(4, 0.05, 0.001)
        self.dca_amount = self._double(2, 1e9, 0.0)
        self.dca_amount.setSpecialValueText("auto")

        self.population = self._spin(8, 20000, 300)
        self.generations = self._spin(1, 1000, 40)
        self.max_depth = self._spin(2, 20, LIMITS[0])
        self.max_size = self._spin(4, 300, LIMITS[1])
        self.islands = self._spin(1, 64, 1)
        self.from_scratch = QCheckBox("Candles and maths only")
        self.from_scratch.toggled.connect(self._scratch_toggled)
        self.objective = QComboBox()
        self.objective.addItems(["sharpe", "calmar", "return"])
        _explain_choices(self.objective, explain("objective"), OBJECTIVES)
        self.aggregate = QComboBox()
        self.aggregate.addItems(list(AGGREGATES))
        _explain_choices(self.aggregate, explain("aggregate"), AGGREGATES)
        self.align = QCheckBox("Common date range only")
        self.state = QCheckBox("Bars in trade, stops, targets", checked=True)
        self.indicators = indicator_picker()

        self.validation = self._spin(0, 40, 20, " %")
        self.test = self._spin(0, 40, 20, " %")
        self.segments = self._spin(1, 20, 3)
        self.min_trades = self._spin(0, 1000, 4)
        self.parsimony = self._double(4, 0.1, 0.002)
        self.parsimony.setSingleStep(0.001)
        self.permutations = self._spin(0, 100, 0, " runs")
        self.permutations.setSpecialValueText("off")
        self.folds = self._spin(0, 20, 0, " folds")
        self.folds.setSpecialValueText("off")
        self.seed = self._spin(0, 2**31 - 1, 0)
        self.seed.setSpecialValueText("random")

        self._extra: dict = {}  # preset-only Evolver settings without a field (tournament, hall of fame, ...)
        self.best_btn = QPushButton("Best result")
        self.best_btn.clicked.connect(self.use_best_result)
        self.fast_btn = QPushButton("Fast (defaults)")
        self.fast_btn.clicked.connect(self.use_fast)
        self.best_btn.setToolTip(tooltip(explain("best_result")))
        self.fast_btn.setToolTip(tooltip(explain("fast_preset")))
        self.run_btn = QPushButton("Evolve", objectName="Primary")
        self.run_btn.clicked.connect(self.run)
        self.stop_btn = QPushButton("Stop", enabled=False)
        self.stop_btn.clicked.connect(self.stop)
        self.save_btn = QPushButton("Save as plugin…", enabled=False)
        self.save_btn.clicked.connect(self._ask_save)
        self.chart_btn = QPushButton("Show on chart", enabled=False)
        self.chart_btn.clicked.connect(lambda: self.show_on_chart.emit(self._shown or self._symbol_list()[0],
                                                                       self.result.algorithm("evolved_preview", False)))
        for button, term in ((self.run_btn, "evolve"), (self.save_btn, "save_plugin"), (self.chart_btn, "show_on_chart")):
            button.setToolTip(tooltip(explain(term)))
        self.progress = QProgressBar()
        self.progress.hide()

        data_form, search_form, honest_form = QFormLayout(), QFormLayout(), QFormLayout()
        for form, rows in (
            (data_form, (("Symbols", symbol_row, "symbol"), ("Timeframe", self.tf, "timeframe"),
                         ("Candles", self.bars, "candles"), ("Dates", self.dates, "dates"),
                         ("Fee / side", self.fee, "fee"), ("Monthly amount", self.dca_amount, "dca_amount"))),
            (search_form, (("Population", self.population, "evo_population"),
                           ("Generations", self.generations, "evo_generations"),
                           ("From scratch", self.from_scratch, "from_scratch"),
                           ("Max depth", self.max_depth, "max_depth"), ("Max size", self.max_size, "max_size"),
                           ("Islands", self.islands, "islands"),
                           ("Objective", self.objective, "objective"), ("Several symbols", self.aggregate, "aggregate"),
                           ("Align dates", self.align, "align"), ("Exit rules may use", self.state, "trade_state"),
                           ("Seed", self.seed, "seed"))),
            (honest_form, (("Validation", self.validation, "validation"), ("Test", self.test, "test_part"),
                           ("Segments", self.segments, "segments"), ("Min trades", self.min_trades, "min_trades"),
                           ("Parsimony", self.parsimony, "parsimony"),
                           ("Shuffled-price runs", self.permutations, "permutations"),
                           ("Walk-forward", self.folds, "evo_folds"))),
        ):
            for label, w, term in rows:
                _explained_row(form, label, w, term)
        left = QVBoxLayout()
        left.addWidget(card(_title("Evolve a strategy")))
        for title, body, term in (("Data", data_form, "symbol"), ("Search", search_form, "evolve"),
                                  ("Keeping it honest", honest_form, "test_part")):
            head = QLabel(title)
            head.setToolTip(tooltip(explain(term)))
            box = card(head)
            if term == "evolve":
                presets = QHBoxLayout()
                presets.addWidget(self.best_btn)
                presets.addWidget(self.fast_btn)
                box.layout().addLayout(presets)
            box.layout().addLayout(body)
            left.addWidget(box)
        ind_title = QLabel("Indicators")
        ind_title.setToolTip(tooltip(explain("evo_indicators")))
        self.indicators_card = card(ind_title, self.indicators)
        left.addWidget(self.indicators_card)
        buttons, more = QHBoxLayout(), QHBoxLayout()
        for b in (self.run_btn, self.stop_btn):
            buttons.addWidget(b)
        for b in (self.save_btn, self.chart_btn):
            more.addWidget(b)
        left.addLayout(buttons)
        left.addLayout(more)
        left.addWidget(self.progress)
        left.addStretch()
        left_w = QWidget()
        left_w.setLayout(left)
        scroll = QScrollArea()
        scroll.setWidget(left_w)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(380)  # the settings never scroll sideways; the divider only widens them
        enable_kinetic_scroll(scroll)
        scroll.setStyleSheet("QScrollArea { background: transparent; border: none; } "
                             "QScrollArea > QWidget > QWidget { background: transparent; }")

        # -- results ----------------------------------------------------------------------------------
        self.rules = QLabel("Press Evolve to search for a strategy.", wordWrap=True)
        self.rules.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        mono = QFont("monospace")
        mono.setStyleHint(QFont.StyleHint.Monospace)
        self.rules.setFont(mono)
        self.rules.setToolTip(tooltip(explain("rules")))
        self.info = QLabel(objectName="Muted", wordWrap=True)
        self.warnings = QLabel(wordWrap=True)
        self.warnings.setTextFormat(Qt.TextFormat.RichText)
        self.warnings.hide()
        self.equity = pg.PlotWidget(axisItems={"bottom": pg.DateAxisItem()})
        self.equity.showGrid(x=True, y=True, alpha=0.15)
        self.equity.addLegend()
        self.equity_fit = PlotAutoFit(self.equity, settings, "evolve.autofit", themes.touch)
        self.log_btn = QCheckBox("log scale")
        self.log_btn.setChecked(bool(settings.get("evolve.log", False)) if settings else False)
        self.log_btn.toggled.connect(self._log_toggled)
        self.history = pg.PlotWidget()
        self.history.showGrid(x=True, y=True, alpha=0.15)
        self.history.addLegend()
        self.history.setLabel("bottom", "generation")
        self.history_fit = PlotAutoFit(self.history, settings, "evolve.history_autofit", themes.touch,
                                       "Auto-fit score axis to visible generations")
        self.history.setToolTip(tooltip(explain("evo_history")))
        plots = Panes(Qt.Orientation.Horizontal, "evolve.plots", settings, themes.touch)
        self.equity.setMinimumHeight(140)
        equity_title = QLabel("Equity: train, then validation and test (shaded), vs. buy & hold and monthly investing")
        equity_title.setToolTip(tooltip(explain("equity")))
        equity_head = QHBoxLayout()
        equity_head.addWidget(equity_title, 1)
        equity_head.addWidget(self.log_btn)
        equity_card = card()
        equity_card.layout().addLayout(equity_head)
        equity_card.layout().addWidget(self.equity)
        history_title = QLabel("Best score per generation")
        history_title.setToolTip(tooltip(explain("evo_history")))
        plots.add(equity_card, 3)
        plots.add(card(history_title, self.history), 1)
        signed = _signed_color(themes, {"vs buy & hold", "vs monthly"})
        self.per_symbol = DataFrameModel(header_tips=lambda col: tooltip(explain(LONG.get(col, col)) or explain("per_symbol")),
                                         colors=signed)
        self.per_symbol_view = _table(self.per_symbol)
        self.per_symbol_view.setSortingEnabled(False)  # the aggregate row stays last
        self.per_symbol_view.clicked.connect(lambda idx: self.show_symbol(self.per_symbol.df.iat[idx.row(), 0]))
        self.per_symbol_title = QLabel("<b>Per symbol</b>")
        self.per_symbol_title.setToolTip(tooltip(explain("per_symbol")))
        self.hall = DataFrameModel(header_tips=lambda col: tooltip(explain("hall_of_fame")))
        self.hall_view = _table(self.hall)
        self.hall_title = QLabel("<b>Best strategies in training</b> · → = the one picked on validation")
        self.hall_title.setToolTip(tooltip(explain("hall_of_fame")))
        # draggable dividers between all parts (#95); the text cards scroll when made small
        right = self.panes = Panes(Qt.Orientation.Vertical, "evolve.panes", settings, themes.touch)
        right.add(card(self.rules, self.info, self.warnings), 1.6, scroll=True)
        right.add(plots, 4)
        right.add(card(self.per_symbol_title, self.per_symbol_view), 2, scroll=True)
        right.add(card(self.hall_title, self.hall_view), 2, scroll=True)
        self.columns = Panes(Qt.Orientation.Horizontal, "evolve.columns", settings, themes.touch)
        self.columns.add(scroll, 0, size=380)
        self.columns.add(right, 1)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.addWidget(self.columns)
        self.generation.connect(self._progress)
        self._tips = LongPressTooltips(self)
        themes.changed.connect(self._style)
        self._style(themes.current)

    @staticmethod
    def _spin(lo: int, hi: int, value: int, suffix: str = "") -> QSpinBox:
        box = QSpinBox()
        box.setRange(lo, hi)
        box.setValue(value)
        box.setSuffix(suffix)
        return box

    @staticmethod
    def _double(decimals: int, hi: float, value: float) -> QDoubleSpinBox:
        box = QDoubleSpinBox()
        box.setDecimals(decimals)  # before the value, which would otherwise be rounded to 2 decimals
        box.setRange(0.0, hi)
        box.setValue(value)
        return box

    def _style(self, t):
        for plot in (self.equity, self.history):
            plot.setBackground(None if t.glass else qcolor(t.panel))
            plot.setStyleSheet("background: transparent;" if t.glass else "")
            for name in ("left", "bottom"):
                plot.getAxis(name).setTextPen(pg.mkPen(qcolor(t.muted)))
                plot.getAxis(name).setPen(pg.mkPen(qcolor(t.panel_border)))
        if self.result is not None:
            self._draw()

    def _scratch_toggled(self, on: bool):
        """From scratch: no indicators to pick, bigger limits, and the safeguards on (each still adjustable)."""
        self.indicators_card.setEnabled(not on)
        old, new = (LIMITS, SCRATCH_LIMITS) if on else (SCRATCH_LIMITS, LIMITS)
        guards = {self.permutations: SCRATCH_GUARDS["permutations"], self.folds: SCRATCH_GUARDS["folds"]}
        islands = (1, SCRATCH_ISLANDS) if on else (SCRATCH_ISLANDS, 1)
        for box, a, b in ((self.max_depth, old[0], new[0]), (self.max_size, old[1], new[1]), (self.islands, *islands),
                          *((box, 0, v) if on else (box, v, 0) for box, v in guards.items())):
            if box.value() == a:  # only values still at the other mode's default
                box.setValue(b)

    def _apply(self, values: dict):
        """Fill the form from Evolver settings (the fields stay editable)."""
        boxes = {"population": self.population, "generations": self.generations, "max_depth": self.max_depth,
                 "max_size": self.max_size, "islands": self.islands, "segments": self.segments,
                 "min_trades": self.min_trades, "permutations": self.permutations, "folds": self.folds}
        for key, box in boxes.items():
            box.setValue(int(values[key]))
        self.validation.setValue(round(values["validation"] * 100))
        self.test.setValue(round(values["test"] * 100))
        self.parsimony.setValue(values["parsimony"])
        self.aggregate.setCurrentText(values["aggregate"])
        self.state.setChecked(values["state"])
        self._extra = {k: values[k] for k in ("tournament", "crossover", "mutation", "elite", "hall_of_fame")
                       if k in values}

    def use_best_result(self):
        """Every setting for the best out-of-sample result, ignoring runtime (#101), scaled to the data."""
        from tradebot.evolve import best_result_settings

        n = len(self._symbol_list()) or 1
        self._apply(best_result_settings(self.bars.value(), n, from_scratch=self.from_scratch.isChecked()))
        self.status.emit("best-result settings filled in: slow on purpose, every field can still be changed")

    def use_fast(self):
        """Today's quick defaults (those of the current mode)."""
        scratch = self.from_scratch.isChecked()
        limits = SCRATCH_LIMITS if scratch else LIMITS
        self._apply(dict(population=300, generations=40, max_depth=limits[0], max_size=limits[1],
                         islands=SCRATCH_ISLANDS if scratch else 1, segments=3, min_trades=4, parsimony=0.002,
                         aggregate="mean", state=True, validation=0.2, test=0.2,
                         permutations=SCRATCH_GUARDS["permutations"] if scratch else 0,
                         folds=SCRATCH_GUARDS["folds"] if scratch else 0))

    def _fill_lists_menu(self):
        self.lists_menu.clear()
        lists = self.watchlists_fn()
        for name, symbols in lists.items():
            self.lists_menu.addAction(f"{name} ({len(symbols)})", lambda s=symbols: self.symbol.setText(", ".join(s)))
        if len(lists) > 1:
            every = list(dict.fromkeys(s for symbols in lists.values() for s in symbols))
            self.lists_menu.addSeparator()
            self.lists_menu.addAction(f"All watchlists ({len(every)})", lambda: self.symbol.setText(", ".join(every)))

    def _symbol_list(self) -> list[str]:
        return list(dict.fromkeys(s for s in re.split(r"[,\s]+", self.symbol.text()) if s))

    def _data(self):
        """Runs in the worker thread: ``{symbol: candles}`` without the ones that failed to load."""
        symbols, bars = self._symbol_list(), self.bars.value()
        if not symbols:
            raise ValueError("enter a symbol")
        start, end = self.dates.range()
        frames, failed = {}, []
        for symbol in symbols:
            try:
                frames[symbol] = df = self.engine.get_ohlc(symbol, self.tf.currentText(), start=start, end=end, bars=bars)
            except Exception:
                if len(symbols) == 1:
                    raise
                failed.append(symbol)
                continue
            if note := shortfall(df, bars, start):
                self.status.emit(f"{symbol}:{note}")
        if failed:
            self.status.emit(f"could not load {', '.join(failed)}")
        if not frames:
            raise ValueError("none of the symbols could be loaded")
        return frames

    def settings_kwargs(self) -> dict:
        """The Evolver settings from the form."""
        picked = self.indicators.selected()
        scratch = self.from_scratch.isChecked()
        return dict(**self._extra, from_scratch=scratch, population=self.population.value(), generations=self.generations.value(),
                    max_depth=self.max_depth.value(), max_size=self.max_size.value(), islands=self.islands.value(),
                    objective=self.objective.currentText(), aggregate=self.aggregate.currentText(),
                    align=self.align.isChecked(), validation=self.validation.value() / 100, test=self.test.value() / 100,
                    segments=self.segments.value(), min_trades=self.min_trades.value(),
                    parsimony=self.parsimony.value(), state=self.state.isChecked(),
                    indicators=None if scratch or len(picked) == len(self.indicators.boxes) else picked,
                    fee=self.fee.value(), dca_amount=self.dca_amount.value() or None, seed=self.seed.value() or None)

    # -- running ----------------------------------------------------------------------------------
    def run(self):
        from tradebot.evolve import evolve

        kwargs = self.settings_kwargs()
        permutations, folds, gens = self.permutations.value(), self.folds.value(), kwargs["generations"]
        self._stop = False
        self._phase = {"name": "evolving", "offset": 0, "runs": 0}
        self.progress.setRange(0, gens * (1 + permutations + folds))
        self.progress.setValue(0)
        self.progress.show()
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status.emit("evolving …")

        def phase(label):  # worker thread: before each search (the real one, then the shuffled ones)
            runs = self._phase["runs"]
            self._phase = {"name": label, "offset": runs * gens, "runs": runs + 1}

        def report(gen, best, _program):
            if self._stop:
                raise Stopped()
            self.generation.emit(self._phase["offset"] + gen, best)

        def work():
            frames = self._data()
            data = frames if len(frames) > 1 else next(iter(frames.values()))
            res = evolve(data, permutations=permutations, folds=folds, progress=report, on_phase=phase, **kwargs)
            names = [r.symbol for r in res.symbols] if res.multi else [next(iter(frames))]
            bases = {name: {BUY_HOLD: buy_and_hold(frames[name], kwargs["fee"]),
                            MONTHLY: monthly_investing(frames[name], kwargs["fee"], amount=kwargs["dca_amount"])}
                     for name in names}  # over all candles, for the plot
            return res, {name: frames[name] for name in names}, bases

        run_async(work, self._done, self._fail)

    def stop(self):
        self._stop = True
        self.stop_btn.setEnabled(False)
        self.status.emit("stopping after this generation …")

    def _progress(self, step, best):
        self.progress.setValue(step + 1)
        self.status.emit(f"{self._phase['name']}: generation {step - self._phase['offset'] + 1}, "
                         f"best {self.objective.currentText()} = {best:.3f}")

    def _finish(self):
        self.progress.hide()
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def _fail(self, err):
        self._finish()
        self.status.emit("evolving stopped" if str(err).startswith("Stopped") else f"evolve failed: {err}")

    def _done(self, payload):
        self.result, frames, self._plot_bases = payload
        self._frames = frames
        self._finish()
        self.save_btn.setEnabled(True)
        self.chart_btn.setEnabled(True)
        self.show_result()
        self.status.emit(f"done · {self.result.evaluations} strategies tried in {self.result.seconds:.1f} s · "
                         f"seed {self.result.seed}")

    # -- showing ----------------------------------------------------------------------------------
    def show_result(self):
        res, t = self.result, self.themes.current
        self.rules.setText(str(res.program))
        dates = [f"train from {res.symbols[0].train.equity.index[0]:%Y-%m-%d}"]
        if res.validation_start is not None:
            dates.append(f"validation from {res.validation_start:%Y-%m-%d}")
        if res.test_start is not None:
            dates.append(f"test from {res.test_start:%Y-%m-%d}")
        modules = f" · {len(res.modules)} evolved building blocks" if res.from_scratch else ""
        self.info.setText(f"{' · '.join(dates)} · found in generation {res.generation + 1} of {len(res.history)} · "
                          f"{res.evaluations} strategies tried{modules} · seed {res.seed}")
        lines = []
        if (pt := res.permutation) is not None:
            color = t.up if pt.p <= 0.05 else t.down
            lines.append(f"<span style='color:{color}'>Shuffled prices: {pt.part} {res.objective} "
                         f"{', '.join(f'{s:.2f}' for s in pt.shuffled)} vs. {pt.score:.2f} for the real result · "
                         f"p = {pt.p:.2f}</span>")
        if (d := res.deflated) is not None:
            color = t.up if d.significant else t.down
            lines.append(f"<span style='color:{color}'>Deflated Sharpe ratio {d.probability:.2f} "
                         f"({d.trials} different strategies tried)</span>")
        if (wf := res.walk_forward) is not None:
            scores = ", ".join("–" if v is None else f"{v:.2f}" for v in wf.scores())
            color = t.up if not wf.warnings(res.objective) else t.down
            lines.append(f"<span style='color:{color}'>Walk-forward, {res.objective} after each window: {scores} · "
                         f"beats buy &amp; hold in {wf.beating()} of {len(wf.folds)}</span>")
        lines += [f"<b style='color:{t.down}'>{w}</b>" for w in res.warnings()]
        self.warnings.setText("<br>".join(lines))
        self.warnings.setVisible(bool(lines))
        self._fill_per_symbol()
        self._fill_hall()
        self._shown = next(iter(self._frames))
        self._draw()

    def _fill_per_symbol(self):
        res = self.result
        obj, unseen = res.objective, res.unseen
        rows = []
        for r in res.rows():
            rows.append({"symbol": r["symbol"] or next(iter(self._frames)),
                         **{f"{obj} {p}": r[p] for p in PARTS},
                         **({f"xirr {unseen}": r["xirr"], "vs buy & hold": r["vs buy & hold"],
                             "vs monthly": r["vs monthly investing"], f"trades {unseen}": r["trades"]}
                            if unseen else {})})
        if res.multi:
            n, beats = len(res.symbols), res.beats(unseen) if unseen else {}
            rows.append({"symbol": res.aggregate, **{f"{obj} {p}": res.aggregated(p) for p in PARTS},
                         **({"vs buy & hold": f"beats on {beats[BUY_HOLD]}/{n}",
                             "vs monthly": f"beats on {beats[MONTHLY]}/{n}"} if unseen else {})})
        columns = list(dict.fromkeys(k for row in rows for k in row))
        self.per_symbol.formats = {c: (lambda v: f"{v:.3f}") if c.startswith(obj) else
                                   (lambda v: f"{v:.0f}") if c.startswith("trades") else
                                   (lambda v, c=c: v if isinstance(v, str) else fmt("xirr" if c.startswith("xirr")
                                                                                    else LONG.get(c, c), v))
                                   for c in columns if c != "symbol"}
        self.per_symbol.set_frame(pd.DataFrame(rows, columns=columns))
        self.per_symbol_title.setText("<b>Per symbol</b>" + (" · click a row to plot that symbol" if res.multi else ""))

    def _fill_hall(self):
        res = self.result
        best = sorted(res.hall_of_fame, key=lambda c: -c.fitness)
        rows = [{"": "→" if c.program.key == res.program.key else "", "training": c.fitness,
                 "validation": c.validation, "generation": c.generation + 1, **c.program.rules()} for c in best]
        self.hall.formats = {"training": lambda v: f"{v:.3f}", "validation": lambda v: f"{v:.3f}",
                             "generation": lambda v: f"{v:.0f}"}
        self.hall.set_frame(pd.DataFrame(rows, columns=["", "training", "validation", "generation", "entry", "exit"]))
        header = self.hall_view.horizontalHeader()
        for i in range(4):  # the numbers fit their contents; the rules share the rest
            header.setSectionResizeMode(i, QHeaderView.ResizeMode.ResizeToContents)

    def show_symbol(self, symbol: str):
        if symbol in self._frames:
            self._shown = symbol
            self._draw()

    def _log_toggled(self, on):
        if self.settings:
            self.settings.set("evolve.log", on)
        if self.result is not None:
            self._draw()

    def curves(self, symbol: str) -> dict[str, pd.Series]:
        """The shown symbol's equity per part (each part continuing where the one before ended) and the baselines."""
        report = next((r for r in self.result.symbols if r.symbol in (symbol, "")), None)
        out, last = {}, None
        for part in PARTS:
            res = getattr(report, part) if report is not None else None
            if res is None:
                continue
            eq = res.equity
            if last is not None:
                eq = eq / eq.iloc[0] * last
            out[part] = eq
            last = float(eq.iloc[-1])
        for name, res in self._plot_bases.get(symbol, {}).items():
            out[name] = res.equity
        return out

    def _draw(self):
        t, res = self.themes.current, self.result
        curves = self.curves(self._shown)
        log = self.log_btn.isChecked() and all((c > 0).all() for c in curves.values())
        self.equity.clear()
        self.equity.setLogMode(y=log)
        muted = qcolor(t.muted)
        for i, (label, start) in enumerate((("validation", res.validation_start), ("test", res.test_start))):
            if start is None or self._shown not in self._frames:
                continue
            index = self._frames[self._shown].index
            end = res.test_start if label == "validation" and res.test_start is not None else index[-1]
            fill = QColor(muted)
            fill.setAlpha(26 if i else 44)
            region = pg.LinearRegionItem(epoch_seconds(pd.DatetimeIndex([start, end])), movable=False,
                                         brush=fill, pen=pg.mkPen(None))
            region.setZValue(-10)
            pg.InfLineLabel(region.lines[0], text=label, position=0.95, anchors=[(0, 0), (0, 0)], color=muted)
            self.equity.addItem(region, ignoreBounds=True)
        colors = dict(zip(PARTS, t.series))
        for name, eq in curves.items():
            if name in PARTS:
                pen = pg.mkPen(colors[name], width=2)
            else:
                pen = pg.mkPen(muted, width=1, style=Qt.PenStyle.DashLine if name == BUY_HOLD else Qt.PenStyle.DotLine)
            self.equity.plot(epoch_seconds(eq.index), eq.to_numpy(dtype=float), pen=pen, name=name)
        self.equity.getPlotItem().setTitle(self._shown if res.multi else None, color=muted)
        self.equity_fit.set_data(*on_common_x(list(curves.values())), log)
        self.history.clear()
        gens = np.array([h["generation"] + 1 for h in res.history], dtype=float)
        self.history.plot(gens, [h["best"] for h in res.history], pen=pg.mkPen(colors["train"], width=2), name="training")
        scores = [np.array([h["best"] for h in res.history], dtype=float)]
        valid = [h["validation"] for h in res.history]
        if all(v is not None for v in valid):
            self.history.plot(gens, valid, pen=pg.mkPen(colors["validation"], width=2), name="validation")
            scores.append(np.array(valid, dtype=float))
        self.history_fit.set_data(gens, scores)

    # -- keeping the result -----------------------------------------------------------------------
    def _ask_save(self):
        base = re.sub(r"[^a-z0-9_]", "_", (self._shown or "x").lower()).strip("_") or "x"
        name, ok = QInputDialog.getText(self, "Save as plugin", "Name of the new strategy (lower-case letters, "
                                        "digits and _):", text=f"evolved_{base}")
        if not ok or not name:
            return
        path = plugin_path(name)
        if path.exists() and QMessageBox.question(self, "Save as plugin", f"{path} exists. Replace it?") \
                != QMessageBox.StandardButton.Yes:
            return
        try:
            self.save_plugin(name, overwrite=True)
        except ValueError as e:
            QMessageBox.warning(self, "Save as plugin", str(e))

    def save_plugin(self, name: str, overwrite: bool = False) -> Path:
        """Write the result as a plugin, register it at once and announce it (``saved``)."""
        from tradebot.evolve import valid_name
        from tradebot.evolve.export import load_plugin

        if not valid_name(name):
            raise ValueError(f"{name!r} can't be a strategy name: use lower-case letters, digits and _")
        path = plugin_path(name)
        if path.exists() and not overwrite:
            raise ValueError(f"{path} exists")
        if name in ALGORITHMS and not path.exists():
            raise ValueError(f"there already is a strategy called {name!r}")
        source = self.result.plugin_source(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
        load_plugin(source, f"tradebot_plugin_{name}")  # usable right away, without a restart
        self.saved.emit(name)
        self.status.emit(f"saved {name} in {path}")
        return path
