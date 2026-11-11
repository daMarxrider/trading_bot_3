# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""The Scanner page (#107): one indicator over a source of symbols, filtered on its plots and alerts.

* #111: the source pill and dialog (a watchlist or a market index, exactly one)
* #112: the indicator pill and picker, its numeric inputs, the timeframe
* #113: filter chips, the results table with the indicator's outputs as columns
* #114: scan, progress, summary, and the rescan prompt when an input changed

The engine is :mod:`tradebot.scanner`; the page is settings.json-backed (``scanner.*`` keys).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from html import escape
from typing import Callable

import numpy as np
import pandas as pd
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from tradebot import scanner as engine_scan
from tradebot import universes
from tradebot.config import Settings
from tradebot.core.registry import resolve_params
from tradebot.gui.common import DataFrameModel, pct, run_async, tooltip
from tradebot.gui.params import ParamDialog
from tradebot.gui.theme import ThemeManager
from tradebot.gui.touch import enable_kinetic_scroll
from tradebot.indicators import INDICATORS
from tradebot.indicators.base import BaseIndicator, is_screenable

DEFAULT_INDICATOR = "ma_cross"
DEFAULT_TIMEFRAME = "1d"
MAX_AUTO_COLUMNS = 10
PROGRESS_AFTER_MS = 2000
#: the engine's timeframes as the page labels them
TF_LABELS = {"1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m", "1h": "1h", "2h": "2h", "4h": "4h",
             "1d": "1D", "1w": "1W", "1mo": "1M"}
OP_LABELS = {">": ">", ">=": "≥", "<": "<", "<=": "≤", "==": "=", "!=": "≠"}
NOT_SCREENABLE_FOOTER = "{n} indicator{s} hidden: chart patterns or nothing to screen"


# -- helpers -------------------------------------------------------------------------------------------
def highlight(text: str, query: str) -> str:
    """``text`` as HTML with the (case-insensitive) matches of ``query`` in bold."""
    query = query.strip()
    if not query:
        return escape(text)
    low, out, i = text.lower(), [], 0
    while (j := low.find(query.lower(), i)) >= 0:
        out += [escape(text[i:j]), f"<b>{escape(text[j:j + len(query)])}</b>"]
        i = j + len(query)
    return "".join(out) + escape(text[i:])


def pretty(name: str) -> str:
    """``ma_cross`` -> ``MA Cross``."""
    return " ".join(w.upper() if len(w) <= 4 else w.capitalize() for w in name.split("_"))


def indicator_group(cls: type[BaseIndicator]) -> str:
    module = cls.__module__
    if module.startswith("tradebot_plugin_"):
        return "My scripts"
    return "Built-in" if module.startswith("tradebot.indicators") else "Community"


def plot_outputs(cls: type[BaseIndicator]) -> list[str]:
    """What the Scanner shows as plots: visible lines and labels, in ``outputs`` order."""
    return [o for o in cls.outputs if o not in cls.hidden and o not in cls.alerts]


def screenable_split() -> tuple[list[type[BaseIndicator]], dict[str, str]]:
    """The indicators the picker lists, and ``{name: reason}`` of those it hides."""
    shown, hidden = [], {}
    for cls in INDICATORS:
        ok, why = is_screenable(cls)
        if ok:
            shown.append(cls)
        else:
            hidden[cls.name] = why
    return shown, hidden


def formatter(cls: type[BaseIndicator], output: str) -> Callable:
    """Number format by what the output measures (its unit)."""
    if output in cls.alerts:
        return lambda v: "✓" if v else "–"
    from tradebot.evolve.primitives import output_unit

    unit = output_unit(cls, output)
    if unit == "ratio":
        return pct
    if unit == "osc":
        return lambda v: f"{v:.1f}"
    return lambda v: f"{v:,.2f}" if abs(v) >= 1 else f"{v:.4g}"


# -- search list + dialogs (#111, #112) --------------------------------------------------------------------
@dataclass
class Row:
    key: str
    title: str
    sub: str = ""
    group: str = ""
    badge: str = ""
    icon: str = ""
    extra: str = ""  #: more text the search matches

    def matches(self, query: str) -> bool:
        q = query.strip().lower()
        return not q or q in f"{self.title} {self.sub} {self.extra}".lower()


class SearchList(QWidget):
    """A search field above a list of rows; the matching part is highlighted, an empty result says so."""

    chosen = pyqtSignal(str)

    def __init__(self, noun: str, rows: list[Row], parent=None):
        super().__init__(parent)
        self.noun, self.rows = noun, rows
        self.search = QLineEdit(placeholderText=f"Search {noun} …", clearButtonEnabled=True)
        self.search.textChanged.connect(self.render)
        self.list = QListWidget()
        self.list.setVerticalScrollMode(QListWidget.ScrollMode.ScrollPerPixel)
        enable_kinetic_scroll(self.list)
        self.list.itemClicked.connect(self._clicked)
        self.empty = QLabel(objectName="Muted", alignment=Qt.AlignmentFlag.AlignCenter)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        for w in (self.search, self.list, self.empty):
            lay.addWidget(w)
        self.render()

    def set_rows(self, rows: list[Row]):
        self.rows = rows
        self.render()

    def render(self, *_):
        query = self.search.text()
        self.list.clear()
        group = None
        shown = [r for r in self.rows if r.matches(query)]
        for row in shown:
            if row.group and row.group != group:
                group = row.group
                head = QListWidgetItem(group)
                head.setFlags(Qt.ItemFlag.NoItemFlags)
                self.list.addItem(head)
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, row.key)
            self.list.addItem(item)
            label = QLabel(self._html(row, query))
            label.setTextFormat(Qt.TextFormat.RichText)
            label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            label.setContentsMargins(6, 3, 6, 3)
            item.setSizeHint(label.sizeHint())
            self.list.setItemWidget(item, label)
        self.empty.setVisible(not shown)
        self.list.setVisible(bool(shown))
        self.empty.setText(f"No {self.noun} match ‘{query.strip()}’." if shown == [] and query.strip()
                           else f"No {self.noun}.")

    @staticmethod
    def _html(row: Row, query: str) -> str:
        parts = [f"{row.icon} " if row.icon else "", f"<b>{highlight(row.title, query)}</b>" if not query.strip()
                 else highlight(row.title, query)]
        if row.sub:
            parts.append(f" <span style='opacity:0.7'>{highlight(row.sub, query)}</span>")
        if row.badge:
            parts.append(f" <small style='opacity:0.6'>· {escape(row.badge)}</small>")
        return "".join(parts)

    def _clicked(self, item: QListWidgetItem):
        key = item.data(Qt.ItemDataRole.UserRole)
        if key is not None:
            self.chosen.emit(key)

    def visible_keys(self) -> list[str]:
        return [self.list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.list.count())
                if self.list.item(i).data(Qt.ItemDataRole.UserRole) is not None]


class SourceDialog(QDialog):
    """Pick the one source to scan: a watchlist or a market index. Choosing a row closes the dialog."""

    def __init__(self, lists: dict[str, list[str]], counts: dict[str, int], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Scan source")
        self.setMinimumSize(420, 440)
        self.choice: tuple[str, str] | None = None
        self.watch = SearchList("watchlists", [Row(n, n, f"{len(s)} symbols", icon="⭐") for n, s in lists.items()])
        self.index = SearchList("indices", [
            Row(k, i.ticker, i.title, icon=i.icon, badge=f"{counts[k]} symbols" if k in counts else "",
                extra=f"{i.ticker} {i.title}") for k, i in universes.INDICES.items()])
        self.tabs = QTabWidget()
        self.tabs.addTab(self.watch, "Watchlists")
        self.tabs.addTab(self.index, "Indices")
        self.tabs.currentChanged.connect(lambda _: (self.watch.search.clear(), self.index.search.clear()))
        self.watch.chosen.connect(lambda k: self._pick("watchlist", k))
        self.index.chosen.connect(lambda k: self._pick("index", k))
        QVBoxLayout(self).addWidget(self.tabs)

    def _pick(self, kind: str, key: str):
        self.choice = (kind, key)
        self.accept()


class IndicatorDialog(QDialog):
    """Pick the indicator: My scripts, Built-in and Community, searchable."""

    def __init__(self, classes: list[type[BaseIndicator]], hidden: dict[str, str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Scan indicator")
        self.setMinimumSize(480, 480)
        self.choice: str | None = None
        order = {"My scripts": 0, "Built-in": 1, "Community": 2}
        rows = [Row(c.name, pretty(c.name), c.description, group=indicator_group(c),
                    badge=f"{len(plot_outputs(c))} plots · {len(c.alerts)} alerts", extra=c.name)
                for c in sorted(classes, key=lambda c: (order[indicator_group(c)], c.name))]
        self.picker = SearchList("indicators", rows)
        self.picker.chosen.connect(self._pick)
        lay = QVBoxLayout(self)
        lay.addWidget(self.picker)
        if hidden:
            foot = QLabel(NOT_SCREENABLE_FOOTER.format(n=len(hidden), s="" if len(hidden) == 1 else "s"))
            foot.setObjectName("Muted")
            foot.setToolTip(tooltip("\n".join(f"{n}: {why}" for n, why in sorted(hidden.items()))))
            lay.addWidget(foot)

    def _pick(self, key: str):
        self.choice = key
        self.accept()


class FilterDialog(QDialog):
    """Add or edit one filter: a plot value (against a number or another output) or an alert state."""

    def __init__(self, cls: type[BaseIndicator], chip: dict | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Filter")
        self.cls = cls
        outputs = [o for o in cls.outputs if o not in cls.alerts]
        self.kind = QComboBox()
        self.kind.addItems(["Plot value", "Alert"])
        self.output = QComboBox()
        self.output.addItems(outputs)
        self.op = QComboBox()
        for op, label in OP_LABELS.items():
            self.op.addItem(label, op)
        self.rhs = QComboBox()
        self.rhs.addItem("number")
        self.rhs.addItems(outputs)
        self.number = QDoubleSpinBox()
        self.number.setRange(-1e12, 1e12)
        self.number.setDecimals(4)
        self.rhs.currentTextChanged.connect(lambda t: self.number.setVisible(t == "number"))
        self.alert = QComboBox()
        for name, text in cls.alerts.items():
            self.alert.addItem(text, name)
        self.state = QComboBox()
        self.state.addItem("is true", True)
        self.state.addItem("is false", False)
        form = QFormLayout()
        form.addRow("Kind", self.kind)
        self.value_rows = [("Output", self.output), ("Operator", self.op), ("Compare with", self.rhs), ("", self.number)]
        self.alert_rows = [("Alert", self.alert), ("State", self.state)]
        for label, w in self.value_rows + self.alert_rows:
            form.addRow(label, w)
        self.form = form
        self.kind.currentIndexChanged.connect(self._kind_changed)
        if not cls.alerts:
            self.kind.model().item(1).setEnabled(False)
        if not outputs:
            self.kind.model().item(0).setEnabled(False)
            self.kind.setCurrentIndex(1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(buttons)
        if chip:
            self._fill(chip)
        self._kind_changed()

    def _kind_changed(self):
        alert = self.kind.currentIndex() == 1
        for _, w in self.value_rows:
            if label := self.form.labelForField(w):
                label.setVisible(not alert)
            w.setVisible(not alert and (w is not self.number or self.rhs.currentText() == "number"))
        for _, w in self.alert_rows:
            if label := self.form.labelForField(w):
                label.setVisible(alert)
            w.setVisible(alert)

    def _fill(self, chip: dict):
        if "alert" in chip:
            self.kind.setCurrentIndex(1)
            self.alert.setCurrentIndex(max(0, self.alert.findData(chip["alert"])))
            self.state.setCurrentIndex(0 if chip.get("state", True) else 1)
            return
        self.output.setCurrentText(chip["output"])
        self.op.setCurrentIndex(max(0, self.op.findData(chip["op"])))
        if isinstance(chip["value"], str):
            self.rhs.setCurrentText(chip["value"])
        else:
            self.number.setValue(float(chip["value"]))

    def chip(self) -> dict:
        if self.kind.currentIndex() == 1:
            return {"alert": self.alert.currentData(), "state": bool(self.state.currentData())}
        value = self.number.value() if self.rhs.currentText() == "number" else self.rhs.currentText()
        return {"output": self.output.currentText(), "op": self.op.currentData(), "value": value}


# -- the page ------------------------------------------------------------------------------------------
class ScannerView(QWidget):
    status = pyqtSignal(str)
    #: chart this symbol: (symbol, timeframe, indicator name, params)
    open_chart = pyqtSignal(str, str, str, object)
    _progress = pyqtSignal(int, int)

    def __init__(self, engine, themes: ThemeManager, lists: Callable[[], dict[str, list[str]]],
                 settings: Settings | None = None, scan_fn=None, index_fn=None):
        super().__init__()
        self.themes, self.lists, self.settings = themes, lists, settings
        self.scan_fn = scan_fn or engine_scan.scan
        cache = getattr(getattr(engine, "disk", None), "symbols_dir", None)
        self.index_fn = index_fn or (lambda key: universes.index_symbols(key, cache))
        self.index_counts: dict[str, int] = {}
        self.symbols: list[str] = []
        self.result: engine_scan.ScanResult | None = None
        self.last_sig = None
        self.scanning = False
        self.notice = ""
        self._cancel = threading.Event()
        self._load_token = 0
        self.model = DataFrameModel()

        self._build()
        self._restore()

    # -- layout -------------------------------------------------------------------------------------
    def _build(self):
        title = QLabel("Scanner", objectName="Title")
        self.source_btn = QPushButton(objectName="Pill", toolTip="Which symbols to scan")
        self.source_btn.clicked.connect(self.choose_source)
        self.indicator_btn = QPushButton(objectName="Pill", toolTip="The indicator to scan with")
        self.indicator_btn.clicked.connect(self.choose_indicator)
        self.params_btn = QToolButton(text="⚙", toolTip="Indicator inputs")
        self.params_btn.clicked.connect(self.edit_params)
        self.tf = QComboBox(toolTip="Candle timeframe (the last 500 candles are scanned)")
        for key, label in TF_LABELS.items():
            self.tf.addItem(label, key)
        self.tf.currentIndexChanged.connect(self._tf_changed)
        self.add_column_btn = QPushButton("＋ Column")
        self.add_column_btn.clicked.connect(lambda: self.column_menu().exec(self.add_column_btn.mapToGlobal(
            self.add_column_btn.rect().bottomLeft())))
        self.scan_btn = QPushButton("Scan", objectName="Primary")
        self.scan_btn.clicked.connect(self.start_scan)
        bar = QHBoxLayout()
        for w in (self.source_btn, self.indicator_btn, self.params_btn, self.tf, self.add_column_btn):
            bar.addWidget(w)
        bar.addStretch()
        bar.addWidget(self.scan_btn)

        self.scan_label = QLabel()
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self._cancel.set)
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(240)
        self.scan_bar = QFrame()
        sl = QHBoxLayout(self.scan_bar)
        sl.setContentsMargins(0, 0, 0, 0)
        for w in (self.scan_label, self.progress, self.stop_btn):
            sl.addWidget(w)
        sl.insertStretch(1)
        self.scan_bar.hide()
        self._progress.connect(self._on_progress)

        self.chips_box = QHBoxLayout()
        self.chips_box.setContentsMargins(0, 0, 0, 0)
        self.add_filter_btn = QPushButton("＋ Filter")
        self.add_filter_btn.clicked.connect(self.add_filter)
        filters = QHBoxLayout()
        filters.addLayout(self.chips_box)
        filters.addWidget(self.add_filter_btn)
        filters.addStretch()
        self.columns_btn = QToolButton(text="☰", toolTip="Show or hide columns")
        self.columns_btn.clicked.connect(lambda: self.column_menu().exec(self.columns_btn.mapToGlobal(
            self.columns_btn.rect().bottomLeft())))
        filters.addWidget(self.columns_btn)

        self.notification = QFrame(objectName="Card")
        nl = QHBoxLayout(self.notification)
        nl.setContentsMargins(10, 4, 10, 4)
        self.notification_label = QLabel("Settings changed.")
        rescan = QPushButton("Rescan", objectName="Primary")
        rescan.clicked.connect(self.start_scan)
        self.rescan_btn = rescan
        nl.addWidget(self.notification_label)
        nl.addWidget(rescan)
        nl.addStretch()
        self.notification.hide()

        from tradebot.gui.views import _table

        self.table = _table(self.model)
        self.table.doubleClicked.connect(self._row_opened)

        self.summary = QLabel()
        self.skipped_btn = QPushButton(flat=True)
        self.skipped_btn.clicked.connect(self.show_skipped)
        sm = QHBoxLayout()
        sm.addWidget(self.summary)
        sm.addWidget(self.skipped_btn)
        sm.addStretch()
        self.summary_bar = QFrame()
        self.summary_bar.setLayout(sm)
        self.summary_bar.hide()

        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.addWidget(title)
        lay.addLayout(bar)
        lay.addWidget(self.scan_bar)
        lay.addLayout(filters)
        lay.addWidget(self.notification)
        lay.addWidget(self.table, 1)
        lay.addWidget(self.summary_bar)
        hint = QLabel("Columns are the indicator's own outputs · double-click a row to chart it", objectName="Muted")
        lay.addWidget(hint)
        self.themes.changed.connect(lambda _: self._style_stale())

    # -- settings -----------------------------------------------------------------------------------
    def _get(self, key, default=None):
        return self.settings.get(f"scanner.{key}", default) if self.settings else default

    def _save(self, key, value):
        if self.settings:
            self.settings.set(f"scanner.{key}", value)

    def _remember(self, key: str, value):
        """Per-indicator state (params, filters, columns)."""
        data = dict(self._get(key, {}) or {})
        data[self.cls.name] = value
        self._save(key, data)

    def _restore(self):
        name = self._get("indicator", DEFAULT_INDICATOR)
        shown, _ = screenable_split()
        self.cls = INDICATORS.get(name) if name in INDICATORS and INDICATORS.get(name) in shown \
            else INDICATORS.get(DEFAULT_INDICATOR)
        tf = self._get("timeframe", DEFAULT_TIMEFRAME)
        self.tf.blockSignals(True)
        self.tf.setCurrentIndex(max(0, self.tf.findData(tf)))
        self.tf.blockSignals(False)
        self._load_indicator_state()
        source = self._get("source") or {}
        self._set_source(source.get("kind", "watchlist"), source.get("key", ""), restoring=True)
        if self.notice:
            QTimer.singleShot(0, lambda: self.status.emit(self.notice))

    # -- indicator state ----------------------------------------------------------------------------
    @property
    def params(self) -> dict:
        saved = (self._get("params", {}) or {}).get(self.cls.name, {})
        try:
            return resolve_params(self.cls.params, saved)
        except (ValueError, TypeError):
            return resolve_params(self.cls.params, {})

    def _load_indicator_state(self):
        cls = self.cls
        plots = plot_outputs(cls)
        cols = (self._get("columns", {}) or {}).get(cls.name)
        valid = [c for c in cols if c in cls.outputs] if isinstance(cols, list) else None
        self.columns = valid if valid is not None else plots[:MAX_AUTO_COLUMNS]
        chips = (self._get("filters", {}) or {}).get(cls.name)
        if isinstance(chips, list):
            self.chips = [c for c in chips if self._chip_valid(c)]
        else:  # the first two alerts become templates, off until clicked
            self.chips = [{"alert": a, "state": True, "template": True, "on": False} for a in list(cls.alerts)[:2]]
        self._update_indicator_pill()
        self._render_chips()

    def _chip_valid(self, chip) -> bool:
        if not isinstance(chip, dict):
            return False
        if "alert" in chip:
            return chip["alert"] in self.cls.alerts
        return chip.get("output") in self.cls.outputs and chip.get("op") in engine_scan.OPS and (
            chip.get("value") in self.cls.outputs if isinstance(chip.get("value"), str)
            else isinstance(chip.get("value"), (int, float)))

    def _update_indicator_pill(self):
        args = ", ".join(f"{v:g}" for v in self.params.values())
        self.indicator_btn.setText(f"ƒ {pretty(self.cls.name)}" + (f" ({args})" if args else ""))
        self.params_btn.setEnabled(bool(self.cls.params))

    def _set_indicator(self, name: str):
        if name == self.cls.name:
            return
        self.cls = INDICATORS.get(name)
        self._save("indicator", name)
        self._load_indicator_state()
        self.result = None  # other outputs: the old rows mean nothing
        self._refresh_table()
        self._check_stale()

    def choose_indicator(self):
        classes, hidden = screenable_split()
        dlg = IndicatorDialog(classes, hidden, self)
        if dlg.exec() and dlg.choice:
            self._set_indicator(dlg.choice)

    def edit_params(self):
        values = ParamDialog.ask(pretty(self.cls.name), self.cls.params, self.params, self)
        if values is not None:
            self._remember("params", resolve_params(self.cls.params, values))
            self._update_indicator_pill()
            self._check_stale()

    def _tf_changed(self, *_):
        self._save("timeframe", self.tf.currentData())
        self._check_stale()

    @property
    def timeframe(self) -> str:
        return self.tf.currentData()

    # -- source -------------------------------------------------------------------------------------
    def default_watchlist(self) -> str:
        return next(iter(self.lists()), "")

    def choose_source(self):
        dlg = SourceDialog(self.lists(), self.index_counts, self)
        if dlg.exec() and dlg.choice:
            self._set_source(*dlg.choice)

    def _set_source(self, kind: str, key: str, restoring: bool = False):
        lists = self.lists()
        gone = kind == "index" and key not in universes.INDICES or kind != "index" and key not in lists
        if gone:
            fallback = self.default_watchlist()
            if key or not restoring:
                self.notice = f"Scanner: ‘{key}’ no longer exists, using {fallback}"
                if not restoring:
                    self.status.emit(self.notice)
            kind, key = "watchlist", fallback
        self.source = (kind, key)
        self._save("source", {"kind": kind, "key": key})
        self.result = None
        self._load_token += 1
        if kind == "watchlist":
            self.symbols = list(lists.get(key, []))
            self._source_loaded()
        else:
            self.symbols = []
            self._source_loaded()
            token = self._load_token
            run_async(lambda: self.index_fn(key), lambda s: self._index_loaded(token, key, s),
                      lambda err: self._index_failed(token, key, err))

    def _index_loaded(self, token: int, key: str, symbols: list[str]):
        if token != self._load_token:
            return  # the user picked another source meanwhile
        self.index_counts[key] = len(symbols)
        self.symbols = list(symbols)
        self._source_loaded()

    def _index_failed(self, token: int, key: str, err):
        if token == self._load_token:
            fallback = self.default_watchlist()
            self.notice = f"Scanner: {universes.INDICES[key].title} can't be loaded, using {fallback}"
            self.status.emit(self.notice)
            self._set_source("watchlist", fallback, restoring=True)

    def _source_loaded(self):
        kind, key = self.source
        if kind == "watchlist":
            self.source_btn.setText(f"⭐ {key} · {len(self.symbols)} symbols")
        else:
            index = universes.INDICES[key]
            count = f" · {len(self.symbols)}" if self.symbols else " · loading …"
            self.source_btn.setText(f"{index.icon} {index.title}{count}")
        self._refresh_table()
        self._check_stale()

    def watchlists_changed(self):
        """The sidebar's lists changed: a scanned watchlist follows it."""
        kind, key = self.source
        if kind != "watchlist" or self.scanning:
            return
        lists = self.lists()
        if key in lists:
            if lists[key] != self.symbols:
                self.symbols = list(lists[key])
                self._source_loaded()
        else:
            self._set_source("watchlist", key)

    # -- filters ------------------------------------------------------------------------------------
    def active_filters(self) -> list[dict]:
        out = []
        for chip in self.chips:
            if "alert" in chip:
                if not chip.get("template") or chip.get("on"):
                    out.append({"alert": chip["alert"], "state": chip.get("state", True)})
            else:
                out.append({k: chip[k] for k in ("output", "op", "value")})
        return out

    def chip_text(self, chip: dict) -> str:
        if "alert" in chip:
            text = self.cls.alerts[chip["alert"]]
            if chip.get("template"):
                return f"{'☑' if chip.get('on') else '◻'} {text}"
            return f"{text}: {'true' if chip.get('state', True) else 'false'}"
        value = chip["value"] if isinstance(chip["value"], str) else f"{chip['value']:g}"
        return f"{chip['output']} {OP_LABELS[chip['op']]} {value}"

    def _render_chips(self):
        while self.chips_box.count():
            w = self.chips_box.takeAt(0).widget()
            if w:
                w.deleteLater()
        self.chip_widgets = []
        for chip in self.chips:
            frame = QFrame(objectName="Chip")
            frame.setFrameShape(QFrame.Shape.StyledPanel)
            row = QHBoxLayout(frame)
            row.setContentsMargins(6, 0, 2, 0)
            main = QToolButton(text=self.chip_text(chip), autoRaise=True)
            main.clicked.connect(lambda _, c=chip: self._chip_clicked(c))
            close = QToolButton(text="✕", autoRaise=True, toolTip="Remove this filter")
            close.clicked.connect(lambda _, c=chip: self._remove_chip(c))
            row.addWidget(main)
            row.addWidget(close)
            self.chips_box.addWidget(frame)
            self.chip_widgets.append((chip, main, close))

    def _filters_changed(self):
        self._remember("filters", self.chips)
        self._render_chips()
        self._check_stale()

    def _chip_clicked(self, chip: dict):
        if chip.get("template"):
            chip["on"] = not chip.get("on")
        else:
            dlg = FilterDialog(self.cls, chip, self)
            if not dlg.exec():
                return
            chip.clear()
            chip.update(dlg.chip())
        self._filters_changed()

    def _remove_chip(self, chip: dict):
        self.chips.remove(chip)
        self._filters_changed()

    def add_filter(self):
        dlg = FilterDialog(self.cls, parent=self)
        if dlg.exec():
            self.chips.append(dlg.chip())
            self._filters_changed()

    # -- table --------------------------------------------------------------------------------------
    def column_menu(self) -> QMenu:
        menu = QMenu(self)
        for output in self.cls.outputs:
            act = menu.addAction(output + (" (alert)" if output in self.cls.alerts else ""))
            act.setCheckable(True)
            act.setChecked(output in self.columns)
            act.toggled.connect(lambda on, o=output: self.set_column(o, on))
        return menu

    def set_column(self, output: str, on: bool):
        cols = set(self.columns) | {output} if on else set(self.columns) - {output}
        self.columns = [o for o in self.cls.outputs if o in cols]
        self._remember("columns", self.columns)
        self._refresh_table()

    def _refresh_table(self):
        cols = list(self.columns)
        if self.result is not None:
            frame = self.result.table[["symbol", *cols]]
        else:
            frame = pd.DataFrame({"symbol": self.symbols})
            for c in cols:
                frame[c] = np.nan
        self.model.formats = {c: formatter(self.cls, c) for c in cols}
        self.model.set_frame(frame)
        self._style_stale()

    def _row_opened(self, index):
        symbol = self.model.df.iat[index.row(), 0]
        self.open_chart.emit(symbol, self.timeframe, self.cls.name, self.params)

    def _style_stale(self):
        stale = self.result is not None and self.last_sig is not None and self._signature() != self.last_sig
        self.table.setStyleSheet(f"QTableView {{ color: {self.themes.current.muted}; }}" if stale else "")

    # -- scan ---------------------------------------------------------------------------------------
    def _signature(self):
        return (self.source, hash(tuple(self.symbols)), self.cls.name, tuple(sorted(self.params.items())),
                self.timeframe, repr(self.active_filters()))

    def _check_stale(self):
        stale = self.last_sig is not None and not self.scanning and self._signature() != self.last_sig
        self.notification.setVisible(stale)
        self._style_stale()

    def _scan_text(self) -> str:
        kind, key = self.source
        name = universes.INDICES[key].title if kind == "index" else key
        args = ", ".join(f"{v:g}" for v in self.params.values())
        ind = pretty(self.cls.name) + (f" ({args})" if args else "")
        return f"Scanning {name} with {ind} on {TF_LABELS[self.timeframe]} …"

    def start_scan(self):
        if self.scanning:
            return
        if not self.symbols:
            self.status.emit("Scanner: no symbols to scan")
            return
        symbols, name, params, tf = list(self.symbols), self.cls.name, self.params, self.timeframe
        filters, sig, cls = self.active_filters(), self._signature(), self.cls
        self._cancel.clear()
        self.scanning = True
        self.scan_btn.setEnabled(False)
        self.rescan_btn.setEnabled(False)
        self.notification.hide()
        self.summary_bar.hide()
        self.scan_label.setText(self._scan_text())
        self.stop_btn.setEnabled(True)
        self.progress.setRange(0, len(symbols))
        self.progress.setValue(0)
        self.progress.hide()  # a fast scan never flashes a bar
        self.scan_bar.show()
        generation = self._load_token
        QTimer.singleShot(PROGRESS_AFTER_MS, lambda: self.scanning and self.progress.show())

        def work():
            try:
                return "ok", self.scan_fn(symbols, name, params, tf, filters, progress=self._progress.emit,
                                          cancel=self._cancel.is_set)
            except engine_scan.ScanBusy:
                return "busy", None

        run_async(work, lambda r: self._scan_done(r, sig, cls, symbols, generation),
                  lambda err: self._scan_failed(err))

    def _on_progress(self, done: int, total: int):
        self.progress.setRange(0, total)
        self.progress.setValue(done)

    def _end_scan(self):
        self.scanning = False
        self.scan_btn.setEnabled(True)
        self.rescan_btn.setEnabled(True)
        self.progress.hide()

    def _scan_failed(self, err):
        self._end_scan()
        self.scan_label.setText(f"Scan failed: {err}")
        self.stop_btn.setEnabled(False)
        self.status.emit(f"Scanner: {err}")

    def _scan_done(self, outcome, sig, cls, symbols, generation):
        kind, result = outcome
        self._end_scan()
        if kind == "busy":
            self.scan_label.setText("Another scan is running")
            self.stop_btn.setEnabled(False)
            return
        self.scan_bar.hide()
        if cls is not self.cls or generation != self._load_token:
            self._check_stale()  # the source or indicator changed while scanning: the rows don't fit
            return
        self.result, self.last_sig = result, sig
        self._refresh_table()
        total = len(symbols)
        tf = TF_LABELS[self.timeframe]
        if result.cancelled:
            self.summary.setText(f"stopped after {result.scanned + len(result.skipped)} of {total}"
                                 f" · {result.matched} match · {tf}")
        else:
            self.summary.setText(f"{result.scanned} scanned · {result.matched} match · {result.seconds:.1f} s · {tf} ·")
        self.skipped_btn.setText(f"{len(result.skipped)} skipped")
        self.skipped_btn.setVisible(bool(result.skipped))
        self.summary_bar.show()
        self._check_stale()
        self.status.emit(f"Scanner: {result.summary()}")

    def show_skipped(self):
        if self.result is None:
            return
        box = QMessageBox(self)
        box.setWindowTitle("Skipped symbols")
        box.setText(f"{len(self.result.skipped)} symbols were skipped.")
        box.setDetailedText("\n".join(f"{s}: {why}" for s, why in self.result.skipped.items()))
        box.exec()
