# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Bots page (#157): the bot set of ``bots.toml`` as a dashboard, with start/stop, an editor and the shared limits.

The app runs dry-run and paper bots itself, in one background scheduler (:class:`tradebot.live.BotManager`); it never
starts a live bot (they run in a terminal with ``tradebot bots run --live``). Bots another process runs show up from
their state files and can be stopped from here.
"""

from __future__ import annotations

import html
from dataclasses import replace

import pandas as pd
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
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
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableView,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from tradebot.config import Settings
from tradebot.core.timeframes import TIMEFRAMES
from tradebot.gui.common import DataFrameModel, Kpi, Panes, SymbolCompleter, TableWithEmpty, kpi_strip, up_down_mark
from tradebot.gui.icons import bind_icon
from tradebot.gui.theme import ThemeManager, qcolor
from tradebot.live.manager import AccountLimits, BotManager, BotSet, BotSpec, load_bots, request_stop, save_bots

AUTOSTART_KEY = "bots.autostart"
MODE_LABELS = {"dry": "dry run", "paper": "paper", "live": "live"}
STATE_LABELS = {"running": "running", "stopped": "stopped", "halted": "halted", "error": "error",
                "elsewhere": "other process"}
COLUMNS = ["bot", "symbol", "strategy", "tf", "mode", "state", "position", "entry", "last", "P&L", "last event", "at"]
EVENT_COLUMNS = ["time", "bot", "action", "amount", "price", "note"]


def parse_params(text: str) -> dict[str, float]:
    """``fast=10, slow=30`` (commas or spaces) -> ``{"fast": 10.0, "slow": 30.0}``; raises ValueError."""
    out = {}
    for part in text.replace(",", " ").split():
        key, sep, value = part.partition("=")
        if not sep or not key:
            raise ValueError(f"parameter {part!r}: write name=value")
        try:
            out[key.strip()] = float(value)
        except ValueError:
            raise ValueError(f"parameter {key}: {value!r} is not a number") from None
    return out


def _money(v: float) -> str:
    return f"{v:,.2f}"


class BotsView(QWidget):
    status = pyqtSignal(str)
    #: the bots this app runs changed: their modes ("dry", "paper"), empty when none runs
    running_changed = pyqtSignal(list)

    def __init__(self, themes: ThemeManager, settings: Settings | None = None, path=None, manager_factory=None,
                 symbol_search=None, journal_factory=None, interval: int = 1000):
        super().__init__()
        self.themes, self.settings, self.path = themes, settings, path
        self._factory = manager_factory or (lambda bots: BotManager(bots, journal_factory=journal_factory))
        self._load_error = None
        self.bots = self._load()
        self.manager: BotManager = self._factory(self.bots)
        self._editing: str | None = None  #: the bot in the editor (None: a new one)
        self._rows: dict[str, object] = {}
        self._last_frame: pd.DataFrame | None = None
        self._last_modes: list[str] = []

        # toolbar
        self.start_btn = QPushButton("Start", toolTip="Start the selected bot (dry-run and paper bots only)")
        self.stop_btn = QPushButton("Stop", toolTip="Stop the selected bot, also when another process runs it")
        self.start_all_btn = QPushButton("Start all", toolTip="Start every enabled dry-run and paper bot")
        self.stop_all_btn = QPushButton("Stop all", toolTip="Stop every bot this app runs")
        self.new_btn = QPushButton("New bot", objectName="Primary")
        for btn, icon in ((self.start_btn, "play"), (self.stop_btn, "stop"), (self.start_all_btn, "play"),
                          (self.stop_all_btn, "stop")):
            bind_icon(btn, icon, themes, off="text")
        bind_icon(self.new_btn, "plus", themes, off="accent_ink")
        self.start_btn.clicked.connect(self.start_selected)
        self.stop_btn.clicked.connect(self.stop_selected)
        self.start_all_btn.clicked.connect(self.start_all)
        self.stop_all_btn.clicked.connect(self.stop_all)
        self.new_btn.clicked.connect(self.new_bot)
        title = QLabel("Bots", objectName="Title")
        top = QHBoxLayout()
        top.setSpacing(8)
        top.addWidget(title)
        top.addSpacing(8)
        for w in (self.start_btn, self.stop_btn):
            top.addWidget(w)
        top.addSpacing(12)
        for w in (self.start_all_btn, self.stop_all_btn):
            top.addWidget(w)
        top.addStretch()
        top.addWidget(self.new_btn)
        self.note = QLabel("This app runs dry-run and paper bots; neither sends a real order. Live bots run in a "
                           "terminal: <code>tradebot bots run --live</code>.", objectName="Muted", wordWrap=True)
        if self._load_error:
            self.note.setText(f'<span style="color:{themes.current.down}">bots.toml could not be read: '
                              f'{html.escape(self._load_error)}</span>')

        # account figures
        self.kpis = {k: Kpi(label) for k, label in (
            ("running", "Running"), ("positions", "Open positions"), ("exposure", "Exposure"),
            ("pnl", "P&L, realised + open"), ("limits", "Shared limits"))}
        self.kpis["limits"].value.setObjectName("KpiText")
        self.kpis["limits"].value.setWordWrap(True)
        strip = kpi_strip(self.kpis.values())

        # the bots
        self.model = DataFrameModel(formats={
            "position": lambda v: f"{v:,.6g}", "entry": lambda v: f"{v:,.6g}" if v else "–",
            "last": lambda v: f"{v:,.6g}" if v else "–", "P&L": lambda v: f"{abs(v):,.2f}"},
            colors=self._colour, row_tips=self._row_tip)
        self.table = QTableView()
        self.table.setModel(self.model)
        self.table.setShowGrid(False)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().hide()
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setWordWrap(False)
        self.table.doubleClicked.connect(self.toggle_selected)
        self.bots_box = TableWithEmpty(self.table, self.model, themes, "bot", "No bots yet",
                                       "Press New bot to define one: a symbol, a strategy and a stake. Bots are saved "
                                       "in bots.toml and run as dry runs until you choose paper.")

        # events of the selected bot (or of all)
        self.events_title = QLabel("Events · all bots", objectName="SidebarTitle")
        self.events_model = DataFrameModel(formats={"amount": lambda v: f"{v:,.6g}" if v else "",
                                                    "price": lambda v: f"{v:,.6g}"})
        self.events = QTableView()
        self.events.setModel(self.events_model)
        self.events.setShowGrid(False)
        self.events.verticalHeader().hide()
        self.events.setWordWrap(False)
        self.events.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        events_box = QWidget()
        el = QVBoxLayout(events_box)
        el.setContentsMargins(0, 0, 0, 0)
        el.addWidget(self.events_title)
        el.addWidget(TableWithEmpty(self.events, self.events_model, themes, "clock", "No events yet",
                                    "A running bot adds one event per closed candle: buy, sell or hold."), 1)

        # editor
        self.tabs = QTabWidget()
        self.tabs.addTab(self._bot_form(symbol_search), "Bot")
        self.tabs.addTab(self._limits_form(), "Shared limits")
        self.tabs.setMinimumWidth(360)

        bottom = Panes(Qt.Orientation.Horizontal, "bots.bottom.panes", settings)
        bottom.add(events_box, 3)
        bottom.add(self.tabs, 2, scroll=True)
        split = Panes(Qt.Orientation.Vertical, "bots.panes", settings)
        split.add(self.bots_box, 3)
        split.add(bottom, size=340)  # the editor fits whole; the bots table takes the rest

        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 12)
        lay.setSpacing(10)
        lay.addLayout(top)
        lay.addWidget(self.note)
        lay.addWidget(strip)
        lay.addWidget(split, 1)

        # bound methods, not lambdas: Qt then holds no strong reference, so a closed window can be freed
        self.table.selectionModel().selectionChanged.connect(self._selected)
        themes.changed.connect(self._theme_changed)
        self.timer = QTimer(self, interval=interval)
        self.timer.timeout.connect(self.refresh)  # runs while the page is shown, see showEvent
        self.new_bot()
        self.refresh(force=True)
        if settings and settings.get(AUTOSTART_KEY, False):
            QTimer.singleShot(0, self._autostart)

    # -- forms -------------------------------------------------------------------------------------------
    def _bot_form(self, symbol_search) -> QWidget:
        from tradebot.algorithms import ALGORITHMS
        from tradebot.brokers import BROKERS

        self.f_name = QLineEdit(placeholderText="e.g. eth-trend")
        self.f_symbol = QLineEdit(placeholderText="e.g. ETH/USDT or AAPL")
        if symbol_search is not None:
            self._completer = SymbolCompleter(self.f_symbol, symbol_search)
        self.f_algo = QComboBox()
        self.f_algo.addItems(sorted(cls.name for cls in ALGORITHMS))
        self.f_params = QLineEdit(placeholderText="defaults; e.g. fast=10, slow=30",
                                  toolTip="The strategy's parameters as name=value, as -p in the CLI")
        self.f_tf = QComboBox()
        self.f_tf.addItems(list(TIMEFRAMES))
        self.f_tf.setCurrentText("1h")
        self.f_stake = QDoubleSpinBox(minimum=0.01, maximum=1e9, decimals=2, value=100.0,
                                      toolTip="Maximum quote-currency value of one entry")
        self.f_broker = QComboBox(toolTip="Where the candles come from; with the paper mode, also the prices the "
                                          "simulated account fills at")
        self.f_broker.addItems([n for n in BROKERS.names() if n != "paper"])
        self.f_mode = QComboBox(toolTip="Dry run: nothing is sent. Paper: the simulated account (tradebot paper). "
                                        "Live: real orders, started from a terminal only")
        for key, label in (("dry", "Dry run"), ("paper", "Paper account"), ("live", "Live (terminal only)")):
            self.f_mode.addItem(label, key)
        self.f_daily = QDoubleSpinBox(maximum=1e9, decimals=2, specialValueText="off",
                                      toolTip="This bot halts for the day after losing this much")
        self.f_drawdown = QDoubleSpinBox(maximum=100, decimals=1, suffix=" %", specialValueText="off",
                                         toolTip="This bot halts when its equity falls this far below its peak, in % "
                                                 "of the stake")
        self.f_enabled = QCheckBox("Enabled: Start all starts it", checked=True,
                                   toolTip="Also `tradebot bots run` and the start when the app opens")
        self.save_btn = QPushButton("Save bot", objectName="Primary")
        self.save_btn.clicked.connect(self.save_bot)
        self.delete_btn = QPushButton("Delete")
        bind_icon(self.delete_btn, "trash", self.themes, off="text")
        self.delete_btn.clicked.connect(self.delete_bot)
        form = QGridLayout()  # two label/field pairs per row: the whole bot fits without scrolling
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(8)
        pairs = (("Name", self.f_name), ("Symbol", self.f_symbol), ("Strategy", self.f_algo),
                 ("Parameters", self.f_params), ("Timeframe", self.f_tf), ("Stake", self.f_stake),
                 ("Data / broker", self.f_broker), ("Mode", self.f_mode), ("Max daily loss", self.f_daily),
                 ("Max drawdown", self.f_drawdown))
        for i, (label, w) in enumerate(pairs):
            row, col = divmod(i, 2)
            form.addWidget(QLabel(label, objectName="Muted"), row, col * 2)
            form.addWidget(w, row, col * 2 + 1)
        form.setColumnStretch(1, 1)
        form.setColumnStretch(3, 1)
        form.addWidget(self.f_enabled, len(pairs) // 2, 1, 1, 3)
        buttons = QHBoxLayout()
        buttons.addWidget(self.delete_btn)
        buttons.addStretch()
        buttons.addWidget(self.save_btn)
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(4, 8, 4, 4)
        lay.addLayout(form)
        lay.addLayout(buttons)
        lay.addStretch()
        return page

    def _limits_form(self) -> QWidget:
        self.l_exposure = QDoubleSpinBox(maximum=1e12, decimals=2, specialValueText="off",
                                         toolTip="Entries are rejected while all bots' open positions together, "
                                                 "plus the new one, would be worth more than this")
        self.l_positions = QSpinBox(maximum=1000, specialValueText="off",
                                    toolTip="Entries are rejected while this many bots hold a position")
        self.l_daily = QDoubleSpinBox(maximum=1e12, decimals=2, specialValueText="off",
                                      toolTip="Every bot halts for the rest of the UTC day when all of them together "
                                              "lost this much, realised plus open")
        self.l_close = QCheckBox("Sell the open positions when the daily limit halts the bots")
        self.l_autostart = QCheckBox("Start the enabled bots when the app opens",
                                     checked=bool(self.settings.get(AUTOSTART_KEY, False)) if self.settings else False)
        self.l_autostart.toggled.connect(self._autostart_toggled)
        save = QPushButton("Save limits", objectName="Primary")
        save.clicked.connect(self.save_limits)
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.addRow("Max total exposure", self.l_exposure)
        form.addRow("Max open positions", self.l_positions)
        form.addRow("Max daily loss, all bots", self.l_daily)
        form.addRow("", self.l_close)
        hint = QLabel("Every bot also stops at the kill switch in the status bar.", objectName="Muted", wordWrap=True)
        row = QHBoxLayout()
        row.addStretch()
        row.addWidget(save)
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(4, 8, 4, 4)
        lay.addLayout(form)
        lay.addWidget(hint)
        lay.addLayout(row)
        lay.addSpacing(12)
        lay.addWidget(self.l_autostart)
        lay.addStretch()
        self._show_limits()
        return page

    def _show_limits(self):
        lim = self.bots.limits
        self.l_exposure.setValue(lim.max_total_exposure or 0)
        self.l_positions.setValue(lim.max_open_positions or 0)
        self.l_daily.setValue(lim.max_daily_loss or 0)
        self.l_close.setChecked(lim.close_on_halt)

    # -- the set -----------------------------------------------------------------------------------------
    def _load(self) -> BotSet:
        try:
            return load_bots(self.path)
        except ValueError as exc:
            self._load_error = str(exc)
            return BotSet()

    def _store(self, bots: BotSet) -> bool:
        if self._load_error:  # never overwrite a file the user has to fix by hand
            QMessageBox.warning(self, "Bots", f"bots.toml could not be read, so it is not overwritten:\n\n"
                                              f"{self._load_error}")
            return False
        try:
            save_bots(bots, self.path)
        except (OSError, ValueError) as exc:
            self.status.emit(f"bots: not saved: {exc}")
            return False
        was = set(self.manager.running())
        self.bots = bots
        self.manager.reload(bots)
        stopped = was - set(self.manager.running())
        if stopped:
            self.status.emit(f"bots: {', '.join(sorted(stopped))} stopped because its settings changed; "
                             f"start it again")
        self.refresh(force=True)
        return True

    def new_bot(self):
        self._editing = None
        self.table.clearSelection()
        self.f_name.clear()
        self.f_symbol.clear()
        self.f_params.clear()
        self.f_mode.setCurrentIndex(0)
        self.f_daily.setValue(0)
        self.f_drawdown.setValue(0)
        self.f_enabled.setChecked(True)
        self.delete_btn.setEnabled(False)
        self.save_btn.setText("Add bot")
        self.tabs.setCurrentIndex(0)
        self.f_name.setFocus()

    def _edit(self, spec: BotSpec):
        self._editing = spec.name
        self.f_name.setText(spec.name)
        self.f_symbol.setText(spec.symbol)
        if self.f_algo.findText(spec.algorithm) < 0:
            self.f_algo.addItem(spec.algorithm)
        self.f_algo.setCurrentText(spec.algorithm)
        self.f_params.setText(", ".join(f"{k}={v:g}" for k, v in spec.params.items()))
        self.f_tf.setCurrentText(spec.timeframe)
        self.f_stake.setValue(spec.stake)
        if self.f_broker.findText(spec.broker) < 0:
            self.f_broker.addItem(spec.broker)
        self.f_broker.setCurrentText(spec.broker)
        self.f_mode.setCurrentIndex(max(0, self.f_mode.findData(spec.mode)))
        self.f_daily.setValue(spec.risk.get("max_daily_loss", 0) or 0)
        self.f_drawdown.setValue(spec.risk.get("max_drawdown", 0) or 0)
        self.f_enabled.setChecked(spec.enabled)
        self.delete_btn.setEnabled(True)
        self.save_btn.setText("Save bot")

    def _spec_from_form(self) -> BotSpec:
        old = next((b for b in self.bots.bots if b.name == self._editing), None)
        risk = dict(old.risk) if old else {}
        for key, box in (("max_daily_loss", self.f_daily), ("max_drawdown", self.f_drawdown)):
            if box.value():
                risk[key] = box.value()
            else:
                risk.pop(key, None)
        fields = dict(name=self.f_name.text(), symbol=self.f_symbol.text(), algorithm=self.f_algo.currentText(),
                      broker=self.f_broker.currentText(), timeframe=self.f_tf.currentText(),
                      stake=self.f_stake.value(), mode=self.f_mode.currentData(),
                      params=parse_params(self.f_params.text()), enabled=self.f_enabled.isChecked(), risk=risk)
        spec = replace(old, **fields) if old else BotSpec(**fields)  # keeps members, rules, sizing of the file
        spec.build_algorithm()  # unknown parameters fail here, not when it starts
        return spec

    def save_bot(self) -> bool:
        try:
            spec = self._spec_from_form()
            bots = BotSet(list(self.bots.bots), self.bots.limits)
            if self._editing is None and any(b.name == spec.name for b in bots.bots):
                raise ValueError(f"there already is a bot named {spec.name!r}")
            bots.put(spec, old_name=self._editing)
        except (KeyError, ValueError, TypeError) as exc:
            msg = exc.args[0] if isinstance(exc, KeyError) and exc.args else str(exc)
            self.status.emit(f"bots: {msg}")
            QMessageBox.warning(self, "Bots", str(msg))
            return False
        if not self._store(bots):
            return False
        self.status.emit(f"bots: {spec.name} saved")
        self._select(spec.name)
        self._edit(spec)
        return True

    def delete_bot(self):
        name = self._editing
        if name is None:
            return
        if QMessageBox.question(self, "Delete bot", f"Delete {name} from bots.toml? A running bot is stopped first."
                                ) != QMessageBox.StandardButton.Yes:
            return
        bots = BotSet([b for b in self.bots.bots if b.name != name], self.bots.limits)
        if self._store(bots):
            self.status.emit(f"bots: {name} deleted")
            self.new_bot()

    def save_limits(self) -> bool:
        limits = AccountLimits(self.l_exposure.value() or None, self.l_positions.value(), self.l_daily.value() or None,
                               self.l_close.isChecked())
        if self._store(BotSet(list(self.bots.bots), limits)):
            self.status.emit(f"bots: shared limits saved ({limits.describe()})")
            return True
        return False

    # -- start / stop ------------------------------------------------------------------------------------
    def selected(self) -> str | None:
        rows = self.table.selectionModel().selectedRows()
        return str(self.model.df.iloc[rows[0].row()]["bot"]) if rows else None

    def _select(self, name: str):
        for i, n in enumerate(self.model.df.get("bot", [])):
            if n == name:
                self.table.selectRow(i)
                return

    def start(self, name: str) -> bool:
        try:
            self.manager.start(name)
        except Exception as exc:
            self.status.emit(f"bots: {exc}")
            return False
        self.manager.start_loop()
        self.status.emit(f"bots: {name} started")
        self.refresh(force=True)
        return True

    def start_selected(self):
        name = self.selected()
        if name:
            self.start(name)

    def stop_selected(self):
        name = self.selected()
        if not name:
            return
        if self.manager.is_running(name):
            self.manager.stop(name)
            self.status.emit(f"bots: {name} stopped")
        elif self.manager.elsewhere(name):
            request_stop(name, self.manager.state_dir)
            self.status.emit(f"bots: asked the process that runs {name} to stop it")
        self.refresh(force=True)

    def _theme_changed(self, _theme=None):
        self.refresh(force=True)

    def _autostart(self):
        self.start_all(quiet=True)

    def _autostart_toggled(self, on: bool):
        if self.settings:
            self.settings.set(AUTOSTART_KEY, on)

    def toggle_selected(self, *_):
        name = self.selected()
        if name and (self.manager.is_running(name) or self.manager.elsewhere(name)):
            self.stop_selected()
        elif name:
            self.start(name)

    def start_all(self, quiet: bool = False):
        wanted = [b for b in self.bots.bots if b.enabled and b.mode != "live"]
        failed = {n: e for n, e in self.manager.start_all().items() if self.bots.get(n).mode != "live"}
        if self.manager.running():
            self.manager.start_loop()
        skipped = sum(1 for b in self.bots.bots if b.enabled and b.mode == "live")
        parts = [f"{len(wanted) - len(failed)} of {len(wanted)} started"]
        if failed:
            parts.append("; ".join(f"{n}: {e}" for n, e in failed.items()))
        if skipped:
            parts.append(f"{skipped} live bot(s) left to the terminal")
        if not quiet or failed:
            self.status.emit("bots: " + " · ".join(parts))
        self.refresh(force=True)

    def stop_all(self):
        self.manager.stop_all()
        self.status.emit("bots: all stopped")
        self.refresh(force=True)

    def showEvent(self, event):
        super().showEvent(event)
        self.refresh()
        self.timer.start()

    def hideEvent(self, event):
        super().hideEvent(event)
        self.timer.stop()  # a hidden page costs nothing; start/stop still refresh it (and the chip) directly

    @staticmethod
    def _set(kpi, text: str):
        if kpi.value.text() != text:  # an unchanged label must not relayout the window every second
            kpi.set(text)

    def shutdown(self):
        """The app closes: stop the loop and the bots (they resume from their state files next time)."""
        self.timer.stop()
        self.manager.shutdown()

    # -- display -----------------------------------------------------------------------------------------
    def _colour(self, col, value):
        t = self.themes.current
        if col == "P&L" and isinstance(value, (int, float)) and value:
            return qcolor(t.up if value > 0 else t.down)
        if col == "state":
            return qcolor({"halted": t.warn, "error": t.down, "stopped": t.muted, "other process": t.muted
                           }.get(value, t.text))
        if col == "mode" and value == "live":
            return qcolor(t.live)
        return None

    _colour.marked = {"P&L"}  # the model draws the triangle next to the coloured figure (#191)

    def _row_tip(self, row) -> str | None:
        r = self._rows.get(row["bot"])
        if r is None:
            return None
        lines = [f"<b>{html.escape(r.name)}</b> · {html.escape(r.strategy)}",
                 f"realised {r.realized:+,.2f} · open {r.open_pnl:+,.2f}"]
        if r.next_check:
            lines.append(f"next check {r.next_check}")
        if r.error:
            lines.append(f"last step failed: {html.escape(r.error)}")
        if r.state == "elsewhere":
            lines.append("run by another process (tradebot bots run); Stop asks it to stop")
        return "<br>".join(lines)

    def refresh(self, force: bool = False):
        rows = self.manager.snapshot()
        self._rows = {r.name: r for r in rows}
        frame = pd.DataFrame([[r.name, r.symbol, r.strategy, r.timeframe, MODE_LABELS[r.mode],
                               STATE_LABELS.get(r.state, r.state), r.position, r.entry, r.last, r.pnl,
                               r.error or r.last_event, r.last_time] for r in rows], columns=COLUMNS)
        if force or self._last_frame is None or not frame.equals(self._last_frame):
            keep = self.selected()
            self.model.set_frame(frame)
            self._last_frame = frame
            self._fit_columns()
            if keep:
                self._restoring = True  # the same bot again: the editor keeps what is being typed
                self._select(keep)
                self._restoring = False
        self._summary(rows)
        self._show_events()
        self._buttons()
        modes = sorted({r.mode for r in rows if self.manager.is_running(r.name)})
        if modes != self._last_modes:
            self._last_modes = modes
            self.running_changed.emit(modes)

    def _fit_columns(self):
        head = self.table.horizontalHeader()
        head.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        head.setSectionResizeMode(COLUMNS.index("last event"), QHeaderView.ResizeMode.Stretch)
        head.setMinimumSectionSize(48)

    def _summary(self, rows):
        s = self.manager.summary(rows)
        t = self.themes.current
        self._set(self.kpis["running"], f"{s['running']} of {s['bots']}")
        self._set(self.kpis["positions"], str(s["positions"]))
        self._set(self.kpis["exposure"], _money(s["exposure"]))
        v = s["pnl"]
        if abs(v) < 0.005:
            self._set(self.kpis["pnl"], "0.00")
        else:
            colour = t.up if v > 0 else t.down
            self._set(self.kpis["pnl"], f'<span style="color:{colour}">{up_down_mark(v)} {_money(abs(v))}</span>')
        limits = s["limits"]
        if s["halted"]:
            limits = f'<span style="color:{t.warn}">halted: {html.escape(s["halted"])}</span>'
        self._set(self.kpis["limits"], limits)
        self.kpis["limits"].value.setToolTip(s["halted"] or "Limits all bots share; set them under Shared limits")

    def _show_events(self):
        name = self.selected()
        rows = [self._rows[name]] if name in self._rows else list(self._rows.values())
        title = f"Events · {name}" if name in self._rows else "Events · all bots"
        if self.events_title.text() != title:
            self.events_title.setText(title)
        events = [(e.get("time", ""), r.name, e.get("action", ""), e.get("amount", 0.0), e.get("price", 0.0),
                   e.get("note", "")) for r in rows for e in r.events]
        events.sort(key=lambda e: e[0], reverse=True)
        frame = pd.DataFrame([(t[:16].replace("T", " "), *rest) for t, *rest in events[:200]], columns=EVENT_COLUMNS)
        if not frame.equals(self.events_model.df if len(self.events_model.df.columns) else pd.DataFrame()):
            self.events_model.set_frame(frame)
            head = self.events.horizontalHeader()
            head.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
            head.setSectionResizeMode(EVENT_COLUMNS.index("note"), QHeaderView.ResizeMode.Stretch)
            self.events.setColumnHidden(1, name in self._rows)

    def _buttons(self):
        name = self.selected()
        r = self._rows.get(name)
        self.start_btn.setEnabled(bool(r) and r.state not in ("running", "halted", "error", "elsewhere")
                                  and r.mode != "live")
        self.stop_btn.setEnabled(bool(r) and r.state != "stopped")
        self.start_btn.setToolTip("Live bots start from a terminal: tradebot bots run --live" if r and r.mode == "live"
                                  else "Start the selected bot")
        self.stop_all_btn.setEnabled(bool(self.manager.running()))
        self.start_all_btn.setEnabled(any(b.enabled and b.mode != "live" for b in self.bots.bots))

    def _selected(self, *_):
        name = self.selected()
        if name and not getattr(self, "_restoring", False):
            try:
                self._edit(self.bots.get(name))
            except KeyError:
                pass
        self._show_events()
        self._buttons()
