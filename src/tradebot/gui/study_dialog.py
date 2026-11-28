# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""The indicator and signal dialog (#230): one place to add, edit and remove what the chart draws on the candles.

It looks and works like the symbol dialog (#220): a search field, a rail of categories (the indicator categories,
then the signals: single strategies and combined ones), and a table. Every view lists what is on the chart first
(marked ✓, with buttons for line colours and removing), then what can be added. Picking a row asks its parameters
and adds it; the dialog stays open for more until Esc. Picking a row that is on the chart edits its parameters.

It opens from the chart toolbar's Indicators button and with ``/`` over the chart. What it shows and changes goes
through the chart (:class:`~tradebot.gui.chart.ChartView`), its ``host``.
"""

from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtCore import QAbstractTableModel, QEvent, QModelIndex, QSize, Qt, QTimer
from PyQt6.QtGui import QColor, QFont, QKeyEvent
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QStackedWidget,
    QTableView,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from tradebot.algorithms import ALGORITHMS, algorithm_group, algorithm_names
from tradebot.gui.icons import icon, pixmap
from tradebot.gui.symbol_dialog import _ColorDelegate, _TypeDelegate, fit_dialog, rail_header, rail_item
from tradebot.indicators import INDICATORS
from tradebot.indicators.base import CATEGORIES, category_of

#: the signal groups in the rail, after the indicator categories
SIGNAL_GROUPS = {"strategy": "Strategies", "combined": "Combined"}
PLACEHOLDER = "Indicator or signal, e.g. rsi, bollinger, volume, macd cross"
HINTS = "↑↓ move    ↵ add · edit    Del remove    Tab category    Esc close"

ON, NAME, DESCRIPTION, KIND, COLOURS, REMOVE = range(6)
HEADERS = ("", "Name", "Description", "Type", "", "")
WIDTHS = {ON: 30, NAME: 190, KIND: 84, COLOURS: 34, REMOVE: 34}
DESCRIPTION_MIN = 200


@dataclass
class StudyRow:
    """One row: an indicator or a signal that can be added, or one on the chart (``live`` is its instance)."""

    kind: str  #: "indicator" or "signal"
    name: str
    description: str
    group: str  #: an indicator category key, or a key of SIGNAL_GROUPS
    cls: type | None = None  #: the indicator class (indicators only)
    live: object = None  #: the indicator on the chart, or the chart's signal
    label: str = ""  #: what the Name column shows; the name when empty

    @property
    def on_chart(self) -> bool:
        return self.live is not None

    @property
    def type_label(self) -> str:
        if self.kind == "signal":
            return "Signal"
        return "Overlay" if self.cls.overlay else "Pane"

    @property
    def key(self) -> tuple:
        return (self.kind, self.name, id(self.live) if self.live is not None else None)


def rank(query: str, row: StudyRow) -> int | None:
    """How well ``row`` matches: 0 the name starts with the query, 1 the name has it, 2 the description has every
    word of it; None: no match. Empty query: everything, as 0. (Categories are the rail's job: "vol" matching every
    "Volatility" indicator would bury the volume ones.)"""
    q = query.strip().lower()
    if not q:
        return 0
    name = row.name.lower()
    if name.startswith(q) or name.replace("_", " ").startswith(q):
        return 0
    if q in name or q.replace(" ", "_") in name:
        return 1
    text = " ".join([name.replace("_", " "), row.description.lower()])
    return 2 if all(word in text for word in q.split()) else None


class StudyTableModel(QAbstractTableModel):
    def __init__(self, dialog: StudyDialog):
        super().__init__(dialog)
        self.dialog = dialog
        self.rows: list[StudyRow] = []
        self.hover = -1

    def set_rows(self, rows: list[StudyRow]):
        self.beginResetModel()
        self.rows = list(rows)
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.rows)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(HEADERS)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation != Qt.Orientation.Horizontal:
            return None
        if role == Qt.ItemDataRole.DisplayRole:
            return HEADERS[section]
        if role == Qt.ItemDataRole.TextAlignmentRole:
            return Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        if role == Qt.ItemDataRole.ToolTipRole and section == ON:
            return "On the chart"
        return None

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        row, col, d = self.rows[index.row()], index.column(), self.dialog
        theme = d.theme()
        shown = index.row() in (self.hover, d.current_row())  # the row actions show on the hovered or current row
        if role == Qt.ItemDataRole.DisplayRole:
            if col == NAME:
                return row.label or row.name
            if col == DESCRIPTION:
                return row.description.replace("`", "")  # signal descriptions quote their parameters in backticks
            if col == KIND:
                return row.type_label
        elif role == Qt.ItemDataRole.TextAlignmentRole:
            return Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        elif role == Qt.ItemDataRole.ForegroundRole and theme is not None and col == DESCRIPTION:
            return QColor(theme.muted)
        elif role == Qt.ItemDataRole.FontRole and col == NAME:
            font = QFont(d.font())
            font.setWeight(QFont.Weight.DemiBold)
            return font
        elif role == Qt.ItemDataRole.DecorationRole and theme is not None:
            if col == ON and row.on_chart:
                return icon("check", theme.accent, 16)
            if col == COLOURS and shown and d.can_colour(row):
                return icon("palette", theme.text, 16)
            if col == REMOVE and shown and row.on_chart:
                return icon("close", theme.text, 16)
        elif role == Qt.ItemDataRole.ToolTipRole:
            if col == ON and row.on_chart:
                return "On the chart"
            if col == COLOURS and d.can_colour(row):
                return "Line colours …"
            if col == REMOVE and row.on_chart:
                return "Remove from the chart (Del)"
            if row.on_chart:
                return f"{row.description}\nClick or ↵ to change its parameters"
            return row.description
        return None

    def refresh_row(self, row: int):
        if 0 <= row < len(self.rows):
            self.dataChanged.emit(self.index(row, 0), self.index(row, len(HEADERS) - 1))


class StudyDialog(QDialog):
    """Add, edit and remove the chart's indicators and its signal. ``host`` is the chart."""

    def __init__(self, host, parent: QWidget | None = None, initial: str = ""):
        super().__init__(parent, Qt.WindowType.Dialog | Qt.WindowType.FramelessWindowHint)
        self.setObjectName("StudyDialog")
        self.setWindowTitle("Indicators and signals")
        self.host = host
        self.source = "all"  # the rail's selection: "all", "chart", "cat:<category>" or "sig:<group>"
        self.added: list[str] = []  # what was added or changed since the dialog opened, for the footer

        # search bar: the same look as the symbol dialog's (the theme styles both by these object names)
        bar = QFrame(objectName="SymbolSearchBar")
        self.search_icon = QLabel()
        self.query = QLineEdit(initial, objectName="SymbolQuery", placeholderText=PLACEHOLDER)
        self.query.textEdited.connect(lambda _: self._refresh())
        self.close_btn = QToolButton(toolTip="Close (Esc)", objectName="SymbolIconButton")
        self.close_btn.clicked.connect(self.reject)
        bl = QHBoxLayout(bar)
        bl.setContentsMargins(14, 0, 10, 0)
        bl.setSpacing(8)
        bl.addWidget(self.search_icon)
        bl.addWidget(self.query, 1)
        bl.addWidget(self.close_btn)
        bar.setFixedHeight(48)

        self.rail = QListWidget(objectName="SymbolRail")
        self.rail.setFixedWidth(180)
        self.rail.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._counts: dict[str, QLabel] = {}
        self._counts["all"] = rail_item(self.rail, "all", "All")
        self._counts["chart"] = rail_item(self.rail, "chart", "On chart")
        rail_header(self.rail, "Indicators")
        for key, label in CATEGORIES.items():
            if key != "script" or any(category_of(c) == "script" for c in INDICATORS):
                self._counts[f"cat:{key}"] = rail_item(self.rail, f"cat:{key}", label)
        rail_header(self.rail, "Signals")
        for key, label in SIGNAL_GROUPS.items():
            self._counts[f"sig:{key}"] = rail_item(self.rail, f"sig:{key}", label)
        self.rail.setCurrentRow(0)
        self.rail.currentItemChanged.connect(self._rail_changed)

        self.model = StudyTableModel(self)
        self.table = QTableView(objectName="SymbolTable")
        self.table.setModel(self.model)
        self.table.setItemDelegateForColumn(KIND, _TypeDelegate(self, accent=("Signal",)))
        self.table.setItemDelegateForColumn(DESCRIPTION, _ColorDelegate(self))
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # the query keeps the keyboard; arrows move the rows
        self.table.setShowGrid(False)
        self.table.setWordWrap(False)
        self.table.setMouseTracking(True)
        self.table.setIconSize(QSize(16, 16))
        self.table.verticalHeader().hide()
        self.table.verticalHeader().setDefaultSectionSize(36)
        header = self.table.horizontalHeader()
        header.setHighlightSections(False)
        header.setSectionResizeMode(DESCRIPTION, QHeaderView.ResizeMode.Stretch)
        for col, width in WIDTHS.items():
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.Fixed)
            self.table.setColumnWidth(col, width)
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.table.clicked.connect(self._clicked)
        self.table.entered.connect(self._hovered)
        self.table.viewport().installEventFilter(self)
        self.empty = QLabel(objectName="SymbolEmpty", wordWrap=True, alignment=Qt.AlignmentFlag.AlignCenter)
        self.results = QStackedWidget()
        self.results.addWidget(self.table)
        self.results.addWidget(self.empty)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        body.addWidget(self.rail)
        body.addWidget(self.results, 1)

        foot = QFrame(objectName="SymbolFooter")
        self.hints = QLabel(HINTS, objectName="Muted")
        self.status = QLabel(objectName="Muted", alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        fl = QHBoxLayout(foot)
        fl.setContentsMargins(14, 7, 14, 7)
        fl.addWidget(self.hints)
        fl.addStretch(1)
        fl.addWidget(self.status)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(1, 1, 1, 1)
        lay.setSpacing(0)
        lay.addWidget(bar)
        lay.addLayout(body, 1)
        lay.addWidget(foot)

        themes = getattr(host, "themes", None)
        if themes is not None:
            themes.changed.connect(self._themed)
        self.query.installEventFilter(self)
        self._themed()
        fit_dialog(self, parent, 940, 600)
        self._refresh()

    def reset(self, parent: QWidget, initial: str = ""):
        """Ready the reused dialog for another opening."""
        self.added = []
        self.status.setText("")
        self.query.setText(initial)
        self.rail.blockSignals(True)
        self.rail.setCurrentRow(0)
        self.rail.blockSignals(False)
        self.source = "all"
        fit_dialog(self, parent, 940, 600)
        self._refresh()

    # -- look --------------------------------------------------------------------------------------------------
    def theme(self):
        themes = getattr(self.host, "themes", None)
        return themes.current if themes is not None else None

    def _themed(self, *_):
        t = self.theme()
        muted = t.muted if t else "#888888"
        self.search_icon.setPixmap(pixmap("search", muted, 18))
        self.close_btn.setIcon(icon("close", muted, 16))
        self.model.layoutChanged.emit()

    def can_colour(self, row: StudyRow) -> bool:
        return row.kind == "indicator" and row.on_chart and bool(self.host.styled_outputs(row.live))

    # -- what the table shows ----------------------------------------------------------------------------------
    def catalogue(self) -> list[StudyRow]:
        """Everything that can be added: the indicators, then the signals (read now: saved strategies appear)."""
        rows = [StudyRow("indicator", c.name, c.description, category_of(c), cls=c) for c in INDICATORS]
        rows.sort(key=lambda r: r.name)
        for name in algorithm_names():
            rows.append(StudyRow("signal", name, getattr(ALGORITHMS.get(name), "description", "") or "",
                                 algorithm_group(name)))
        return rows

    def on_chart(self) -> list[StudyRow]:
        """The chart's indicators in their order, then its signal."""
        label = self.host.study_label
        rows = [StudyRow("indicator", ind.name, ind.description, category_of(type(ind)), cls=type(ind), live=ind,
                         label=label(ind)) for ind in self.host.indicators]
        algo = self.host.algorithm
        if algo is not None:
            rows.append(StudyRow("signal", algo.name, algo.description or "", algorithm_group(algo.name), live=algo,
                                 label=label(algo)))
        return rows

    def _in_source(self, row: StudyRow, source: str) -> bool:
        kind, _, key = source.partition(":")
        if kind == "all":
            return True
        if kind == "chart":
            return row.on_chart
        return row.group == key and row.kind == ("indicator" if kind == "cat" else "signal")

    def _refresh(self, keep: tuple | None = None):
        q = self.query.text().strip()
        live = [r for r in self.on_chart() if rank(q, r) is not None]
        found = sorted((r for r in self.catalogue() if rank(q, r) is not None), key=lambda r: rank(q, r))
        for key, label in self._counts.items():  # counts while searching, and always for "On chart"
            n = sum(1 for r in live + found if self._in_source(r, key)) if q else len(live) if key == "chart" else None
            label.setText(str(n) if n else "")
        rows = [r for r in live + found if self._in_source(r, self.source)]
        self.model.set_rows(rows)
        if rows:
            self.results.setCurrentWidget(self.table)
            # what Enter should do: while searching, add the best match (not edit one that's on the chart)
            first = next((i for i, r in enumerate(rows) if not r.on_chart), 0) if q else 0
            again = next((i for i, r in enumerate(rows) if r.key == keep), first) if keep else first
            self.select_row(again)
        else:
            self.results.setCurrentWidget(self.empty)
            self.empty.setText(self._empty_text(q))

    def _empty_text(self, q: str) -> str:
        if q:
            return f"No indicator or signal matches “{q}”.\nTry a shorter name or a word from its description."
        if self.source == "chart":
            return "Nothing on the chart yet.\nPick an indicator or a signal under All to add it."
        return "Nothing here yet."

    # -- selection and actions ---------------------------------------------------------------------------------
    def current_row(self) -> int:
        return self.table.currentIndex().row()

    def current(self) -> StudyRow | None:
        row = self.current_row()
        return self.model.rows[row] if 0 <= row < len(self.model.rows) else None

    def select_row(self, row: int):
        if not self.model.rows:
            return
        old = self.current_row()
        row = max(0, min(row, len(self.model.rows) - 1))
        self.table.setCurrentIndex(self.model.index(row, NAME))
        self.table.scrollTo(self.model.index(row, NAME))
        self.model.refresh_row(old)
        self.model.refresh_row(row)

    def _clicked(self, index: QModelIndex):
        row = self.model.rows[index.row()]
        if index.column() == REMOVE and row.on_chart:
            self.remove(row)
        elif index.column() == COLOURS and self.can_colour(row):
            self.host.edit_styles(row.live, parent=self)
        else:
            self.choose(row)

    def _hovered(self, index: QModelIndex):
        old, self.model.hover = self.model.hover, index.row()
        for row in {old, index.row()} - {-1}:
            self.model.refresh_row(row)

    def choose(self, row: StudyRow):
        """Add the row (asking its parameters), or edit it if it is on the chart. The dialog stays open."""
        keep = row.key
        if row.kind == "indicator" and row.on_chart:
            done = self.host.edit_indicator(row.live, parent=self)
        elif row.kind == "indicator":
            done = self.host.add_indicator(row.cls, parent=self)
        else:
            done = self.host.choose_signal(row.name, parent=self)
        if done is None or done is False:
            return  # cancelled
        if row.on_chart:  # the edited one is a new instance now: keep the highlight on it
            keep = (row.kind, row.name, id(done if row.kind == "indicator" else self.host.algorithm))
        self.added.append(self.host.study_label(done) if row.kind == "indicator" else f"signal {row.name}")
        self.status.setText("Added " + ", ".join(self.added[-3:]) + (" …" if len(self.added) > 3 else ""))
        self._refresh(keep)

    def remove(self, row: StudyRow):
        if row.kind == "indicator":
            self.host.remove_indicator(row.live)
        else:
            self.host.remove_signal()
        self.status.setText(f"Removed {row.label or row.name}")
        at = self.current_row()
        self._refresh()
        self.select_row(at)

    def set_source(self, key: str):
        for i in range(self.rail.count()):
            if self.rail.item(i).data(Qt.ItemDataRole.UserRole) == key:
                self.rail.setCurrentRow(i)
                return

    def _rail_changed(self, item, _old=None):
        key = item.data(Qt.ItemDataRole.UserRole) if item is not None else None
        if key:
            self.source = key
            self._refresh()
            self.query.setFocus()

    def _next_source(self, step: int):
        keys = [self.rail.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.rail.count())]
        keys = [k for k in keys if k]
        i = keys.index(self.source) if self.source in keys else -1
        self.set_source(keys[(i + step) % len(keys)])

    # -- keyboard ----------------------------------------------------------------------------------------------
    def eventFilter(self, obj, event):
        if obj is self.query and event.type() == QEvent.Type.KeyPress:
            return self._key(event)
        if obj is self.table.viewport() and event.type() == QEvent.Type.Resize:
            self.table.setColumnHidden(KIND, self.table.viewport().width() - DESCRIPTION_MIN < sum(WIDTHS.values()))
        if obj is self.table.viewport() and event.type() == QEvent.Type.Leave and self.model.hover >= 0:
            self._hovered(self.model.index(-1, 0))
        return super().eventFilter(obj, event)

    def _key(self, event: QKeyEvent) -> bool:
        key = event.key()
        page = max(1, self.table.viewport().height() // 36 - 1)
        moves = {Qt.Key.Key_Down: 1, Qt.Key.Key_Up: -1, Qt.Key.Key_PageDown: page, Qt.Key.Key_PageUp: -page}
        if key in moves:
            self.select_row(self.current_row() + moves[key])
            return True
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            row = self.current()
            if row is not None:
                self.choose(row)
            return True
        if key == Qt.Key.Key_Delete:
            row = self.current()
            if row is not None and row.on_chart:
                self.remove(row)
            return True
        if key == Qt.Key.Key_Tab:
            self._next_source(1)
            return True
        if key == Qt.Key.Key_Backtab:
            self._next_source(-1)
            return True
        return False

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.reject()
            return
        super().keyPressEvent(event)

    def showEvent(self, event):
        super().showEvent(event)
        self.query.setFocus()
        QTimer.singleShot(0, lambda: self.query.end(False))
