# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Shared GUI building blocks: background workers, pandas table model, cards."""

from __future__ import annotations

import traceback
from typing import Callable

import numpy as np
import pandas as pd
from PyQt6.QtCore import (
    QAbstractTableModel,
    QDate,
    QModelIndex,
    QObject,
    QPoint,
    QRect,
    QRunnable,
    QSize,
    Qt,
    QThreadPool,
    QTimer,
    pyqtSignal,
)
from PyQt6.QtGui import QColor, QIcon, QStandardItem, QStandardItemModel
from PyQt6.QtWidgets import (
    QCalendarWidget,
    QCompleter,
    QDateEdit,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLayout,
    QLineEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from tradebot.symbols import SymbolSearch


class _Signals(QObject):
    done = pyqtSignal(object)
    failed = pyqtSignal(str)


class _Task(QRunnable):
    def __init__(self, fn, signals):
        super().__init__()
        self.fn, self.signals = fn, signals

    def run(self):
        try:
            result = self.fn()
        except Exception as exc:  # surfaced in the UI, never crashes the app
            traceback.print_exc()
            self.signals.failed.emit(f"{type(exc).__name__}: {exc}")
        else:
            self.signals.done.emit(result)


_alive: set[_Signals] = set()


def run_async(fn: Callable[[], object], on_done: Callable[[object], None], on_error: Callable[[str], None] | None = None):
    """Run ``fn`` on the global thread pool; callbacks are delivered on the GUI thread."""
    signals = _Signals()
    _alive.add(signals)
    signals.done.connect(on_done)
    signals.done.connect(lambda _: _alive.discard(signals))
    signals.failed.connect(lambda _: _alive.discard(signals))
    if on_error:
        signals.failed.connect(on_error)
    QThreadPool.globalInstance().start(_Task(fn, signals))


class DataFrameModel(QAbstractTableModel):
    """Read-only, sortable view of a DataFrame with per-column formatters."""

    def __init__(self, df: pd.DataFrame | None = None, formats: dict[str, Callable] | None = None,
                 colors: Callable[[str, object], QColor | None] | None = None):
        super().__init__()
        self.df = df if df is not None else pd.DataFrame()
        self.formats, self.colors = formats or {}, colors

    def set_frame(self, df: pd.DataFrame):
        self.beginResetModel()
        self.df = df.reset_index(drop=True)
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return len(self.df)

    def columnCount(self, parent=QModelIndex()):
        return len(self.df.columns)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        col = self.df.columns[index.column()]
        value = self.df.iat[index.row(), index.column()]
        if role == Qt.ItemDataRole.DisplayRole:
            if value is None or (isinstance(value, float) and np.isnan(value)):
                return "–"
            fmt = self.formats.get(col)
            return fmt(value) if fmt else (f"{value:,.4g}" if isinstance(value, (float, np.floating)) else str(value))
        if role == Qt.ItemDataRole.ForegroundRole and self.colors:
            return self.colors(col, value)
        if role == Qt.ItemDataRole.TextAlignmentRole and not isinstance(value, str):
            return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        if role == Qt.ItemDataRole.UserRole:
            return value
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return str(self.df.columns[section]).replace("_", " ")
        return None

    def sort(self, column, order=Qt.SortOrder.AscendingOrder):
        if self.df.empty:
            return
        self.beginResetModel()
        col = self.df.columns[column]
        self.df = self.df.sort_values(col, ascending=order == Qt.SortOrder.AscendingOrder, na_position="last",
                                      ignore_index=True)
        self.endResetModel()


def card(*widgets, layout_cls=QVBoxLayout) -> QFrame:
    frame = QFrame()
    frame.setObjectName("Card")
    lay = layout_cls(frame)
    lay.setContentsMargins(14, 12, 14, 12)
    for w in widgets:
        lay.addWidget(w)
    return frame


class Kpi(QFrame):
    """Label + big value tile."""

    def __init__(self, label: str):
        super().__init__()
        self.setObjectName("Card")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 12, 16, 12)
        self.label = QLabel(label)
        self.label.setObjectName("KpiLabel")
        self.value = QLabel("–")
        self.value.setObjectName("KpiValue")
        lay.addWidget(self.label)
        lay.addWidget(self.value)

    def set(self, text: str):
        self.value.setText(text)


class FlowLayout(QLayout):
    """Lays items out left to right and wraps to a new row when the width runs out (Qt's flow layout).

    With ``last_right``, the last item sits flush with the right edge of its row (e.g. a toolbar's tools).
    """

    def __init__(self, parent=None, spacing: int = 6, last_right: bool = False):
        super().__init__(parent)
        self.last_right = last_right
        self._items: list = []
        self.setSpacing(spacing)
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item):  # noqa: N802 (Qt API)
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, index):  # noqa: N802
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index):  # noqa: N802
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self):  # noqa: N802
        return Qt.Orientation(0)

    def hasHeightForWidth(self):  # noqa: N802
        return True

    def heightForWidth(self, width):  # noqa: N802
        return self._arrange(QRect(0, 0, width, 0), move=False)

    def setGeometry(self, rect):  # noqa: N802
        super().setGeometry(rect)
        self._arrange(rect, move=True)

    def sizeHint(self):  # noqa: N802
        return self.minimumSize()

    def minimumSize(self):  # noqa: N802
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        return size

    def _arrange(self, rect, move: bool) -> int:
        x, y, row_height, gap = rect.x(), rect.y(), 0, self.spacing()
        for n, item in enumerate(self._items):
            hint = item.sizeHint()
            if x > rect.x() and x + hint.width() > rect.right() + 1:  # doesn't fit: next row
                x, y, row_height = rect.x(), y + row_height + gap, 0
            if move:
                left = x
                if self.last_right and n == len(self._items) - 1:
                    left = max(x, rect.right() + 1 - hint.width())
                item.setGeometry(QRect(QPoint(left, y), hint))
            x += hint.width() + gap
            row_height = max(row_height, hint.height())
        return y + row_height - rect.y()


def toolbar_group(*widgets) -> QWidget:
    """Widgets in one row that a :class:`FlowLayout` moves as a unit."""
    group = QWidget()
    lay = QHBoxLayout(group)
    lay.setContentsMargins(0, 0, 0, 0)
    for w in widgets:
        lay.addWidget(w)
    return group


class _Calendar(QCalendarWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        # Qt's month arrows are fixed dark icons, invisible on dark themes; text follows the theme colour
        for name, arrow in (("qt_calendar_prevmonth", "‹"), ("qt_calendar_nextmonth", "›")):
            button = self.findChild(QToolButton, name)
            if button is not None:
                button.setIcon(QIcon())
                button.setText(arrow)

    def showEvent(self, event):
        super().showEvent(event)
        if self.selectedDate() == self.minimumDate():  # unset field: open on this month, not 1970
            today = QDate.currentDate()
            self.setCurrentPage(today.year(), today.month())


class DateRange(QWidget):
    """Optional From/To dates as an alternative to a candle count.

    An unset field shows "auto": no From means "use the candle count", no To means "up to now".
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.start = self._edit("from: auto", "First day (overrides the candle count)")
        self.end = self._edit("to: now", "Last day, inclusive")
        clear = QToolButton(text="✕", toolTip="Clear the dates and use the candle count")
        clear.clicked.connect(self.clear)
        for w in (self.start, self.end, clear):
            lay.addWidget(w)

    @staticmethod
    def _edit(unset_text: str, tip: str) -> QDateEdit:
        edit = QDateEdit(calendarPopup=True, displayFormat="yyyy-MM-dd", toolTip=tip)
        edit.setMinimumDate(QDate(1970, 1, 1))
        edit.setSpecialValueText(unset_text)  # shown while the date sits at the minimum, i.e. unset
        edit.setCalendarWidget(_Calendar(edit))  # the edit doesn't take ownership: parent it so it outlives Python's ref
        edit.setDate(edit.minimumDate())
        return edit

    def clear(self):
        for edit in (self.start, self.end):
            edit.setDate(edit.minimumDate())

    def range(self) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
        """``(start, end)`` for ``MarketDataEngine.get_ohlc``; ``end`` is exclusive, so the To day is included."""
        def day(edit):
            d = edit.date()
            return None if d == edit.minimumDate() else pd.Timestamp(d.year(), d.month(), d.day(), tz="UTC")

        start, end = day(self.start), day(self.end)
        return start, end + pd.Timedelta(days=1) if end is not None else None


class SymbolCompleter(QObject):
    """Autocompletion for a symbol field: local matches while typing, network matches after a pause.

    The popup lists ``symbol — name · exchange · type``; choosing a row inserts the plain
    symbol and emits :attr:`chosen`.
    """

    chosen = pyqtSignal(str)
    SYMBOL_ROLE = Qt.ItemDataRole.UserRole + 1
    DEBOUNCE_MS = 250
    MIN_REMOTE_CHARS = 2

    def __init__(self, edit: QLineEdit, search: SymbolSearch):
        super().__init__(edit)
        self.edit, self.search = edit, search
        self.model = QStandardItemModel(self)
        self.completer = QCompleter(self.model, self)
        # the search already filters and ranks, so show the rows as they are
        self.completer.setCompletionMode(QCompleter.CompletionMode.UnfilteredPopupCompletion)
        self.completer.setCompletionRole(self.SYMBOL_ROLE)  # what gets inserted: the plain symbol
        self.completer.popup().setObjectName("SymbolPopup")
        self.completer.popup().setMinimumWidth(440)
        self.completer.activated[str].connect(self._chosen)
        edit.setCompleter(self.completer)
        self._timer = QTimer(self, singleShot=True, interval=self.DEBOUNCE_MS, timeout=self._search_remote)
        self._pending: str | None = None  # query of the network search whose results are still wanted
        edit.textEdited.connect(self._edited)
        edit.editingFinished.connect(self._cancel)  # Enter or leaving the field: no late popup

    def _edited(self, text: str):
        self._show(self.search.local(text))  # instant and offline
        if len(text.strip()) >= self.MIN_REMOTE_CHARS:
            self._timer.start()  # restarts while typing: one network search per pause
        else:
            self._timer.stop()

    def _search_remote(self):
        query = self._pending = self.edit.text()
        run_async(lambda: self.search.search(query), lambda found: self._remote_done(query, found))

    def _remote_done(self, query: str, found):
        if query == self._pending == self.edit.text():  # drop results for text already changed or done
            self._show(found)

    def _cancel(self):
        self._timer.stop()
        self._pending = None

    def _chosen(self, symbol: str):
        self._cancel()
        self.edit.setText(symbol)  # this slot runs before QLineEdit's own one, so the field still holds the query
        self.chosen.emit(symbol)

    def _show(self, matches):
        self.model.clear()
        for m in matches:
            item = QStandardItem(m.label)
            item.setData(m.symbol, self.SYMBOL_ROLE)
            item.setToolTip(m.label)
            self.model.appendRow(item)
        if matches:
            self.completer.complete()
        else:
            self.completer.popup().hide()


def shortfall(df: pd.DataFrame, bars: int, start) -> str:
    """Status note when a candle-count request found less history than asked for."""
    return f" (only {len(df)} of {bars} candles available)" if start is None and len(df) < bars else ""


def epoch_seconds(index: pd.DatetimeIndex) -> np.ndarray:
    """Unix seconds for chart x-axes, independent of the index's datetime unit."""
    return index.as_unit("ns").asi8 / 1e9


def pct(v) -> str:
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v * 100:.2f}%"
