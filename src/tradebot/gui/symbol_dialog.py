# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""The symbol dialog (#220): search every source, filter by category, see price and day change, divide two symbols.

Categories sit in a rail on the left (All, Stocks, Forex, Crypto, Indices, Futures, Bonds; ETFs and funds count as
stocks), followed by the user's lists (recent symbols and the watchlists). Results fill a sortable table. ``/`` or
the ÷ button on a row starts a ratio: the next pick is the divisor, and the dialog returns ``A:B``, the spread
symbol the data engine builds from both legs (#117).

It opens from a symbol field (:class:`~tradebot.gui.common.SymbolCompleter`) and by typing over the chart. What it
needs from the main window (prices, recent symbols, watchlists, theme) comes from a :class:`SymbolContext`, found
as the window's ``symbol_context``; without one it still searches, just without those extras.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from PyQt6.QtCore import QAbstractTableModel, QEvent, QModelIndex, QObject, QRectF, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QKeyEvent, QPainter, QPen
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QStackedWidget,
    QStyledItemDelegate,
    QTableView,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from tradebot.core.models import split_pair
from tradebot.gui.common import last_and_change, run_async
from tradebot.gui.icons import icon, pixmap
from tradebot.gui.watchlists import fmt_price
from tradebot.symbols import CATEGORIES, SymbolMatch, SymbolSearch, category, rank, symbol_name

RECENT_KEY = "symbols.recent"
RECENT_MAX = 15
QUOTE_MAX_AGE = 60.0  #: seconds a price stays fresh
QUOTE_BARS = 5  #: daily candles per price: the last close and the one before, with room for holidays
BROWSE_MAX = 300  #: rows listed for a category before anything is typed
SEARCH_LIMIT = 200
DEBOUNCE_MS = 250

#: the Type column's text for a source kind
KIND_LABELS = {"stock": "Stock", "ETF": "ETF", "fund": "Fund", "index": "Index", "crypto": "Crypto", "forex": "FX",
               "future": "Future"}
#: and for a symbol without a known kind (from a watchlist), by category; plain tickers could be stock or ETF
CATEGORY_LABELS = {"crypto": "Crypto", "forex": "FX", "futures": "Future", "indices": "Index", "bonds": "Yield"}

PLACEHOLDER = "Symbol or name, e.g. AAPL, bitcoin, EURUSD, BTCUSD/ETHUSD"
HINTS = "↑↓ move    ↵ open    / divide by…    Tab category    Esc close"
HINTS_RATIO = "↑↓ move    ↵ open    ⌫ back to one symbol    Tab category    Esc close"
HINTS_MULTI = "↑↓ move    ↵ add    / divide by…    Tab category    Esc done"


def can_divide(numerator: str, divisor: str) -> bool:
    """Whether ``numerator:divisor`` is a spread the engine can load: two crypto pairs or two other symbols."""
    return numerator != divisor and split_pair(f"{numerator}:{divisor}") is not None


def _compact(text: str) -> str:
    return text.strip().lower().replace("/", "").replace(" ", "")


class QuoteBook(QObject):
    """Last price and day change per symbol, loaded in the background and kept for :data:`QUOTE_MAX_AGE`."""

    updated = pyqtSignal(str)

    def __init__(self, engine, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.quotes: dict[str, tuple[float, float | None]] = {}
        self._fetched: dict[str, float] = {}
        self._loading: set[str] = set()

    def get(self, symbol: str) -> tuple[float, float | None] | None:
        return self.quotes.get(symbol)

    def loading(self, symbol: str) -> bool:
        """Asked for and not answered yet (and no older price to show meanwhile)."""
        return symbol in self._loading and symbol not in self.quotes

    def request(self, symbol: str):
        if self.engine is None or symbol in self._loading or time.time() - self._fetched.get(symbol, 0) < QUOTE_MAX_AGE:
            return
        from tradebot.data import MarketDataEngine

        engine = self.engine
        if isinstance(engine, MarketDataEngine):  # its memory cache would keep the day's first price all day
            engine = MarketDataEngine(engine.providers, cache=False, disk=engine.disk)
        self._loading.add(symbol)
        if symbol not in self.quotes:  # the disk cache first: a price right away, even offline
            run_async(lambda: engine.peek(symbol, "1d", bars=QUOTE_BARS), lambda df: self._got(symbol, df, False),
                      lambda err: None)
        run_async(lambda: engine.get_ohlc(symbol, "1d", bars=QUOTE_BARS), lambda df: self._got(symbol, df, True),
                  lambda err: self._failed(symbol))  # offline or unknown: no price, and no error

    def _failed(self, symbol: str):
        self._loading.discard(symbol)
        self._fetched[symbol] = time.time()  # not again right away
        self.updated.emit(symbol)  # "…" becomes "–"

    def _got(self, symbol: str, df, fresh: bool):
        if fresh:
            self._loading.discard(symbol)
            self._fetched[symbol] = time.time()
        elif symbol in self._fetched:
            return  # the current price came first
        quote = last_and_change(df)
        if quote is not None:
            self.quotes[symbol] = quote
            self.updated.emit(symbol)


@dataclass
class SymbolContext:
    """What the dialog takes from the main window; every part is optional."""

    engine: object = None
    settings: object = None
    themes: object = None
    lists: Callable[[], dict[str, list[str]]] = dict  #: the watchlists, name -> symbols
    quotes: QuoteBook | None = field(default=None, repr=False)

    def __post_init__(self):
        if self.quotes is None:
            self.quotes = QuoteBook(self.engine)

    def recent(self) -> list[str]:
        return list(self.settings.get(RECENT_KEY, []) or []) if self.settings is not None else []

    def remember(self, symbol: str):
        if self.settings is not None:
            self.settings.set(RECENT_KEY, [symbol, *(s for s in self.recent() if s != symbol)][:RECENT_MAX])

    def watched(self) -> set[str]:
        return {s for symbols in self.lists().values() for s in symbols}

    def dialog(self, search: SymbolSearch, window: QWidget) -> SymbolDialog:
        """The window's one symbol dialog, built on first use (or by :meth:`prewarm`)."""
        if getattr(self, "_dialog", None) is None:
            self._dialog = SymbolDialog(search, window, self)
        return self._dialog

    def prewarm(self, search: SymbolSearch, window: QWidget):
        """Build and style the dialog while the app is idle, so the first open is as quick as the next ones."""
        dialog = self.dialog(search, window)
        dialog.ensurePolished()
        dialog.grab()  # lays out and paints it once, off screen


def _scaled(font: QFont, factor: float) -> QFont:
    """``font`` at ``factor`` of its size; the stylesheet sets sizes in pixels, so the point size may be unset."""
    font = QFont(font)
    if font.pixelSize() > 0:
        font.setPixelSize(max(8, round(font.pixelSize() * factor)))
    else:
        font.setPointSizeF(font.pointSizeF() * factor)
    return font


def fit_dialog(dialog: QDialog, parent: QWidget | None, width: int = 1000, height: int = 640):
    """Size a picker dialog to ``width`` x ``height``, smaller in a small window and never past it, and place it in
    the upper middle of ``parent``'s window."""
    window = parent.window() if parent is not None else None
    w, h = (window.width(), window.height()) if window is not None else (1100, 760)
    dialog.resize(max(620, min(width, w - 48)), max(400, min(height, h - 96)))
    if window is not None:
        center = window.mapToGlobal(window.rect().center())
        dialog.move(center.x() - dialog.width() // 2, max(window.mapToGlobal(window.rect().topLeft()).y() + 48,
                                                          center.y() - dialog.height() // 2 - 40))


def rail_item(rail: QListWidget, key: str, label: str) -> QLabel:
    """A selectable row of a picker's rail: ``label`` with room for a count at the right, which is returned."""
    item = QListWidgetItem()
    item.setData(Qt.ItemDataRole.UserRole, key)
    row = QWidget()
    row.setStyleSheet("background: transparent;")
    name, count = QLabel(label), QLabel(objectName="Muted")
    name.setStyleSheet("background: transparent;")
    count.setStyleSheet("background: transparent;")
    rl = QHBoxLayout(row)
    rl.setContentsMargins(10, 6, 10, 6)
    rl.addWidget(name, 1)
    rl.addWidget(count)
    item.setSizeHint(QSize(160, 32))
    rail.addItem(item)
    rail.setItemWidget(item, row)
    return count


def rail_header(rail: QListWidget, label: str):
    """A heading in a picker's rail, not selectable."""
    item = QListWidgetItem(label)
    item.setFlags(Qt.ItemFlag.NoItemFlags)
    item.setSizeHint(QSize(160, 30))
    font = _scaled(rail.font(), 0.85)
    font.setWeight(QFont.Weight.Medium)
    item.setFont(font)
    rail.addItem(item)


def context_for(widget: QWidget) -> SymbolContext | None:
    return getattr(widget.window(), "symbol_context", None)


# -- the results table -------------------------------------------------------------------------------------------
STAR, SYMBOL, NAME, TYPE, EXCHANGE, LAST, DAY, DIVIDE = range(8)
HEADERS = ("", "Symbol", "Name", "Type", "Exchange", "Last", "Day", "")
NAME_MIN = 160
WIDTHS = {STAR: 30, SYMBOL: 128, TYPE: 70, EXCHANGE: 112, LAST: 96, DAY: 96, DIVIDE: 34}


def type_label(m: SymbolMatch) -> str:
    if category(m) == "bonds":
        return "Yield"
    return KIND_LABELS.get(m.kind) or CATEGORY_LABELS.get(category(m), "")


class SymbolTableModel(QAbstractTableModel):
    def __init__(self, dialog: SymbolDialog):
        super().__init__(dialog)
        self.dialog = dialog
        self.rows: list[SymbolMatch] = []
        self.hover = -1
        self._sort: tuple[int, Qt.SortOrder] | None = None

    def set_rows(self, rows: list[SymbolMatch]):
        self.beginResetModel()
        self.rows = list(rows)
        if self._sort:
            self._sorted(*self._sort)
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
        if role == Qt.ItemDataRole.TextAlignmentRole and section in (LAST, DAY):
            return Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        if role == Qt.ItemDataRole.TextAlignmentRole:
            return Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        if role == Qt.ItemDataRole.ToolTipRole and section == STAR:
            return "In one of your watchlists"
        return None

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        m, col, d = self.rows[index.row()], index.column(), self.dialog
        quote = d.ctx.quotes.get(m.symbol)
        theme = d.theme()
        if role == Qt.ItemDataRole.DisplayRole:
            if col == SYMBOL:
                return m.symbol
            if col == NAME:
                return m.name or d.names.get(m.symbol, "")
            if col == TYPE:
                return type_label(m)
            if col == EXCHANGE:
                return m.exchange
            if col in (LAST, DAY) and quote is None:
                return "…" if d.ctx.quotes.loading(m.symbol) else "–"
            if col == LAST:
                return fmt_price(quote[0])
            if col == DAY:
                if quote[1] is None:
                    return "–"
                change = quote[1]
                mark = "▲ " if change > 0 else "▼ " if change < 0 else ""  # colour is never the only cue
                return f"{mark}{change * 100:+.2f}%"
        elif role == Qt.ItemDataRole.TextAlignmentRole:
            if col in (LAST, DAY):
                return Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            return Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        elif role == Qt.ItemDataRole.ForegroundRole and theme is not None:
            if col == DAY and quote and quote[1]:
                return QColor(theme.up if quote[1] > 0 else theme.down)
            if col in (NAME, EXCHANGE) or (col in (LAST, DAY) and not quote):
                return QColor(theme.muted)
        elif role == Qt.ItemDataRole.FontRole and col == SYMBOL:
            font = QFont(d.font())
            font.setWeight(QFont.Weight.DemiBold)
            return font
        elif role == Qt.ItemDataRole.DecorationRole and theme is not None:
            if col == STAR and m.symbol in d.watched:
                return icon("star", theme.accent, 14)
            if col == DIVIDE and d.numerator is None and ":" not in m.symbol and index.row() in (self.hover, d.current_row()):
                return icon("divide", theme.text, 16)
        elif role == Qt.ItemDataRole.ToolTipRole:
            if col == DIVIDE and d.numerator is None and ":" not in m.symbol:
                return f"Divide {m.symbol} by another symbol (/)"
            if col == STAR and m.symbol in d.watched:
                return "In one of your watchlists"
            return m.label
        return None

    def sort(self, column, order=Qt.SortOrder.AscendingOrder):
        if column in (STAR, DIVIDE):
            return
        self.layoutAboutToBeChanged.emit()
        self._sort = (column, order)
        self._sorted(column, order)
        self.layoutChanged.emit()

    def _sorted(self, column, order):
        quotes, names = self.dialog.ctx.quotes, self.dialog.names

        def key(m: SymbolMatch):
            q = quotes.get(m.symbol)
            if column in (LAST, DAY):  # rows without a price go last either way
                v = q and (q[0] if column == LAST else q[1])
                return (v is None, v if v is not None else 0)
            text = {SYMBOL: m.symbol, NAME: m.name or names.get(m.symbol, ""), TYPE: type_label(m),
                    EXCHANGE: m.exchange}[column]
            return (not text, text.lower())

        self.rows.sort(key=key, reverse=order == Qt.SortOrder.DescendingOrder)

    def refresh_symbol(self, symbol: str):
        for i, m in enumerate(self.rows):
            if m.symbol == symbol:
                self.dataChanged.emit(self.index(i, 0), self.index(i, len(HEADERS) - 1))


class _ColorDelegate(QStyledItemDelegate):
    """Draws a cell's text in its own colour on the selected row too: the stylesheet's selection colour would turn
    a red or green day change into plain text."""

    def paint(self, painter: QPainter, option, index):
        color = index.data(Qt.ItemDataRole.ForegroundRole)
        if color is None:
            return super().paint(painter, option, index)
        self.initStyleOption(option, index)
        text, option.text = option.text, ""
        style = option.widget.style()
        style.drawControl(style.ControlElement.CE_ItemViewItem, option, painter, option.widget)
        rect = style.subElementRect(style.SubElement.SE_ItemViewItemText, option, option.widget)
        painter.save()
        painter.setFont(option.font)
        painter.setPen(color)
        painter.drawText(rect, int(option.displayAlignment), option.fontMetrics.elidedText(
            text, Qt.TextElideMode.ElideRight, rect.width()))
        painter.restore()


class _TypeDelegate(QStyledItemDelegate):
    """The Type column as a small outlined tag; ``accent`` ones (ETFs here) in the accent colour, so they stand out."""

    def __init__(self, dialog: QDialog, accent: tuple[str, ...] = ("ETF",)):
        super().__init__(dialog)
        self.dialog, self.accent = dialog, accent

    def paint(self, painter: QPainter, option, index):
        text = index.data()
        theme = self.dialog.theme()
        if not text or theme is None:
            return super().paint(painter, option, index)
        self.initStyleOption(option, index)
        option.text = ""
        option.widget.style().drawControl(option.widget.style().ControlElement.CE_ItemViewItem, option, painter,
                                          option.widget)
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setFont(_scaled(option.font, 0.85))
        width = painter.fontMetrics().horizontalAdvance(text) + 12
        height = painter.fontMetrics().height() + 2
        rect = QRectF(option.rect.left() + 6, option.rect.center().y() - height / 2 + 0.5, width, height)
        color = QColor(theme.accent if text in self.accent else theme.muted)
        painter.setPen(QPen(color, 1))
        painter.drawRoundedRect(rect, 4, 4)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)
        painter.restore()


# -- the dialog --------------------------------------------------------------------------------------------------
class SymbolDialog(QDialog):
    """Pick a symbol; :attr:`picked` carries it (``A:B`` for a ratio). With ``multi`` it stays open for more."""

    picked = pyqtSignal(str)

    def __init__(self, search: SymbolSearch, parent: QWidget | None = None, context: SymbolContext | None = None,
                 initial: str = "", multi: bool = False):
        super().__init__(parent, Qt.WindowType.Dialog | Qt.WindowType.FramelessWindowHint)
        self.setObjectName("SymbolDialog")
        self.setWindowTitle("Choose a symbol")
        self.search, self.multi = search, multi
        self.ctx = context or (context_for(parent) if parent is not None else None) or SymbolContext()
        self.numerator: str | None = None  # set while picking the divisor of a ratio
        self.source = "cat:all"  # the rail's selection: "cat:<category>" or "list:<name>" ("list:" alone = recent)
        self.names: dict[str, str] = {}  # names looked up for symbols the sources gave without one
        self.watched = self.ctx.watched()
        self.added: list[str] = []  # multi: what was picked so far
        self._remote: tuple[str, list[SymbolMatch]] | None = None  # (query, its network matches)
        self._browse: dict[str, list[SymbolMatch]] = {}
        self._pending = ""  # the query whose network search is running

        # search bar: [icon] [A ÷ ×] query [×]
        bar = QFrame(objectName="SymbolSearchBar")
        self.search_icon = QLabel()
        self.token = QFrame(objectName="RatioToken", visible=False)
        self.token_label = QLabel()
        self.token_close = QToolButton(toolTip="Back to one symbol (Backspace)", objectName="SymbolIconButton")
        self.token_close.clicked.connect(self.clear_ratio)
        tl = QHBoxLayout(self.token)
        tl.setContentsMargins(8, 2, 2, 2)
        tl.setSpacing(4)
        tl.addWidget(self.token_label)
        tl.addWidget(self.token_close)
        self.token.setFixedHeight(28)
        self.query = QLineEdit(initial, objectName="SymbolQuery",
                               placeholderText=PLACEHOLDER)
        self.query.setClearButtonEnabled(False)
        self.query.textEdited.connect(self._edited)
        self.close_btn = QToolButton(toolTip="Close (Esc)", objectName="SymbolIconButton")
        self.close_btn.clicked.connect(self.reject)
        bl = QHBoxLayout(bar)
        bl.setContentsMargins(14, 0, 10, 0)
        bl.setSpacing(8)
        bl.addWidget(self.search_icon)
        bl.addWidget(self.token, 0, Qt.AlignmentFlag.AlignVCenter)
        bl.addWidget(self.query, 1)
        bl.addWidget(self.close_btn)
        bar.setFixedHeight(48)
        self.ratio_note = QLabel(objectName="RatioNote", visible=False, wordWrap=True)

        # rail: categories, then the lists
        self.rail = QListWidget(objectName="SymbolRail")
        self.rail.setFixedWidth(180)
        self.rail.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.rail.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._counts: dict[str, QLabel] = {}
        for key, label in CATEGORIES.items():
            self._rail_item(f"cat:{key}", label)
        self._rail_header("Lists")
        self._rail_item("list:", "Recent")
        self._fixed_rail_rows = self.rail.count()
        self._fill_lists()
        self.rail.setCurrentRow(0)
        self.rail.currentItemChanged.connect(self._rail_changed)

        # results
        self.model = SymbolTableModel(self)
        self.table = QTableView(objectName="SymbolTable")
        self.table.setModel(self.model)
        self.table.setItemDelegateForColumn(TYPE, _TypeDelegate(self))
        self._color = _ColorDelegate(self)
        for col in (NAME, EXCHANGE, LAST, DAY):
            self.table.setItemDelegateForColumn(col, self._color)
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
        header.setSectionResizeMode(NAME, QHeaderView.ResizeMode.Stretch)
        for col, width in WIDTHS.items():
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.Fixed)
            self.table.setColumnWidth(col, width)
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)  # columns drop instead
        self.table.setSortingEnabled(True)
        header.setSortIndicatorShown(False)  # until the user sorts: the ranking is the order
        header.sortIndicatorChanged.connect(lambda *_: header.setSortIndicatorShown(True))
        self.model._sort = None
        self.table.clicked.connect(self._clicked)
        self.table.entered.connect(self._hovered)
        self.table.viewport().installEventFilter(self)
        self.table.verticalScrollBar().valueChanged.connect(lambda _: self._quote_timer.start())
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
        self.hints = QLabel(objectName="Muted")
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
        lay.addWidget(self.ratio_note)
        lay.addLayout(body, 1)
        lay.addWidget(foot)

        self._search_timer = QTimer(self, singleShot=True, interval=DEBOUNCE_MS, timeout=self._search_remote)
        self._quote_timer = QTimer(self, singleShot=True, interval=120, timeout=self._load_visible)
        self.ctx.quotes.updated.connect(self.model.refresh_symbol)
        if self.ctx.themes is not None:
            self.ctx.themes.changed.connect(self._themed)
        self.query.installEventFilter(self)
        self._themed()
        self._fit_to(parent)
        self._update_hints()
        self._refresh()
        if initial.strip():
            self._search_timer.start()

    def reset(self, search: SymbolSearch, parent: QWidget, initial: str = "", multi: bool = False):
        """Ready a reused dialog for a new pick: shown at once, filled on the next turn of the event loop."""
        self.search, self.multi = search, multi
        self._search_timer.stop()
        self.numerator, self._pending, self._remote = None, "", None
        self.token.hide()
        self.ratio_note.hide()
        self.query.setPlaceholderText(PLACEHOLDER)
        self.added = []
        self.watched = self.ctx.watched()
        self.query.setText(initial)
        self.rail.blockSignals(True)
        self._fill_lists()
        self.rail.setCurrentRow(0)
        self.rail.blockSignals(False)
        self.source = "cat:all"
        self.model.set_rows([])
        self.results.setCurrentWidget(self.table)
        self.status.setText("")
        self._update_hints()
        self._fit_to(parent)
        QTimer.singleShot(0, self._refresh)  # the window first, the rows right after
        if initial.strip():
            self._search_timer.start()

    # -- look --------------------------------------------------------------------------------------------------
    def theme(self):
        return self.ctx.themes.current if self.ctx.themes is not None else None

    def _themed(self, *_):
        t = self.theme()
        muted, text = (t.muted, t.text) if t else ("#888888", "#dddddd")
        self.search_icon.setPixmap(pixmap("search", muted, 18))
        self.close_btn.setIcon(icon("close", muted, 16))
        self.token_close.setIcon(icon("close", muted, 12))
        self.model.layoutChanged.emit()

    def _fit_to(self, parent):
        fit_dialog(self, parent)

    def _rail_item(self, key: str, label: str):
        self._counts[key] = rail_item(self.rail, key, label)

    def _fill_lists(self):
        """The watchlists under Recent, as they are now."""
        while self.rail.count() > self._fixed_rail_rows:
            item = self.rail.takeItem(self.rail.count() - 1)
            self._counts.pop(item.data(Qt.ItemDataRole.UserRole), None)
        for name in self.ctx.lists():
            self._rail_item(f"list:{name}", name)

    def _rail_header(self, label: str):
        rail_header(self.rail, label)

    def _update_hints(self):
        self.hints.setText(HINTS_RATIO if self.numerator else HINTS_MULTI if self.multi else HINTS)

    # -- what the table shows ----------------------------------------------------------------------------------
    def _local_pool(self) -> list[SymbolMatch]:
        """The user's own symbols: recent first, then the watchlists and portfolio."""
        symbols = dict.fromkeys([*self.ctx.recent(), *self.search.local_symbols()])
        return [SymbolMatch(s, kind="watchlist") for s in symbols]

    def _list_symbols(self, name: str) -> list[str]:
        return self.ctx.recent() if not name else list(self.ctx.lists().get(name, []))

    def _pool(self) -> list[SymbolMatch]:
        """Everything that matches the query, before the category filter."""
        q = self.query.text().strip()
        kind, _, name = self.source.partition(":")
        if kind == "list":
            pool = [SymbolMatch(s, kind="watchlist") for s in self._list_symbols(name)]
            return [m for m in pool if rank(q, m) is not None] if q else pool
        if not q:
            if name in ("all", ""):
                return self._local_pool()
            if name not in self._browse:
                self._browse[name] = []
                run_async(lambda c=name: self.search.browse(c)[:BROWSE_MAX], lambda found, c=name: self._browsed(c, found),
                          lambda err: None)
            own = [m for m in self._local_pool() if category(m) == name]
            seen = {m.symbol for m in own}
            return own + [m for m in self._browse[name] if m.symbol not in seen]
        local = [m for m in self._local_pool() if rank(q, m) is not None]
        remote = self._remote[1] if self._remote and self._remote[0] == q else []
        details = {m.symbol: m for m in remote}
        own = [details.get(m.symbol, m) for m in local]  # the user's symbols, with the source's name and type
        seen = {m.symbol for m in own}
        merged = own + [m for m in remote if m.symbol not in seen]
        return sorted(merged, key=lambda m: rank(q, m))  # stable: on a tie the user's own symbol comes first

    def _refresh(self):
        q = self.query.text().strip()
        pool = self._pool()
        if self.numerator:
            pool = [m for m in pool if can_divide(self.numerator, m.symbol)]
        kind, _, name = self.source.partition(":")
        rows = [m for m in pool if kind == "list" or name == "all" or category(m) == name]
        for key, label in self._counts.items():
            k, _, n = key.partition(":")
            label.setText(str(sum(1 for m in pool if n == "all" or category(m) == n)) if q and k == "cat" else "")
        self.model.set_rows(rows)
        if rows:
            self.results.setCurrentWidget(self.table)
            self.select_row(0)
            self.table.scrollToTop()
            self._quote_timer.start()
        else:
            self.results.setCurrentWidget(self.empty)
            self.empty.setText(self._empty_text(q, kind, name))
        self._status()

    def _empty_text(self, q: str, kind: str, name: str) -> str:
        if self._pending and self._pending == q:
            return "Searching…"
        if q:
            extra = ""
            if self.numerator:
                extra = (" A ratio needs two crypto pairs, or two symbols that aren't crypto pairs."
                         if any(not can_divide(self.numerator, m.symbol) for m in self._pool()) else "")
            return (f"No symbol matches “{q}”.{extra}\nTry the company or coin name, or check the spelling. "
                    "Offline, only your lists and recent symbols are searched.")
        if kind == "list":
            return "No recent symbols yet." if not name else f"“{name}” has no symbols yet."
        if name in self._browse and name == "crypto":
            return "Loading the exchange's pairs…"
        return f"Type a ticker or a name to search {CATEGORIES.get(name, 'symbols').lower()}."

    def _status(self):
        if self.multi and self.added:
            self.status.setText("Added " + ", ".join(self.added[-4:]) + (" …" if len(self.added) > 4 else ""))
        elif self._pending:
            self.status.setText("Searching…")
        else:
            self.status.setText("")

    def _browsed(self, name: str, found: list[SymbolMatch]):
        self._browse[name] = found
        if self.source == f"cat:{name}" and not self.query.text().strip():
            self._refresh()

    # -- searching ---------------------------------------------------------------------------------------------
    def _edited(self, text: str):
        if self.numerator is None:  # "BTC/USD / " (a slash between spaces): the first part becomes the numerator
            head, sep, tail = text.rpartition(" / ")
            if sep and not tail.strip() and self._divide_by_text(head):
                return
        self._refresh()
        if len(text.strip()) >= 1:
            self._search_timer.start()  # restarts while typing: one network search per pause
        else:
            self._search_timer.stop()
            self._pending = ""

    def _divide_by_text(self, text: str) -> bool:
        target = _compact(text)
        match = next((m for m in self.model.rows if _compact(m.symbol) == target), None)
        if match is None:
            return False
        self.start_ratio(match.symbol)
        return True

    def _search_remote(self):
        query = self._pending = self.query.text().strip()
        if not query:
            return
        self._status()
        run_async(lambda: self.search.remote_matches(query)[:SEARCH_LIMIT], lambda found: self._searched(query, found),
                  lambda err: self._searched(query, []))

    def _searched(self, query: str, found: list[SymbolMatch]):
        if query != self.query.text().strip():
            return  # the text changed meanwhile: a newer search is coming
        self._pending = ""
        self._remote = (query, found)
        current = self.current_match() if self.current_row() > 0 else None
        self._refresh()
        if current is not None:  # keep a row the user moved to when the network results arrive
            for i, m in enumerate(self.model.rows):
                if m.symbol == current.symbol:
                    self.select_row(i)
                    break

    # -- prices and names --------------------------------------------------------------------------------------
    def _load_visible(self):
        if not self.model.rows:
            return
        first = max(0, self.table.rowAt(0))
        last = self.table.rowAt(self.table.viewport().height() - 1)
        last = len(self.model.rows) - 1 if last < 0 else last
        for m in self.model.rows[first:last + 1]:
            self.ctx.quotes.request(m.symbol)
            if not m.name and m.symbol not in self.names and split_pair(m.symbol) is None:
                self.names[m.symbol] = ""  # asked once
                run_async(lambda s=m.symbol: symbol_name(s), lambda name, s=m.symbol: self._named(s, name),
                          lambda err: None)

    def _named(self, symbol: str, name: str | None):
        if name:
            self.names[symbol] = name
            self.model.refresh_symbol(symbol)

    # -- selection, picking, ratios ----------------------------------------------------------------------------
    def current_row(self) -> int:
        return self.table.currentIndex().row()

    def current_match(self) -> SymbolMatch | None:
        row = self.current_row()
        return self.model.rows[row] if 0 <= row < len(self.model.rows) else None

    def select_row(self, row: int):
        if not self.model.rows:
            return
        row = max(0, min(row, len(self.model.rows) - 1))
        self.table.setCurrentIndex(self.model.index(row, SYMBOL))
        self.table.scrollTo(self.model.index(row, SYMBOL))
        self.model.dataChanged.emit(self.model.index(0, DIVIDE), self.model.index(len(self.model.rows) - 1, DIVIDE))

    def _clicked(self, index: QModelIndex):
        m = self.model.rows[index.row()]
        if index.column() == DIVIDE and self.numerator is None and ":" not in m.symbol:
            self.start_ratio(m.symbol)
        else:
            self.pick(m.symbol)

    def _hovered(self, index: QModelIndex):
        old, self.model.hover = self.model.hover, index.row()
        for row in {old, index.row()} - {-1}:
            if row < len(self.model.rows):
                self.model.dataChanged.emit(self.model.index(row, DIVIDE), self.model.index(row, DIVIDE))

    def pick(self, symbol: str):
        full = f"{self.numerator}:{symbol}" if self.numerator else symbol
        self.ctx.remember(full)
        self.picked.emit(full)
        if not self.multi:
            self.accept()
            return
        self.added.append(full)
        self.watched = self.ctx.watched()
        self.clear_ratio()
        self.query.selectAll()
        self._status()

    def start_ratio(self, symbol: str):
        """``symbol`` becomes the numerator; the next pick divides it."""
        self.numerator = symbol
        self.token_label.setText(f"{symbol}  ÷")
        self.token.show()
        self.ratio_note.setText(f"Ratio: {symbol} divided by the symbol you pick next. Backspace goes back.")
        self.ratio_note.show()
        self.query.setPlaceholderText("Divide by… e.g. ETH/USD")
        self.query.clear()
        self._remote = None
        self._update_hints()
        self._refresh()

    def clear_ratio(self):
        if self.numerator is None:
            return
        self.numerator = None
        self.token.hide()
        self.ratio_note.hide()
        self.query.setPlaceholderText(PLACEHOLDER)
        self._update_hints()
        self._refresh()

    def _wants_divide(self) -> bool:
        """``/`` divides when the query names the highlighted symbol ("BTCUSD/", "KO/") or the user moved the highlight;
        otherwise it is typed, so "EUR/USD" still works."""
        m, q = self.current_match(), self.query.text().strip()
        if self.numerator is not None or m is None or not q or ":" in m.symbol:
            return False
        return _compact(q) == _compact(m.symbol) or self.current_row() > 0

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

    def _next_category(self, step: int):
        keys = [f"cat:{k}" for k in CATEGORIES]
        i = keys.index(self.source) if self.source in keys else -1
        self.set_source(keys[(i + step) % len(keys)])

    # -- keyboard ----------------------------------------------------------------------------------------------
    def eventFilter(self, obj, event):
        if obj is self.query and event.type() == QEvent.Type.KeyPress:
            return self._key(event)
        if obj is self.table.viewport() and event.type() == QEvent.Type.Resize:
            self._fit_columns()
        if obj is self.table.viewport() and event.type() == QEvent.Type.Leave and self.model.hover >= 0:
            self._hovered(self.model.index(-1, 0))
        return super().eventFilter(obj, event)

    def _key(self, event: QKeyEvent) -> bool:
        key = event.key()
        page = max(1, self.table.viewport().height() // 36 - 1)
        moves = {Qt.Key.Key_Down: 1, Qt.Key.Key_Up: -1, Qt.Key.Key_PageDown: page, Qt.Key.Key_PageUp: -page}
        if key in moves:
            self.select_row(self.current_row() + moves[key])
            self._quote_timer.start()
            return True
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            m = self.current_match()
            if m is not None:
                self.pick(m.symbol)
            return True
        if key == Qt.Key.Key_Tab:
            self._next_category(1)
            return True
        if key == Qt.Key.Key_Backtab:
            self._next_category(-1)
            return True
        if event.text() == "/" and self._wants_divide():
            self.start_ratio(self.current_match().symbol)
            return True
        if key == Qt.Key.Key_Backspace and self.numerator is not None and not self.query.text():
            numerator = self.numerator
            self.clear_ratio()
            self.query.setText(numerator)
            self._edited(numerator)
            self._search_timer.stop()
            self._search_remote()  # no pause to wait for: the symbol is back as it was
            return True
        return False

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.reject()
            return
        super().keyPressEvent(event)

    def _fit_columns(self):
        """In a small window Exchange goes first, then Type, so the name keeps room to be read."""
        room = self.table.viewport().width() - NAME_MIN
        fixed = sum(WIDTHS.values())
        for col in (EXCHANGE, TYPE):
            hide = fixed > room
            self.table.setColumnHidden(col, hide)
            if hide:
                fixed -= WIDTHS[col]

    def showEvent(self, event):
        super().showEvent(event)
        self.query.setFocus()
        QTimer.singleShot(0, lambda: self.query.end(False))  # the caret after the typed text, nothing selected


def open_symbol_dialog(search: SymbolSearch, parent: QWidget, on_pick: Callable[[str], None], initial: str = "",
                       multi: bool = False) -> SymbolDialog:
    """Show the dialog over ``parent``'s window; ``on_pick`` gets each chosen symbol.

    With a :class:`SymbolContext` the window keeps one dialog and reuses it, so it opens without being built."""
    ctx = context_for(parent)
    window = parent.window()
    if ctx is None:
        dialog = SymbolDialog(search, window, None, initial, multi)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
    else:
        dialog = ctx.dialog(search, window)
        dialog.reset(search, window, initial, multi)
        try:
            dialog.picked.disconnect()  # the field it was opened for last time
        except TypeError:
            pass
    dialog.picked.connect(on_pick)
    dialog.open()
    return dialog
