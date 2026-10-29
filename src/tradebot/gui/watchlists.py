# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""The chart sidebar's watchlists (#67): named lists of symbols with their last price; a click charts one.

The lists live in config.toml's ``[watchlists]`` (see :mod:`tradebot.config`), so they stay editable by hand;
which one is shown is remembered in settings.json.
"""

from __future__ import annotations

import html

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from tradebot.config import Settings, delete_watchlist, save_watchlist, watchlists
from tradebot.gui.common import SymbolCompleter, run_async
from tradebot.gui.theme import ThemeManager
from tradebot.gui.touch import LongPressTooltips, enable_kinetic_scroll
from tradebot.symbols import SymbolSearch, symbol_name

REFRESH_MS = 60 * 1000
SELECTED_KEY = "chart.watchlist"
QUOTE_BARS = 5  #: daily candles per price lookup: the last close and the one before, with room for holidays
NAME_CHARS = 30
EMPTY = "No symbols yet. Add one above or from the chart."


class _SymbolList(QListWidget):
    """Drag and drop reorders; Delete removes the current row."""

    reordered = pyqtSignal()
    delete_pressed = pyqtSignal()

    def dropEvent(self, event):
        super().dropEvent(event)
        self.reordered.emit()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace) and self.currentItem() is not None:
            self.delete_pressed.emit()
            return
        super().keyPressEvent(event)


class _Row(QWidget):
    """One symbol: ticker and name on the left, price and day change on the right, ✕ on hover (always on touch)."""

    def __init__(self, symbol: str, on_remove, touch: bool):
        super().__init__()
        self.touch = touch
        self.label = QLabel()
        self.quote = QLabel(alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.remove = QToolButton(text="✕", toolTip=f"Remove {symbol}", objectName="WatchlistRemove")
        self.remove.clicked.connect(lambda: on_remove(symbol))
        self.remove.setVisible(touch)
        for w in (self, self.label, self.quote):
            w.setStyleSheet("background: transparent;")
        lay = QHBoxLayout(self)
        m = 12 if touch else 6
        lay.setContentsMargins(8, m, 4, m)
        lay.addWidget(self.label, 1)
        lay.addWidget(self.quote)
        lay.addWidget(self.remove)

    def enterEvent(self, event):
        super().enterEvent(event)
        self.remove.show()

    def leaveEvent(self, event):
        super().leaveEvent(event)
        self.remove.setVisible(self.touch)


class WatchlistPanel(QWidget):
    """The watchlists: pick a list, add, remove and reorder its symbols, click one to chart it."""

    #: chart this symbol
    open_symbol = pyqtSignal(str)
    #: the lists changed (names or symbols)
    changed = pyqtSignal()
    #: a note for the status bar, e.g. a failed save
    status = pyqtSignal(str)

    def __init__(self, engine, themes: ThemeManager, settings: Settings | None = None,
                 symbol_search: SymbolSearch | None = None, parent=None):
        super().__init__(parent)
        self.engine, self.themes, self.settings = engine, themes, settings
        self.lists: dict[str, list[str]] = watchlists()
        self.charted = ""  # the chart's symbol: highlighted, and what "Add current symbol" adds
        self.names: dict[str, str] = {}  # symbol -> full name
        self.quotes: dict[str, tuple[float, float | None, bool]] = {}  # symbol -> (last, day change, fresh)
        self._visible = False

        self.selector = QComboBox(toolTip="The watchlist shown")
        self.new_btn = QToolButton(text="＋", toolTip="New list…")
        self.rename_btn = QToolButton(text="✎", toolTip="Rename this list…")
        self.delete_btn = QToolButton(text="🗑", toolTip="Delete this list")
        self.new_btn.clicked.connect(self.new_list)
        self.rename_btn.clicked.connect(self.rename_list)
        self.delete_btn.clicked.connect(self.delete_list)
        self.field = QLineEdit(placeholderText="Add a symbol…")
        self.completer = SymbolCompleter(self.field, symbol_search or SymbolSearch(self.all_symbols))
        self.completer.chosen.connect(self.add_symbol)
        self.field.returnPressed.connect(lambda: self.add_symbol(self.field.text()))
        self.add_btn = QToolButton(text="＋", toolTip="Add the symbol to this list")
        self.add_btn.clicked.connect(lambda: self.add_symbol(self.field.text()))
        self.add_current_btn = QPushButton("Add current symbol", toolTip="Add the symbol on the chart")
        self.add_current_btn.clicked.connect(lambda: self.add_symbol(self.charted))

        self.list = _SymbolList(objectName="Watchlist")
        self.list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.list.setSpacing(1)
        self.list.itemClicked.connect(lambda item: self.open_symbol.emit(item.data(Qt.ItemDataRole.UserRole)))
        self.list.reordered.connect(self._reordered)
        self.list.delete_pressed.connect(lambda: self.remove_symbol(self.list.currentItem().data(Qt.ItemDataRole.UserRole)))
        self.list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.list.customContextMenuRequested.connect(lambda pos: self._row_menu(self.list.viewport().mapToGlobal(pos)))
        enable_kinetic_scroll(self.list)
        self._long_press = LongPressTooltips(self.list.viewport(), self._row_menu)  # touch: the row's menu
        self.empty = QLabel(EMPTY, objectName="Muted", wordWrap=True, alignment=Qt.AlignmentFlag.AlignCenter)
        self.body = QStackedWidget()
        self.body.addWidget(self.empty)
        self.body.addWidget(self.list)

        head = QHBoxLayout()
        head.addWidget(self.selector, 1)
        for b in (self.new_btn, self.rename_btn, self.delete_btn):
            head.addWidget(b)
        add = QHBoxLayout()
        add.addWidget(self.field, 1)
        add.addWidget(self.add_btn)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addLayout(head)
        lay.addLayout(add)
        lay.addWidget(self.add_current_btn)
        lay.addWidget(self.body, 1)

        self.timer = QTimer(self, interval=REFRESH_MS, timeout=self.refresh)
        themes.changed.connect(lambda _: self._render())
        self._fill_selector(settings.get(SELECTED_KEY) if settings else None)
        self.selector.currentTextChanged.connect(self._list_chosen)

    # -- the lists -------------------------------------------------------------------------------------
    @property
    def current(self) -> str:
        return self.selector.currentText()

    def symbols(self) -> list[str]:
        """The shown list's symbols."""
        return list(self.lists.get(self.current, []))

    def all_symbols(self) -> list[str]:
        """Every list's symbols, de-duplicated, in order: for autocomplete and the other pages."""
        return list(dict.fromkeys(s for symbols in self.lists.values() for s in symbols))

    def select(self, name: str):
        self.selector.setCurrentText(name)

    def new_list(self):
        name = self._ask_name("New watchlist", "Name of the new list:", "")
        if name and self._save(name, []):
            self.lists[name] = []
            self._fill_selector(name)
            self.changed.emit()

    def rename_list(self):
        old = self.current
        name = self._ask_name("Rename watchlist", f"New name for “{old}”:", old)
        if not name or name == old or not self._save(name, self.lists[old], rename=old):
            return
        self.lists = {name if n == old else n: syms for n, syms in self.lists.items()}  # keeps the order
        self._fill_selector(name)
        self.changed.emit()

    def delete_list(self):
        name = self.current
        if len(self.lists) <= 1:
            QMessageBox.information(self, "Delete watchlist", "The last list can't be deleted.")
            return
        count = len(self.lists[name])
        answer = QMessageBox.question(self, "Delete watchlist",
                                      f"Delete the list “{name}” ({count} symbol{'s' * (count != 1)})?",
                                      QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                      QMessageBox.StandardButton.No)
        if answer != QMessageBox.StandardButton.Yes:
            return
        if not delete_watchlist(name):
            self.status.emit(f"watchlist “{name}” not deleted: couldn't write the config file")
            return
        del self.lists[name]
        self._fill_selector(None)
        self.changed.emit()

    def _ask_name(self, title: str, label: str, text: str) -> str | None:
        name, ok = QInputDialog.getText(self, title, label, text=text)
        name = name.strip()
        if not ok or not name:
            return None
        if name in self.lists and name != text:
            QMessageBox.information(self, title, f"There already is a list “{name}”.")
            return None
        return name

    def _fill_selector(self, name: str | None):
        self.selector.blockSignals(True)
        self.selector.clear()
        self.selector.addItems(list(self.lists))
        self.selector.setCurrentText(name if name in self.lists else next(iter(self.lists)))
        self.selector.blockSignals(False)
        self._list_chosen(self.current)

    def _list_chosen(self, name: str):
        if self.settings and name:
            self.settings.set(SELECTED_KEY, name)
        self.delete_btn.setEnabled(len(self.lists) > 1)
        self._render()
        if self._visible:
            self.refresh()

    # -- symbols ---------------------------------------------------------------------------------------
    def add_symbol(self, symbol: str, name: str | None = None):
        """Add ``symbol`` to the list ``name`` (default: the shown one); already there: nothing changes."""
        symbol, name = symbol.strip(), name or self.current
        if not symbol:
            return
        self.field.clear()
        if symbol in self.lists[name]:
            self.status.emit(f"{symbol} is already in “{name}”")
            return
        if self._save(name, self.lists[name] + [symbol]):
            self.lists[name].append(symbol)
            self.changed.emit()
            if name == self.current:
                self._render()
                if self._visible:
                    self._load_quote(symbol)

    def remove_symbol(self, symbol: str):
        symbols = [s for s in self.lists[self.current] if s != symbol]
        if self._save(self.current, symbols):
            self.lists[self.current] = symbols
            self.changed.emit()
            self._render()

    def move_symbol(self, symbol: str, step: int):
        """Move ``symbol`` ``step`` rows down (negative: up)."""
        symbols = self.symbols()
        i = symbols.index(symbol)
        j = max(0, min(len(symbols) - 1, i + step))
        symbols.insert(j, symbols.pop(i))
        self._set_order(symbols)

    def _reordered(self):
        """A row was dropped: the list's new order is what the rows show."""
        self._set_order([self.list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.list.count())])

    def _set_order(self, symbols: list[str]):
        if symbols != self.lists[self.current] and self._save(self.current, symbols):
            self.lists[self.current] = symbols
            self.changed.emit()
        self._render()  # also rebuilds rows a drop may have left without their widgets

    def _save(self, name: str, symbols: list[str], rename: str | None = None) -> bool:
        if save_watchlist(name, symbols, rename=rename):
            return True
        self.status.emit(f"watchlist “{name}” not saved: couldn't write the config file")
        return False

    def _row_menu(self, global_pos):
        """Right click, or long press on a touchscreen."""
        item = self.list.itemAt(self.list.viewport().mapFromGlobal(global_pos))
        if item is None:
            return
        symbol = item.data(Qt.ItemDataRole.UserRole)
        menu = QMenu(self)
        menu.addAction(f"Chart {symbol}", lambda: self.open_symbol.emit(symbol))
        menu.addAction("Move up", lambda: self.move_symbol(symbol, -1))
        menu.addAction("Move down", lambda: self.move_symbol(symbol, 1))
        menu.addSeparator()
        menu.addAction(f"Remove {symbol}", lambda: self.remove_symbol(symbol))
        menu.popup(global_pos)
        self._menu = menu  # keep a reference while open (and for tests)

    # -- the chart -------------------------------------------------------------------------------------
    def set_symbol(self, symbol: str):
        """The chart shows ``symbol`` now: highlight its row."""
        self.charted = symbol
        self.add_current_btn.setEnabled(bool(symbol))
        self._highlight()

    def _highlight(self):
        for i in range(self.list.count()):
            if self.list.item(i).data(Qt.ItemDataRole.UserRole) == self.charted:
                self.list.setCurrentRow(i)
                return
        self.list.setCurrentRow(-1)
        self.list.clearSelection()

    # -- prices ----------------------------------------------------------------------------------------
    def set_visible(self, on: bool):
        """Whether the sidebar shows this page: prices are only loaded (and refreshed) while it does."""
        self._visible = on
        if not on:
            self.timer.stop()
            return
        self.timer.start()
        self.refresh()

    def refresh(self):
        for symbol in self.symbols():
            self._load_quote(symbol)

    def _load_quote(self, symbol: str):
        """The disk cache's daily candles right away, then current ones; the full name once."""
        engine = self.engine
        if engine is None:
            return
        from tradebot.data import MarketDataEngine

        if isinstance(engine, MarketDataEngine):  # its memory cache would keep the day's first price all day
            engine = MarketDataEngine(engine.providers, cache=False, disk=engine.disk)
        if symbol not in self.quotes:
            run_async(lambda: engine.peek(symbol, "1d", bars=QUOTE_BARS), lambda df: self._quoted(symbol, df, False))
        run_async(lambda: engine.get_ohlc(symbol, "1d", bars=QUOTE_BARS), lambda df: self._quoted(symbol, df, True),
                  lambda err: None)  # offline: the cached price stays
        if symbol not in self.names:
            run_async(lambda: symbol_name(symbol), lambda name: self._named(symbol, name))

    def _quoted(self, symbol: str, df, fresh: bool):
        if df is None or df.empty or (not fresh and self.quotes.get(symbol, (0, 0, False))[2]):
            return  # nothing cached, or the current price already came
        close = df["close"]
        change = float(close.iloc[-1] / close.iloc[-2] - 1) if len(close) > 1 and close.iloc[-2] else None
        self.quotes[symbol] = (float(close.iloc[-1]), change, fresh)
        self._update_row(symbol)

    def _named(self, symbol: str, name: str | None):
        if name:
            self.names[symbol] = name
            self._update_row(symbol)

    # -- rendering -------------------------------------------------------------------------------------
    def _render(self):
        self.list.clear()
        for symbol in self.symbols():
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, symbol)
            row = _Row(symbol, self.remove_symbol, self.themes.touch)
            self.list.addItem(item)
            self.list.setItemWidget(item, row)
            self._update_row(symbol)
        self.body.setCurrentWidget(self.list if self.list.count() else self.empty)
        self._highlight()

    def _row(self, symbol: str) -> tuple[QListWidgetItem, _Row] | None:
        for i in range(self.list.count()):
            item = self.list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == symbol:
                return item, self.list.itemWidget(item)
        return None

    def _update_row(self, symbol: str):
        found = self._row(symbol)
        if found is None or found[1] is None:
            return
        item, row = found
        t = self.themes.current
        name = self.names.get(symbol, "")
        if len(name) > NAME_CHARS:
            name = name[: NAME_CHARS - 1] + "…"
        row.label.setText(f"<b>{html.escape(symbol)}</b><br><span style='color:{t.muted}'>{html.escape(name)}</span>"
                          if name else f"<b>{html.escape(symbol)}</b>")
        quote = self.quotes.get(symbol)
        if quote is None:
            row.quote.setText(f"<span style='color:{t.muted}'>–</span>")
        else:
            last, change, _ = quote
            text = f"<b>{fmt_price(last)}</b>"
            if change is not None:
                text += f"<br><span style='color:{t.up if change >= 0 else t.down}'>{change * 100:+.2f}%</span>"
            row.quote.setText(text)
        item.setSizeHint(row.sizeHint())


def fmt_price(v: float) -> str:
    """Enough decimals for small prices: ``187.40``, ``0.08734``."""
    if v >= 1000:
        return f"{v:,.2f}"
    if v >= 1:
        return f"{v:.2f}"
    return f"{v:.4g}"
