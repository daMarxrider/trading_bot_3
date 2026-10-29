# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Headlines in the GUI: the shared item rendering, and the chart sidebar's news for the charted symbol (#41)."""

from __future__ import annotations

import html
from datetime import datetime, timezone

from PyQt6.QtCore import QSize, Qt, QTimer, QUrl
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from tradebot import news
from tradebot.gui.common import run_async
from tradebot.gui.theme import ThemeManager
from tradebot.gui.touch import enable_kinetic_scroll

REFRESH_MS = 5 * 60 * 1000


def ago(when: datetime, now: datetime | None = None) -> str:
    """``5 min ago`` · ``3 h ago`` · ``2 d ago``, then the date."""
    seconds = ((now or datetime.now(timezone.utc)) - when).total_seconds()
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        return f"{int(seconds // 60)} min ago"
    if seconds < 86400:
        return f"{int(seconds // 3600)} h ago"
    if seconds < 7 * 86400:
        return f"{int(seconds // 86400)} d ago"
    return f"{when:%Y-%m-%d}"


def fill_news_list(widget: QListWidget, items, themes: ThemeManager, show_symbol: bool = True,
                   summary_chars: int = 240, title_px: int = 15, relative: bool = False):
    """One rich-text row per headline; the URL is the item's data (opened by the caller)."""
    t = themes.current
    widget.clear()
    width = max(160, widget.viewport().width() - 12)
    for n in items:
        when = ago(n.published) if relative else f"{n.published:%Y-%m-%d %H:%M}"
        meta = " · ".join(p for p in (when, html.escape(n.publisher)) if p)
        symbol = f"<span style='color:{t.accent}; font-weight:600'>{html.escape(n.symbol)}</span>&nbsp;&nbsp;" \
            if show_symbol else ""
        label = QLabel(
            f"{symbol}<span style='color:{t.muted}'>{meta}</span>"
            f"<br><span style='font-size:{title_px}px; font-weight:600'>{html.escape(n.title)}</span>"
            + (f"<br><span style='color:{t.muted}'>{html.escape(n.summary[:summary_chars])}</span>"
               if n.summary and summary_chars else ""))
        label.setWordWrap(True)
        label.setContentsMargins(10, 8, 10, 8)
        label.setStyleSheet("background: transparent;")
        item = QListWidgetItem()
        item.setData(Qt.ItemDataRole.UserRole, n.url)
        item.setToolTip(n.url)
        item.setSizeHint(QSize(width, label.heightForWidth(width) + 4))
        widget.addItem(item)
        widget.setItemWidget(item, label)


def open_link(item: QListWidgetItem):
    url = item.data(Qt.ItemDataRole.UserRole)
    if url:
        QDesktopServices.openUrl(QUrl(url))


class NewsPanel(QWidget):
    """News for one symbol: follows the chart, loads in the background, only while it's visible."""

    def __init__(self, themes: ThemeManager, parent=None):
        super().__init__(parent)
        self.themes = themes
        self.symbol = ""
        self.items: list = []
        self._loaded_for = None  # the symbol self.items belong to
        self._visible = False
        self.header = QLabel(objectName="Muted", wordWrap=True)
        self.refresh_btn = QToolButton(text="⟳", toolTip="Refresh the news")
        self.refresh_btn.clicked.connect(self.refresh)
        self.list = QListWidget(objectName="SidebarNews")
        self.list.setWordWrap(True)
        self.list.setSpacing(2)
        self.list.itemClicked.connect(open_link)  # one click or tap: a sidebar is no place for double-clicks
        enable_kinetic_scroll(self.list)
        self.message = QLabel(objectName="Muted", wordWrap=True, alignment=Qt.AlignmentFlag.AlignCenter)
        self.body = QStackedWidget()
        self.body.addWidget(self.message)
        self.body.addWidget(self.list)
        head = QHBoxLayout()
        head.addWidget(self.header, 1)
        head.addWidget(self.refresh_btn)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addLayout(head)
        lay.addWidget(self.body, 1)
        self.timer = QTimer(self, interval=REFRESH_MS, timeout=self.refresh)
        themes.changed.connect(lambda _: self._render())
        self._show_message("Load a chart to see its news.")

    def set_symbol(self, symbol: str, name: str | None = None):
        """The chart shows ``symbol`` now; ``name`` is its full name, when known."""
        self.header.setText(html.escape(f"{name} ({symbol})" if name and name != symbol else symbol))
        if symbol == self.symbol:
            return
        self.symbol = symbol
        if self._visible:
            self.refresh()

    def set_visible(self, on: bool):
        """Whether the sidebar shows this page: news is only fetched (and refreshed) while it does."""
        self._visible = on
        if not on:
            self.timer.stop()
            return
        self.timer.start()
        if self.symbol and self._loaded_for != self.symbol:
            self.refresh()

    def refresh(self):
        symbol = self.symbol
        if not symbol:
            return
        if self._loaded_for != symbol:
            self._show_message(f"Loading news for {symbol} …")
        fetch = news.fetch_news  # bound now, not when the worker thread gets to it
        run_async(lambda: fetch([symbol], 30), lambda items: self._loaded(symbol, items),
                  lambda err: self._failed(symbol, err))

    def _loaded(self, symbol, items):
        if symbol != self.symbol:
            return  # the chart moved on to another symbol meanwhile
        self.items, self._loaded_for = items, symbol
        self._render()

    def _failed(self, symbol, err):
        if symbol == self.symbol and self._loaded_for != symbol:
            self._show_message(f"News for {symbol} is unavailable right now.\n{err}")

    def _render(self):
        if self._loaded_for is None:
            return
        if not self.items:
            self._show_message(f"No news for {self._loaded_for}.")
            return
        self.body.setCurrentWidget(self.list)
        fill_news_list(self.list, self.items, self.themes, show_symbol=False, summary_chars=160, title_px=13,
                       relative=True)

    def _show_message(self, text: str):
        self.message.setText(text)
        self.body.setCurrentWidget(self.message)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.body.currentWidget() is self.list:
            self._render()  # re-wrap the headlines to the new width
