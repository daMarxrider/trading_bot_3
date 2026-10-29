# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Collapsible sidebar at the right of the chart (#41), with one page per category.

A strip at the right edge holds the toggle on top (» collapses the sidebar to the right, « expands it) and
one button per category below it; the open category's page sits to the left of the strip. A new category
is one :meth:`ChartSidebar.add_category` call. Open/closed and the category are remembered in settings.json.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QButtonGroup, QFrame, QHBoxLayout, QLabel, QStackedWidget, QToolButton, QVBoxLayout, QWidget

from tradebot.config import Settings

OPEN_KEY, CATEGORY_KEY = "chart.sidebar.open", "chart.sidebar.category"
PANEL_WIDTH = 340


class ChartSidebar(QFrame):
    #: a category's page became visible (opened, or switched to while open): the key
    shown = pyqtSignal(str)

    def __init__(self, settings: Settings | None = None, touch: bool = False, parent=None):
        super().__init__(parent, objectName="ChartSidebar")
        self.settings = settings
        self.keys: list[str] = []
        self.button_size = 52 if touch else 42  # the strip's square buttons; finger-sized in touch mode
        self.toggle = QToolButton(objectName="SidebarToggle", toolTip="Hide the sidebar")
        self._square(self.toggle)
        self.toggle.clicked.connect(lambda: self.set_open(not self.is_open))
        self.group = QButtonGroup(self)
        self.title = QLabel(objectName="SidebarTitle")
        self.pages = QStackedWidget()

        self.strip = QVBoxLayout()
        self.strip.setContentsMargins(4, 6, 4, 6)
        self.strip.setSpacing(4)
        self.strip.addWidget(self.toggle)
        self.strip.addStretch()
        self.panel = QWidget(objectName="SidebarPanel")
        self.panel.setFixedWidth(PANEL_WIDTH)
        panel = QVBoxLayout(self.panel)
        panel.setContentsMargins(12, 10, 6, 10)
        panel.addWidget(self.title)
        panel.addWidget(self.pages, 1)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self.panel)
        lay.addLayout(self.strip)
        self._open = bool(settings.get(OPEN_KEY, True)) if settings else True
        self._apply()

    def add_category(self, key: str, icon: str, name: str, page: QWidget):
        btn = QToolButton(text=icon, objectName="SidebarTab", checkable=True, toolTip=name)
        self._square(btn)
        btn.clicked.connect(lambda: self.show_category(key))
        self.group.addButton(btn)
        self.strip.insertWidget(len(self.keys) + 1, btn)  # below the toggle, in the order added
        self.keys.append(key)
        page.setProperty("sidebar_name", name)
        self.pages.addWidget(page)
        saved = self.settings.get(CATEGORY_KEY) if self.settings else None
        if len(self.keys) == 1 or key == saved:
            self._select(key)

    @property
    def is_open(self) -> bool:
        return self._open

    @property
    def current(self) -> str | None:
        return self.keys[self.pages.currentIndex()] if self.keys else None

    def set_open(self, on: bool):
        self._open = on
        if self.settings:
            self.settings.set(OPEN_KEY, on)
        self._apply()
        if on and self.current:
            self.shown.emit(self.current)

    def show_category(self, key: str):
        """Open the sidebar on that category (a category button also opens a collapsed sidebar)."""
        self._select(key)
        if self.settings:
            self.settings.set(CATEGORY_KEY, key)
        if not self._open:
            self.set_open(True)
        else:
            self.shown.emit(key)

    def page(self, key: str) -> QWidget:
        return self.pages.widget(self.keys.index(key))

    def _select(self, key: str):
        i = self.keys.index(key)
        self.pages.setCurrentIndex(i)
        self.group.buttons()[i].setChecked(True)
        self.title.setText(self.pages.widget(i).property("sidebar_name"))

    def _square(self, button: QToolButton):
        """A fixed square: set on the button itself, since the theme's min-height would override setFixedSize."""
        n = self.button_size
        button.setStyleSheet(f"min-width: {n}px; max-width: {n}px; min-height: {n}px; max-height: {n}px;")

    def _apply(self):
        self.panel.setVisible(self._open)
        self.toggle.setText("»" if self._open else "«")  # » pushes it closed to the right, « pulls it open
        self.toggle.setToolTip("Hide the sidebar" if self._open else "Show the sidebar")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
