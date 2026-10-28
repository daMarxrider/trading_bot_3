# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Desktop application entry point and main window."""

from __future__ import annotations

import sys

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from tradebot import __version__
from tradebot.config import Settings, config_path, load_config, save_config_value, watchlist
from tradebot.core.registry import load_plugins
from tradebot.data import MarketDataEngine
from tradebot.data.providers import quiet_yfinance
from tradebot.gui.chart import ChartView
from tradebot.gui.theme import GlassBackdrop, ThemeManager
from tradebot.gui.touch import has_touchscreen
from tradebot.gui.views import BacktestView, NewsView, PortfolioView, ScreenerView
from tradebot.symbols import SymbolSearch


class MainWindow(QMainWindow):
    def __init__(self, themes: ThemeManager, theme_key: str = "glass", settings: Settings | None = None):
        super().__init__()
        self.setWindowTitle("tradebot")
        self.resize(1440, 900)
        self.themes = themes
        engine = MarketDataEngine()
        symbols = watchlist()
        self.portfolio_symbols: list[str] = []

        search = SymbolSearch(lambda: symbols + self.portfolio_symbols)  # own symbols first, then the network
        self.chart = ChartView(engine, themes, symbols, settings, search)
        self.screener = ScreenerView(engine, themes, symbols)
        self.news = NewsView(themes, lambda: list(dict.fromkeys(symbols + self.portfolio_symbols)))
        self.portfolio = PortfolioView(engine, themes)
        self.lab = BacktestView(engine, themes, symbols, search)
        pages = [("Chart", "📈", self.chart), ("Screener", "🔎", self.screener), ("Backtest", "🧪", self.lab),
                 ("Portfolio", "💼", self.portfolio), ("News", "📰", self.news)]

        self.stack = QStackedWidget()
        sidebar = QWidget(objectName="Sidebar")
        sidebar.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        sidebar.setFixedWidth(200)
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(12, 18, 12, 14)
        brand = QLabel("◆ tradebot")
        brand.setObjectName("Title")
        side.addWidget(brand)
        version = QLabel(f"v{__version__}")
        version.setObjectName("Muted")
        side.addWidget(version)
        side.addSpacing(16)
        group = QButtonGroup(self)
        for i, (name, icon, page) in enumerate(pages):
            self.stack.addWidget(page)
            btn = QPushButton(f"{icon}  {name}", objectName="Nav", checkable=True)
            btn.clicked.connect(lambda _, n=i: self.stack.setCurrentIndex(n))
            group.addButton(btn)
            side.addWidget(btn)
            if i == 0:
                btn.setChecked(True)
        side.addStretch()
        side.addWidget(QLabel("Theme", objectName="Muted"))
        self.theme_box = QComboBox()
        for key, theme in themes.themes.items():
            self.theme_box.addItem(theme.name, key)
        self.theme_box.currentIndexChanged.connect(lambda _: self._theme_chosen(self.theme_box.currentData()))
        side.addWidget(self.theme_box)

        # cross-page navigation
        def chart_symbol(sym):
            self.stack.setCurrentWidget(self.chart)
            group.buttons()[0].setChecked(True)
            self.chart.open_symbol(sym)

        self.screener.open_symbol.connect(chart_symbol)
        self.portfolio.open_symbol.connect(chart_symbol)
        self.portfolio.loaded.connect(lambda syms: setattr(self, "portfolio_symbols", syms))

        def lab_to_chart(sym, algo):
            chart_symbol(sym)
            self.chart.set_algorithm(algo)

        self.lab.show_on_chart.connect(lab_to_chart)

        root = GlassBackdrop(themes)
        lay = QHBoxLayout(root)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(sidebar)
        lay.addWidget(self.stack, 1)
        self.setCentralWidget(root)
        self.setStatusBar(QStatusBar())
        for page in pages:
            page[2].status.connect(self.statusBar().showMessage)

        idx = self.theme_box.findData(theme_key if theme_key in themes.themes else "glass")  # e.g. a deleted user theme
        self.theme_box.blockSignals(True)  # starting with the configured theme isn't a choice to save
        self.theme_box.setCurrentIndex(idx)
        self.theme_box.blockSignals(False)
        themes.apply(self.theme_box.currentData())
        self.chart.load()

    def _theme_chosen(self, key: str):
        """Apply a theme picked in the sidebar and remember it as ``theme`` in the config file (#49)."""
        self.themes.apply(key)
        if not save_config_value("theme", key):
            self.statusBar().showMessage(f"theme not saved: couldn't write {config_path()}", 8000)


def main(argv: list[str] | None = None) -> int:
    quiet_yfinance()
    load_plugins()
    app = QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName("tradebot")
    config = load_config()
    # touch_mode: "auto" (default, on when a touchscreen is present), "on" or "off"
    mode = str(config.get("touch_mode", "auto")).lower()
    themes = ThemeManager(touch=mode == "on" or (mode == "auto" and has_touchscreen()))
    window = MainWindow(themes, config.get("theme", "glass"), Settings())
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
