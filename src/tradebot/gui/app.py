# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Desktop application entry point and main window."""

from __future__ import annotations

import logging
import sys
from functools import partial

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
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
from tradebot.data.cache import DiskCache
from tradebot.data.providers import quiet_yfinance
from tradebot.gui.chart import ChartView
from tradebot.gui.common import run_async
from tradebot.gui.settings import SettingsDialog
from tradebot.legal import NOT_AFFILIATED, NO_WARRANTY
from tradebot.gui.theme import GlassBackdrop, ThemeManager
from tradebot.gui.touch import has_touchscreen
from tradebot.gui.views import BacktestView, NewsView, PortfolioView, ScreenerView
from tradebot.symbols import disk_symbol_search, symbol_name, yahoo_search

log = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    def __init__(self, themes: ThemeManager, theme_key: str = "glass", settings: Settings | None = None):
        super().__init__()
        self.setWindowTitle("tradebot")
        self.resize(1440, 900)
        self.themes = themes
        engine = self.engine = MarketDataEngine()
        symbols = watchlist()
        self.portfolio_symbols: list[str] = []

        # own symbols first, then crypto pairs and Yahoo, from the disk cache when they can (#63)
        search = disk_symbol_search(lambda: symbols + self.portfolio_symbols,
                                    engine.disk.symbols_dir if engine.disk else None, yahoo=yahoo_search)
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
        about = QPushButton("ⓘ  About", objectName="Nav")  # disclaimers and licence (#58)
        about.clicked.connect(self.show_about)
        side.addWidget(about)
        self.settings_btn = QPushButton("⚙  Settings", objectName="Nav", toolTip="Theme, third-party tools")  # #59
        self.settings_btn.clicked.connect(self.show_settings)
        side.addWidget(self.settings_btn)

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

        self.theme_key = theme_key if theme_key in themes.themes else "glass"  # e.g. a deleted user theme
        themes.apply(self.theme_key)  # starting with the configured theme isn't a choice to save
        self.chart.load()
        run_async(partial(warm_up, engine, symbols), lambda _: None, lambda err: None)

    def show_about(self):
        QMessageBox.about(self, "About tradebot",
                          f"<h3>tradebot v{__version__}</h3><p>{NOT_AFFILIATED}</p><p>{NO_WARRANTY}</p>"
                          "<p>Licensed under the GNU GPL v3.0 or later. Licences of the bundled libraries: "
                          "THIRD_PARTY_NOTICES.txt next to the program. Source code: "
                          "<a href='https://github.com/daMarxrider/trading_bot_3'>github.com/daMarxrider/trading_bot_3</a></p>")

    def show_settings(self):
        dialog = SettingsDialog(self.themes, self.theme_key, self)
        dialog.theme_chosen.connect(self._theme_chosen)
        dialog.message.connect(lambda text: self.statusBar().showMessage(text, 8000))
        dialog.cache_changed.connect(self._cache_changed)
        self._settings_dialog = dialog  # for tests
        dialog.exec()

    def _cache_changed(self):
        """Cache switched on/off, resized or cleared in the settings: use it from now on."""
        self.engine.disk = DiskCache.from_config()
        self.engine.clear_memory()

    def _theme_chosen(self, key: str):
        """Apply a theme picked in the settings and remember it as ``theme`` in the config file (#49)."""
        self.theme_key = key
        self.themes.apply(key)
        if not save_config_value("theme", key):
            self.statusBar().showMessage(f"theme not saved: couldn't write {config_path()}", 8000)


def warm_up(engine: MarketDataEngine, symbols: list[str]) -> None:
    """In the background after startup: the daily cache cleanup, then the watchlist's names and daily
    candles, so opening them later doesn't wait for the network (#63)."""
    if engine.disk is None:
        return
    engine.disk.maybe_cleanup()
    for symbol in symbols:
        try:
            symbol_name(symbol)
            engine.get_ohlc(symbol, "1d", bars=500)
        except Exception as exc:  # offline or unknown symbol: nothing to warm
            log.debug("warm-up of %s failed: %s", symbol, exc)


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
