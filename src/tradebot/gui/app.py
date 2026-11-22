# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Desktop application entry point and main window."""

from __future__ import annotations

import logging
import sys
from functools import partial

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QFrame,
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
from tradebot.algorithms.combo import load_combos
from tradebot.core.registry import load_plugins
from tradebot.data import MarketDataEngine
from tradebot.data.cache import DiskCache
from tradebot.data.providers import quiet_yfinance
from tradebot.gui.chart import ChartView
from tradebot.gui.common import run_async
from tradebot.gui.evolve import EvolveView
from tradebot.gui.icons import pixmap
from tradebot.gui.pairs import PairView
from tradebot.gui.settings import SettingsDialog
from tradebot.live.risk import kill_active, set_kill
from tradebot.legal import NOT_AFFILIATED, NO_WARRANTY
from tradebot.gui.theme import GlassBackdrop, ThemeManager
from tradebot.gui.touch import has_touchscreen
from tradebot.gui.scanner import ScannerView
from tradebot.gui.views import BacktestView, NewsView, PortfolioView, ScreenerView
from tradebot.symbols import disk_symbol_search, symbol_name, yahoo_search

log = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    def __init__(self, themes: ThemeManager, theme_key: str = "graphite", settings: Settings | None = None,
                 opengl: bool = False):
        super().__init__()
        self.setWindowTitle("tradebot")
        self.resize(1440, 900)
        self.themes = themes
        engine = self.engine = MarketDataEngine()
        symbols = watchlist()
        self.portfolio_symbols: list[str] = []

        # own symbols first, then crypto pairs and Yahoo, from the disk cache when they can (#63)
        search = disk_symbol_search(lambda: self.chart.watchlists.all_symbols() + self.portfolio_symbols,
                                    engine.disk.symbols_dir if engine.disk else None, yahoo=yahoo_search)
        self.chart = ChartView(engine, themes, symbols, settings, search, opengl=opengl)  # opengl: #71
        self.screener = ScreenerView(engine, themes, symbols)
        self.scanner = ScannerView(engine, themes, lambda: self.chart.watchlists.lists, settings)
        self.chart.watchlists.changed.connect(self.scanner.watchlists_changed)
        self.scanner.indicator_reloaded.connect(self.chart.reload_indicators)
        self.news = NewsView(themes, lambda: list(dict.fromkeys(self.chart.watchlists.all_symbols()
                                                                + self.portfolio_symbols)))
        self.pairs = PairView(engine, themes, symbols)
        self.portfolio = PortfolioView(engine, themes)
        self.lab = BacktestView(engine, themes, symbols, search, settings, lambda: self.chart.watchlists.lists)
        self.evolve = EvolveView(engine, themes, symbols, search, settings, lambda: self.chart.watchlists.lists)
        groups = [("Markets", [("Chart", "chart", self.chart), ("Screener", "screener", self.screener),
                               ("Scanner", "scanner", self.scanner), ("Pairs", "pairs", self.pairs), ("News", "news", self.news)]),
                  ("Strategies", [("Backtest", "flask", self.lab), ("Evolve", "dna", self.evolve)]),
                  ("Account", [("Portfolio", "wallet", self.portfolio)])]
        pages = [item for _, items in groups for item in items]

        self.stack = QStackedWidget()
        sidebar = QWidget(objectName="Sidebar")
        sidebar.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        sidebar.setFixedWidth(192)
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(10, 16, 10, 12)
        side.setSpacing(2)
        head = QHBoxLayout()
        head.setContentsMargins(8, 0, 8, 6)
        brand = QLabel("tradebot")
        brand.setObjectName("Brand")
        head.addWidget(brand)
        version = QLabel(f"v{__version__}")
        version.setObjectName("Muted")
        head.addWidget(version)
        head.addStretch()
        side.addLayout(head)
        group = QButtonGroup(self)
        self._nav_icons: list[tuple[QPushButton, str]] = []
        index = 0
        for title, items in groups:
            heading = QLabel(title)
            heading.setObjectName("NavGroup")
            side.addWidget(heading)
            for name, glyph, page in items:
                n = index
                index += 1
                self.stack.addWidget(page)
                btn = QPushButton(name, objectName="Nav", checkable=True)
                btn.setIconSize(QSize(16, 16))
                btn.clicked.connect(lambda _, n=n: self.stack.setCurrentIndex(n))
                group.addButton(btn)
                side.addWidget(btn)
                self._nav_icons.append((btn, glyph))
                if n == 0:
                    btn.setChecked(True)
        side.addStretch()
        about = QPushButton("About", objectName="Nav")  # disclaimers and licence (#58)
        about.clicked.connect(self.show_about)
        side.addWidget(about)
        self._nav_icons.append((about, "info"))
        self.settings_btn = QPushButton("Settings", objectName="Nav", toolTip="Theme, third-party tools")  # #59
        self.settings_btn.clicked.connect(self.show_settings)
        side.addWidget(self.settings_btn)
        self._nav_icons.append((self.settings_btn, "settings"))
        for btn in (about, self.settings_btn):
            btn.setIconSize(QSize(16, 16))
        themes.changed.connect(self._refresh_icons)

        # cross-page navigation
        def chart_symbol(sym):
            self.stack.setCurrentWidget(self.chart)
            group.buttons()[0].setChecked(True)
            self.chart.open_symbol(sym)

        self.screener.open_symbol.connect(chart_symbol)

        def scanned_symbol(sym, timeframe, indicator, params):
            self.stack.setCurrentWidget(self.chart)
            group.buttons()[0].setChecked(True)
            self.chart.show_scanned(sym, timeframe, indicator, params)

        self.scanner.open_chart.connect(scanned_symbol)
        self.portfolio.open_symbol.connect(chart_symbol)
        self.portfolio.loaded.connect(lambda syms: setattr(self, "portfolio_symbols", syms))

        def lab_to_chart(sym, algo):
            chart_symbol(sym)
            self.chart.set_algorithm(algo)

        self.lab.show_on_chart.connect(lab_to_chart)
        self.evolve.show_on_chart.connect(lab_to_chart)

        def strategy_saved(name):  # an evolved strategy saved as a plugin shows up in every list at once (#83)
            for box in (self.lab.algo, self.chart.algo_box):
                if box.findText(name) < 0:
                    box.addItem(name)

        self.evolve.saved.connect(strategy_saved)

        root = GlassBackdrop(themes)
        lay = QHBoxLayout(root)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(sidebar)
        right = QVBoxLayout()
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(0)
        self.kill_banner = self._build_kill_banner()
        right.addWidget(self.kill_banner)
        right.addWidget(self.stack, 1)
        lay.addLayout(right, 1)
        self.setCentralWidget(root)
        self.setStatusBar(QStatusBar())
        # the desktop app never places orders, so the chip says so; the kill switch lives next to it (#154, #191)
        self.mode_chip = QLabel("RESEARCH · NO ORDERS", objectName="ModeChip",
                                toolTip="The desktop app only analyses. Live bots run with `tradebot bot run`.")
        self.kill_btn = QPushButton("Kill switch", objectName="Kill", checkable=True,
                                    toolTip="Stops every running live bot at its next check (same as `tradebot bot kill`)")
        self.kill_btn.setChecked(kill_active())
        self.kill_btn.toggled.connect(self.toggle_kill)  # #154
        self.statusBar().addPermanentWidget(self.mode_chip)
        self.statusBar().addPermanentWidget(self.kill_btn)
        self._kill_ui(self.kill_btn.isChecked())
        for page in pages:
            page[2].status.connect(self.statusBar().showMessage)

        self.theme_key = theme_key if theme_key in themes.themes else "graphite"  # e.g. a deleted user theme
        themes.apply(self.theme_key)  # starting with the configured theme isn't a choice to save
        self.chart.load()
        if self.chart.opengl_note:  # GPU rendering is on but OpenGL isn't available (#71)
            self.statusBar().showMessage(self.chart.opengl_note, 15000)
        run_async(partial(warm_up, engine, symbols), lambda _: None, lambda err: None)

    def show_about(self):
        QMessageBox.about(self, "About tradebot",
                          f"<h3>tradebot v{__version__}</h3><p>{NOT_AFFILIATED}</p><p>{NO_WARRANTY}</p>"
                          "<p>Licensed under the GNU GPL v3.0 or later. Licences of the bundled libraries: "
                          "THIRD_PARTY_NOTICES.txt next to the program. Source code: "
                          "<a href='https://github.com/daMarxrider/trading_bot_3'>github.com/daMarxrider/trading_bot_3</a></p>")

    def _build_kill_banner(self) -> QFrame:
        banner = QFrame(objectName="KillBanner")
        row = QHBoxLayout(banner)
        row.setContentsMargins(16, 6, 16, 6)
        row.addWidget(QLabel("Kill switch is on. Every live bot stops at its next check and no new orders go out."))
        row.addStretch()
        off = QPushButton("Turn off")
        off.clicked.connect(lambda: self.kill_btn.setChecked(False))
        row.addWidget(off)
        banner.hide()
        return banner

    def _kill_ui(self, on: bool):
        self.kill_banner.setVisible(on)
        self.kill_btn.setText("Kill switch ON" if on else "Kill switch")
        self.mode_chip.setProperty("state", "kill" if on else "")
        self.mode_chip.setText("KILL SWITCH ON" if on else "RESEARCH · NO ORDERS")
        self.mode_chip.style().unpolish(self.mode_chip)
        self.mode_chip.style().polish(self.mode_chip)

    def _refresh_icons(self, theme=None):
        """Recolour the nav icons for the active theme: muted when idle, accent when the page is open."""
        t = theme or self.themes.current
        for btn, glyph in self._nav_icons:
            ic = QIcon()
            ic.addPixmap(pixmap(glyph, t.muted), QIcon.Mode.Normal, QIcon.State.Off)
            ic.addPixmap(pixmap(glyph, t.accent), QIcon.Mode.Normal, QIcon.State.On)
            btn.setIcon(ic)

    def toggle_kill(self, on: bool):
        set_kill(on, "gui")
        self._kill_ui(on)
        self.statusBar().showMessage("kill switch ON: live bots stop trading" if on else "kill switch off", 8000)

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
    load_combos()
    app = QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName("tradebot")
    config = load_config()
    # touch_mode: "auto" (default, on when a touchscreen is present), "on" or "off"
    mode = str(config.get("touch_mode", "auto")).lower()
    themes = ThemeManager(touch=mode == "on" or (mode == "auto" and has_touchscreen()))
    window = MainWindow(themes, config.get("theme", "graphite"), Settings(), opengl=bool(config.get("gpu_rendering", False)))
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
