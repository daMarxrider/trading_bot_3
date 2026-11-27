# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Desktop application entry point and main window."""

from __future__ import annotations

import logging
import os
import sys
from functools import partial

from PyQt6.QtCore import QSize, Qt, QTimer
from PyQt6.QtGui import QGuiApplication, QIcon
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
from tradebot.gui.about import AboutDialog
from tradebot.gui.alerts import AlertsDialog, AlertWatcher
from tradebot.gui.bots import BotsView
from tradebot.gui.chart import ChartView
from tradebot.gui.common import run_async
from tradebot.gui.evolve import EvolveView
from tradebot.gui.icons import pixmap
from tradebot.gui.journal import JournalView
from tradebot.gui.pairs import PairView
from tradebot.gui.portfolio_backtest import PortfolioBacktestView
from tradebot.gui.scripts_page import ScriptsView
from tradebot.gui.settings import SettingsDialog
from tradebot.live.risk import kill_active, set_kill
from tradebot.gui import blur
from tradebot.gui.theme import GlassBackdrop, ThemeManager
from tradebot.gui.touch import has_touchscreen
from tradebot.gui.scanner import ScannerView
from tradebot.gui.views import BacktestView, NewsView, PortfolioView, ScreenerView
from tradebot.gui.symbol_dialog import SymbolContext
from tradebot.symbols import disk_symbol_search, symbol_name, yahoo_search

log = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    def __init__(self, themes: ThemeManager, theme_key: str = "graphite", settings: Settings | None = None,
                 opengl: bool = False, glass_blur: bool = False):
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
        # what the symbol dialog shows besides search results: prices, recent symbols, the watchlists (#220)
        self.symbol_context = SymbolContext(engine, settings, themes, lambda: self.chart.watchlists.lists)
        QTimer.singleShot(1500, lambda: self.symbol_context.prewarm(search, self))  # ready before the first click
        self.screener = ScreenerView(engine, themes, symbols)
        self.scanner = ScannerView(engine, themes, lambda: self.chart.watchlists.lists, settings)
        self.chart.watchlists.changed.connect(self.scanner.watchlists_changed)
        self.scanner.indicator_reloaded.connect(self.chart.reload_indicators)
        self.news = NewsView(themes, lambda: list(dict.fromkeys(self.chart.watchlists.all_symbols()
                                                                + self.portfolio_symbols)))
        self.pairs = PairView(engine, themes, symbols, search)
        self.basket = PortfolioBacktestView(engine, themes, symbols)  # #148
        self.portfolio = PortfolioView(engine, themes)
        self.lab = BacktestView(engine, themes, symbols, search, settings, lambda: self.chart.watchlists.lists)
        self.evolve = EvolveView(engine, themes, symbols, search, settings, lambda: self.chart.watchlists.lists)
        self.journal = JournalView(engine, themes)
        from tradebot.journal import Journal

        self.bots = BotsView(themes, settings, symbol_search=search, journal_factory=Journal)  # #157
        self.bots.running_changed.connect(self._bots_running)
        self.scripts_page = ScriptsView(themes)
        self.scripts_page.reloaded.connect(self.scanner._script_reloaded)  # the scanner and the charts follow a reload
        groups = [("Markets", [("Chart", "chart", self.chart), ("Screener", "screener", self.screener),
                               ("Scanner", "scanner", self.scanner), ("Pairs", "pairs", self.pairs), ("News", "news", self.news)]),
                  ("Strategies", [("Backtest", "flask", self.lab), ("Evolve", "dna", self.evolve), ("Basket", "basket", self.basket),
                                  ("Scripts", "code", self.scripts_page)]),
                  ("Account", [("Bots", "bot", self.bots), ("Portfolio", "wallet", self.portfolio),
                               ("Journal", "journal", self.journal)])]
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
        self.alerts_btn = QPushButton("Alerts", objectName="Nav", toolTip="Price/indicator alerts and scheduled scans")
        self.alerts_btn.clicked.connect(self.show_alerts)  # #156
        side.addWidget(self.alerts_btn)
        self._nav_icons.append((self.alerts_btn, "bell"))
        self.alerts_btn.setIconSize(QSize(16, 16))
        self.alert_watcher = AlertWatcher(self)
        self.alert_watcher.fired.connect(lambda line: self.statusBar().showMessage(line, 15000))
        about = QPushButton("About", objectName="Nav")  # disclaimers and licence (#58)
        about.clicked.connect(self.show_about)
        side.addWidget(about)
        self._nav_icons.append((about, "info"))
        self.settings_btn = QPushButton("Settings", objectName="Nav", checkable=True,
                                        toolTip="Theme, third-party tools")  # #59
        self.settings_btn.clicked.connect(self.show_settings)
        group.addButton(self.settings_btn)  # one page at a time: opening Settings unchecks the others
        side.addWidget(self.settings_btn)
        self._settings_dialog = None  # built on first use
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
        self.news.open_symbol.connect(chart_symbol)

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

        root = self.backdrop = GlassBackdrop(themes)
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
        # the desktop app never places real orders, so the chip says so; the kill switch lives next to it (#154, #191)
        self.mode_chip = QLabel("RESEARCH · NO ORDERS", objectName="ModeChip")
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
        # real compositor blur behind the Glass theme (#7): opt-in, only where a backend exists
        self.blur_backend = blur.current_backend() if glass_blur else "none"
        blur.prepare(self, self.blur_backend)
        themes.changed.connect(lambda _: self._sync_blur())
        themes.apply(self.theme_key)  # starting with the configured theme isn't a choice to save
        self.chart.load()
        if self.chart.opengl_note:  # GPU rendering is on but OpenGL isn't available (#71)
            self.statusBar().showMessage(self.chart.opengl_note, 15000)
        run_async(partial(warm_up, engine, symbols), lambda _: None, lambda err: None)

    def _bots_running(self, _modes=None):
        self._kill_ui(self.kill_btn.isChecked())

    def closeEvent(self, event):
        self.bots.shutdown()  # the bots' state is saved, they resume when started again
        super().closeEvent(event)

    def show_about(self):
        AboutDialog(self).exec()

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
        modes = self.bots.manager.running() and sorted({self.bots.bots.get(n).mode for n in self.bots.manager.running()})
        if on:
            state, text = "kill", "KILL SWITCH ON"
        elif modes:  # dry-run and paper bots run in the app (#157): still no real order
            state, text = "bots", ("PAPER BOTS" if "paper" in modes else "DRY-RUN BOTS") + " · NO REAL ORDERS"
        else:
            state, text = "", "RESEARCH · NO ORDERS"
        self.mode_chip.setProperty("state", state)
        self.mode_chip.setText(text)
        self.mode_chip.setToolTip("The app runs only dry-run and paper bots (Bots page); live bots run with "
                                  "`tradebot bots run --live` or `tradebot bot run --live`.")
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

    def show_alerts(self):
        dialog = AlertsDialog(self)
        self._alerts_dialog = dialog  # for tests
        dialog.exec()

    def show_settings(self):
        """Settings is a page of the window (#191), built the first time it is opened."""
        if self._settings_dialog is None:
            dialog = self._settings_dialog = SettingsDialog(self.themes, self.theme_key, None, embedded=True)
            dialog.theme_chosen.connect(self._theme_chosen)
            dialog.message.connect(lambda text: self.statusBar().showMessage(text, 8000))
            dialog.cache_changed.connect(self._cache_changed)
            self.stack.addWidget(dialog)
        self.stack.setCurrentWidget(self._settings_dialog)
        self.settings_btn.setChecked(True)

    def _cache_changed(self):
        """Cache switched on/off, resized or cleared in the settings: use it from now on."""
        self.engine.disk = DiskCache.from_config()
        self.engine.clear_memory()

    def showEvent(self, event):
        super().showEvent(event)
        self._sync_blur()  # the native window only exists once shown

    def _sync_blur(self):
        """Blur behind the window while the Glass theme is on; other themes stay opaque. Falls back to the
        painted gradient when the compositor can't (``blur_active`` stays False)."""
        if self.blur_backend == "none" or not self.isVisible():
            return
        self.backdrop.blur_active = blur.set_blur(self, self.blur_backend, self.themes.current.glass)
        self.backdrop.update()

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


def app_icon() -> QIcon:
    from tradebot.desktop import ASSETS, ICON_SIZES

    icon = QIcon()
    for size in ICON_SIZES:
        icon.addFile(str(ASSETS / f"tradebot-{size}.png"), QSize(size, size))
    return icon


def main(argv: list[str] | None = None) -> int:
    quiet_yfinance()
    load_plugins()
    load_combos()
    app = QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName("tradebot")
    # matches the launcher entry (#213): Wayland shows the entry's icon and name via this id; the dev entry
    # (`tradebot desktop install --dev`) sets TRADEBOT_DESKTOP_ID=tradebot-dev
    QGuiApplication.setDesktopFileName(os.environ.get("TRADEBOT_DESKTOP_ID") or "tradebot")
    app.setWindowIcon(app_icon())
    config = load_config()
    # touch_mode: "auto" (default, on when a touchscreen is present), "on" or "off"
    mode = str(config.get("touch_mode", "auto")).lower()
    themes = ThemeManager(touch=mode == "on" or (mode == "auto" and has_touchscreen()))
    window = MainWindow(themes, config.get("theme", "graphite"), Settings(), opengl=bool(config.get("gpu_rendering", False)),
                        glass_blur=blur.wants_blur(config))
    window.show()
    window.alert_watcher.start()  # price/indicator alerts and scheduled scans while the app is open (#156)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
