# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""The settings dialog behind the sidebar's gear button (#59).

Categories on the left, one page each: Appearance (theme), Integrations (third-party tools, broker
API keys) and Data & cache (the disk cache, #63). A new category is one more :meth:`SettingsDialog._add_page` call.
"""

from __future__ import annotations

import os
from html import escape

from PyQt6.QtCore import Qt, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from tradebot import tools
from tradebot.brokers import BROKERS
from tradebot.config import config_path, load_config, os_cache_dir, save_config_value
from tradebot.data.cache import DEFAULT_MAX_MB, DiskCache, human_size
from tradebot.gui.theme import ThemeManager

#: How a credential key is labelled; others read "Some key" from "some_key"
KEY_LABELS = {"api_key": "API key", "secret": "Secret", "password": "Passphrase", "holdings_csv": "Holdings CSV",
              "phone_no": "Phone number", "pin": "PIN"}
SECRET_KEYS = {"secret", "password", "pin"}  #: shown as dots


class SettingsDialog(QDialog):
    """Changes apply and are saved to config.toml right away; Close just closes."""

    theme_chosen = pyqtSignal(str)
    message = pyqtSignal(str)  #: e.g. a setting that couldn't be saved, for the status bar
    cache_changed = pyqtSignal()  #: the cache was switched on/off, resized or cleared

    def __init__(self, themes: ThemeManager, theme_key: str, parent=None):
        super().__init__(parent)
        self.themes = themes
        self.setWindowTitle("Settings")
        self.setMinimumSize(780, 520)
        self.nav = QVBoxLayout()
        self.nav.setContentsMargins(0, 0, 0, 0)
        self.nav_group = QButtonGroup(self)
        self.pages = QStackedWidget()
        self.paths: dict[str, QLineEdit] = {}
        self.status: dict[str, QLabel] = {}
        self.credentials: dict[tuple[str, str], QLineEdit] = {}  # (broker, key) -> field
        self.page_names: list[str] = []  # the categories, in the order of the left bar

        self._add_page("Appearance", "🎨", self._appearance(theme_key))
        self._add_page("Integrations", "🔌", self._integrations())
        self._add_page("Data & cache", "💾", self._cache_page())
        self.nav_group.buttons()[0].setChecked(True)
        self.nav.addStretch()

        nav = QWidget()
        nav.setLayout(self.nav)
        nav.setFixedWidth(180)
        body = QHBoxLayout()
        body.addWidget(nav)
        body.addWidget(self.pages, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addLayout(body, 1)
        lay.addWidget(buttons)
        themes.changed.connect(self._refresh)  # the links take the new theme's accent colour

    def _add_page(self, name: str, icon: str, content: QWidget):
        """A category: a button in the left bar and a scrollable page."""
        scroll = QScrollArea(widgetResizable=True, frameShape=QFrame.Shape.NoFrame)
        scroll.setStyleSheet("QScrollArea { background: transparent; } "
                             "QScrollArea > QWidget > QWidget { background: transparent; }")
        scroll.setWidget(content)
        self.pages.addWidget(scroll)
        btn = QPushButton(f"{icon}  {name.replace('&', '&&')}", objectName="Nav", checkable=True)  # "&": not a shortcut
        btn.clicked.connect(lambda _, w=scroll: self.pages.setCurrentWidget(w))
        self.nav_group.addButton(btn)
        self.nav.addWidget(btn)
        self.page_names.append(name)

    def show_page(self, name: str):
        self.nav_group.buttons()[self.page_names.index(name)].click()

    # -- Appearance ------------------------------------------------------------------------------
    def _appearance(self, theme_key: str) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 4, 0)
        box = QGroupBox("Theme")
        form = QFormLayout(box)
        self.theme_box = QComboBox()
        for key, theme in self.themes.themes.items():
            self.theme_box.addItem(theme.name, key)
        self.theme_box.setCurrentIndex(max(self.theme_box.findData(theme_key), 0))
        self.theme_box.currentIndexChanged.connect(lambda _: self.theme_chosen.emit(self.theme_box.currentData()))
        form.addRow("Theme", self.theme_box)
        lay.addWidget(box)
        lay.addStretch()
        return page

    # -- Integrations ----------------------------------------------------------------------------
    def _integrations(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 4, 0)
        lay.addWidget(self._tools_box())
        lay.addWidget(self._brokers_box())
        lay.addStretch()
        return page

    def _tools_box(self) -> QGroupBox:
        box = QGroupBox("Third-party tools")
        lay = QVBoxLayout(box)
        lay.addWidget(QLabel("tradebot doesn't include these tools. Install them yourself; leave a path empty to "
                             "find the tool on PATH.", objectName="Muted", wordWrap=True))
        form = QFormLayout()
        for t in tools.TOOLS:
            edit = QLineEdit(tools.configured_path(t.key), placeholderText=f"{t.command} (on PATH)")
            edit.setToolTip(f"Used for: {t.used_for}")
            edit.editingFinished.connect(lambda k=t.key: self._save_path(k))
            status = QLabel(openExternalLinks=True, wordWrap=True)
            self.paths[t.key], self.status[t.key] = edit, status
            form.addRow(t.name, _with_browse(edit, lambda k=t.key: self._browse(k)))
            form.addRow("", status)
            self._show_status(t.key)
        lay.addLayout(form)
        return box

    def _brokers_box(self) -> QGroupBox:
        box = QGroupBox("Broker API keys")
        lay = QVBoxLayout(box)
        lay.addWidget(QLabel("Saved in plain text in config.toml. To keep a key out of the file, set the "
                             "environment variable TRADEBOT_<BROKER>_<KEY> instead (e.g. TRADEBOT_BINANCE_SECRET). "
                             "Use API keys without withdrawal rights.", objectName="Muted", wordWrap=True))
        saved = load_config().get("brokers", {})
        form = QFormLayout()  # one form for all brokers, so their fields line up
        for broker in BROKERS:
            if not broker.credential_keys:
                continue
            form.addRow(QLabel(f"<b>{escape(broker.title or broker.name.capitalize())}</b>"))
            for key in broker.credential_keys:
                edit = QLineEdit(str(saved.get(broker.name, {}).get(key, "")))
                if key in SECRET_KEYS:
                    edit.setEchoMode(QLineEdit.EchoMode.Password)
                env = f"TRADEBOT_{broker.name.upper()}_{key.upper()}"
                if os.environ.get(env):  # the environment wins over the file (broker_credentials)
                    edit.clear()
                    edit.setEnabled(False)
                    edit.setPlaceholderText(f"set by {env}")
                edit.editingFinished.connect(lambda b=broker.name, k=key: self._save_credential(b, k))
                self.credentials[broker.name, key] = edit
                field = edit
                if key.endswith("_csv"):
                    field = _with_browse(edit, lambda b=broker.name, k=key: self._browse_file(b, k))
                form.addRow(KEY_LABELS.get(key, key.replace("_", " ").capitalize()), field)
        lay.addLayout(form)
        return box

    # -- Data & cache ----------------------------------------------------------------------------
    def _cache_page(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 4, 0)
        conf = load_config().get("cache", {})

        box = QGroupBox("Disk cache")
        form = QFormLayout(box)
        form.addRow(QLabel("Candles and symbol lists are kept on disk, so charts and symbol search open without "
                           "waiting for the network. Data not used for 60 days (14 days for 1m–30m candles) is "
                           "deleted automatically, and the least recently used data goes first when the cache "
                           "is full.", objectName="Muted", wordWrap=True))
        self.cache_enabled = QCheckBox("Keep market data on disk")
        self.cache_enabled.setChecked(bool(conf.get("enabled", True)))
        self.cache_enabled.toggled.connect(self._save_cache_enabled)
        form.addRow(self.cache_enabled)
        self.cache_max = QSpinBox(minimum=50, maximum=50_000, singleStep=100, suffix=" MB")
        self.cache_max.setValue(int(conf.get("max_mb", DEFAULT_MAX_MB)))
        self.cache_max.editingFinished.connect(self._save_cache_max)
        form.addRow("Maximum size", self.cache_max)
        lay.addWidget(box)

        box = QGroupBox("Cached data")
        form = QFormLayout(box)
        self.cache_usage = {k: QLabel() for k in ("ohlc", "symbols", "total")}
        form.addRow("Chart data", self.cache_usage["ohlc"])
        form.addRow("Symbol lists", self.cache_usage["symbols"])
        form.addRow("Total", self.cache_usage["total"])
        folder = QLabel(escape(str(os_cache_dir())), wordWrap=True)
        folder.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        open_btn = QPushButton("Open folder")
        open_btn.clicked.connect(lambda _: QDesktopServices.openUrl(QUrl.fromLocalFile(str(os_cache_dir()))))
        row = QHBoxLayout()
        row.addWidget(folder, 1)
        row.addWidget(open_btn)
        form.addRow("Folder", row)
        clear = QHBoxLayout()
        for label, what in (("Clear chart data", "ohlc"), ("Clear symbol lists", "symbols"), ("Clear all", "all")):
            btn = QPushButton(label)
            btn.clicked.connect(lambda _, w=what: self.clear_cache(w))
            clear.addWidget(btn)
        clear.addStretch()
        form.addRow(clear)
        lay.addWidget(box)
        lay.addStretch()
        self._show_cache_usage()
        return page

    def _show_cache_usage(self):
        info = DiskCache().info()  # what's on disk, also while the cache is switched off
        series = f"{info.series} series" if info.series != 1 else "1 series"
        self.cache_usage["ohlc"].setText(f"{human_size(info.ohlc_bytes)} in {series}")
        self.cache_usage["symbols"].setText(human_size(info.symbol_bytes))
        self.cache_usage["total"].setText(f"{human_size(info.total_bytes)} of at most {self.cache_max.value():,} MB")

    def clear_cache(self, what: str, confirm: bool = True):
        """``what``: "ohlc", "symbols" or "all"."""
        names = {"ohlc": "the cached chart data", "symbols": "the cached symbol lists", "all": "all cached data"}
        if confirm:
            answer = QMessageBox.question(self, "Clear cache", f"Delete {names[what]}? It is downloaded again when "
                                          "needed.", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                          QMessageBox.StandardButton.No)
            if answer != QMessageBox.StandardButton.Yes:
                return
        DiskCache().clear(ohlc=what in ("ohlc", "all"), symbols=what in ("symbols", "all"))
        self._show_cache_usage()
        self.cache_changed.emit()
        self.message.emit(f"deleted {names[what]}")

    def _save_cache_enabled(self, on: bool):
        if not save_config_value("enabled", on, table="cache"):
            self.message.emit(f"cache setting not saved: couldn't write {config_path()}")
        self.cache_changed.emit()

    def _save_cache_max(self):
        value = self.cache_max.value()
        if value == int(load_config().get("cache", {}).get("max_mb", DEFAULT_MAX_MB)):
            return
        if not save_config_value("max_mb", value, table="cache"):
            self.message.emit(f"cache size not saved: couldn't write {config_path()}")
        self._show_cache_usage()
        self.cache_changed.emit()

    def _save_credential(self, broker: str, key: str):
        value = self.credentials[broker, key].text().strip()
        if value == str(load_config().get("brokers", {}).get(broker, {}).get(key, "")):
            return
        if not save_config_value(key, value or None, table=f"brokers.{broker}"):
            self.message.emit(f"{broker} {key} not saved: couldn't write {config_path()}")

    def _browse_file(self, broker: str, key: str):
        edit = self.credentials[broker, key]
        path, _ = QFileDialog.getOpenFileName(self, KEY_LABELS.get(key, key), edit.text(),
                                              "CSV files (*.csv);;All files (*)")
        if path:
            edit.setText(path)
            self._save_credential(broker, key)

    def _refresh(self, _theme=None):
        for key in self.paths:
            self._show_status(key)

    def _browse(self, key: str):
        path, _ = QFileDialog.getOpenFileName(self, f"Find {tools.tool(key).name}", self.paths[key].text())
        if path:
            self.paths[key].setText(path)
            self._save_path(key)

    def _save_path(self, key: str):
        path = self.paths[key].text().strip()
        if path != tools.configured_path(key) and not tools.set_path(key, path):
            self.message.emit(f"{tools.tool(key).name} path not saved: couldn't write {config_path()}")
        self._show_status(key)

    def _show_status(self, key: str):
        t, found = tools.tool(key), tools.locate(key, self.paths[key].text())
        link = f"<a style='color:{self.themes.current.accent}' href='{t.url}'>download</a>"  # Qt's default blue: unreadable on dark
        if found:
            text = f"✓ found: {escape(found)}"
        elif self.paths[key].text().strip():
            text = f"✗ not an executable file · {link}"
        else:
            text = f"✗ not found on PATH · {link}"
        self.status[key].setText(text)


def _with_browse(edit: QLineEdit, browse) -> QHBoxLayout:
    button = QPushButton("Browse…")
    button.clicked.connect(lambda _: browse())
    row = QHBoxLayout()
    row.addWidget(edit, 1)
    row.addWidget(button)
    return row
