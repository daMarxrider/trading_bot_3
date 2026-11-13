# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""The settings dialog behind the sidebar's gear button (#59).

Categories on the left, one page each: Appearance (theme, GPU rendering), Integrations (third-party tools, broker
API keys), Notifications (#155) and Data & cache (the disk cache, #63). A new category is one more :meth:`SettingsDialog._add_page` call.
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

from tradebot import notify, tools
from tradebot.brokers import BROKERS
from tradebot.config import config_path, load_config, os_cache_dir, save_config_value
from tradebot.data.cache import DEFAULT_MAX_MB, DiskCache, human_size
from tradebot.gui.theme import ThemeManager

#: How a credential key is labelled; others read "Some key" from "some_key"
KEY_LABELS = {"api_key": "API key", "secret": "Secret", "password": "Passphrase", "holdings_csv": "Holdings CSV",
              "phone_no": "Phone number", "pin": "PIN"}
SECRET_KEYS = {"secret", "password", "pin"}  #: shown as dots

#: The channels the dialog edits: config table (``[notify.<key>]``) -> title
NOTIFY_CHANNELS = {"telegram": "Telegram bot", "discord": "Discord webhook", "webhook": "Generic webhook (JSON)",
                   "email": "Email (SMTP)", "desktop": "Desktop notification"}
NOTIFY_SECRETS = {"token", "password", "url"}  #: shown as dots (a webhook URL is a secret too)
NOTIFY_LABELS = {"chat_id": "Chat ID", "url": "Webhook URL", "host": "SMTP server", "to": "To", "user": "User",
                 "from": "From (optional)"}
NOTIFY_EVENTS = {"trade": "A bot bought or sold", "error": "A bot step failed", "risk": "A risk guard stopped trading",
                 "scanner": "Matches of a scan sent with tradebot scan --notify"}


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
        self.notify_fields: dict[tuple[str, str], QLineEdit] = {}  # (channel type, option) -> field
        self.notify_events: dict[str, dict[str, QCheckBox]] = {}  # channel type -> event kind -> box
        self.notify_enabled: dict[str, QCheckBox] = {}
        self.page_names: list[str] = []  # the categories, in the order of the left bar

        self._add_page("Appearance", "🎨", self._appearance(theme_key))
        self._add_page("Integrations", "🔌", self._integrations())
        self._add_page("Notifications", "🔔", self._notifications())
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
        lay.addWidget(self._rendering_box())
        lay.addStretch()
        return page

    def _rendering_box(self) -> QGroupBox:
        """GPU rendering of the chart (#71): off by default, read at startup."""
        box = QGroupBox("Rendering")
        lay = QVBoxLayout(box)
        self.gpu_box = QCheckBox("GPU rendering (experimental)")
        self.gpu_box.setChecked(bool(load_config().get("gpu_rendering", False)))
        self.gpu_box.toggled.connect(self._save_gpu_rendering)
        lay.addWidget(self.gpu_box)
        lay.addWidget(QLabel("Draws the chart through OpenGL, for smoother panning. Applies after a restart. "
                             "Without OpenGL (no driver, a VM, remote desktop) the chart uses the normal renderer.",
                             objectName="Muted", wordWrap=True))
        return box

    def _save_gpu_rendering(self, on: bool):
        if save_config_value("gpu_rendering", on):
            self.message.emit(f"GPU rendering {'on' if on else 'off'}: restart tradebot to apply")
        else:
            self.message.emit(f"GPU rendering not saved: couldn't write {config_path()}")

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

    # -- Notifications ---------------------------------------------------------------------------
    def _notifications(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 4, 0)
        lay.addWidget(QLabel("Get bot trades, failed steps, risk-guard stops and scanner hits on your phone or desktop. "
                             "Saved in plain text in config.toml; to keep a secret out of the file, write "
                             "$ENV_NAME as the value (e.g. $TELEGRAM_TOKEN) and set that environment variable. "
                             "More channels or other names: edit the [notify.*] tables by hand.",
                             objectName="Muted", wordWrap=True))
        saved = load_config().get("notify", {})
        for kind, title in NOTIFY_CHANNELS.items():
            conf = saved.get(kind, {}) if isinstance(saved.get(kind), dict) else {}
            box = QGroupBox(title)
            form = QFormLayout(box)
            enabled = QCheckBox("Enabled")
            enabled.setChecked(bool(conf.get("enabled", True)) and bool(conf))
            enabled.setToolTip("Needs the settings below; a channel without any is off.")
            enabled.toggled.connect(lambda on, k=kind: self._save_notify(k, "enabled", on))
            self.notify_enabled[kind] = enabled
            form.addRow(enabled)
            for key in notify.CHANNEL_OPTIONS[kind]:
                edit = QLineEdit(str(conf.get(key, "")))
                if key in NOTIFY_SECRETS:
                    edit.setEchoMode(QLineEdit.EchoMode.Password)
                edit.editingFinished.connect(lambda k=kind, o=key: self._save_notify_field(k, o))
                self.notify_fields[kind, key] = edit
                form.addRow(NOTIFY_LABELS.get(key, key.capitalize()), edit)
            events = QHBoxLayout()
            wanted = conf.get("events", notify.KINDS)
            self.notify_events[kind] = {}
            for ev in notify.KINDS[:-1]:  # "test" always goes through
                cb = QCheckBox(ev)
                cb.setChecked(ev in wanted)
                cb.setToolTip(NOTIFY_EVENTS[ev])
                cb.toggled.connect(lambda _, k=kind: self._save_notify_events(k))
                self.notify_events[kind][ev] = cb
                events.addWidget(cb)
            events.addStretch()
            form.addRow("Send", events)
            lay.addWidget(box)
        row = QHBoxLayout()
        self.notify_test_btn = QPushButton("Send test message")
        self.notify_test_btn.setToolTip("Sends a test to every enabled channel (also those in config.toml only), "
                                        "whatever their event filter.")
        self.notify_test_btn.clicked.connect(lambda _: self.test_notifications())
        self.notify_result = QLabel(wordWrap=True)
        row.addWidget(self.notify_test_btn)
        row.addWidget(self.notify_result, 1)
        lay.addLayout(row)
        lay.addStretch()
        return page

    def test_notifications(self) -> dict[str, str | None]:
        """Send the test message and show per channel whether it worked (the send blocks for up to its timeout)."""
        results = notify.test_all()
        lines = [f"✗ {name}: {escape(err)}" if err else f"✓ {name}" for name, err in results.items()]
        self.notify_result.setText("<br>".join(lines) or "No channel is enabled.")
        return results

    def _save_notify(self, kind: str, key: str, value) -> bool:
        table = f"notify.{kind}"
        ok = save_config_value("type", kind, table=table) and save_config_value(key, value, table=table)
        if not ok:
            self.message.emit(f"notification setting not saved: couldn't write {config_path()}")
        return ok

    def _save_notify_field(self, kind: str, key: str):
        text = self.notify_fields[kind, key].text().strip()
        current = load_config().get("notify", {}).get(kind, {}).get(key, "")
        if text == str(current):
            return
        value = int(text) if key == "port" and text.isdigit() else text or None
        if self._save_notify(kind, key, value) and text and not self.notify_enabled[kind].isChecked() \
                and not load_config()["notify"][kind].get("enabled", False):
            self.notify_enabled[kind].setChecked(True)  # filling in a channel switches it on

    def _save_notify_events(self, kind: str):
        chosen = [ev for ev, cb in self.notify_events[kind].items() if cb.isChecked()]
        self._save_notify(kind, "events", chosen)

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
