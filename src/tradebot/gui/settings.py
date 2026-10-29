# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""The settings dialog behind the sidebar's gear button (#59): theme and paths to third-party tools."""

from __future__ import annotations

from html import escape

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)

from tradebot import tools
from tradebot.config import config_path
from tradebot.gui.theme import ThemeManager


class SettingsDialog(QDialog):
    """Changes apply and are saved to config.toml right away; Close just closes."""

    theme_chosen = pyqtSignal(str)
    message = pyqtSignal(str)  #: e.g. a setting that couldn't be saved, for the status bar

    def __init__(self, themes: ThemeManager, theme_key: str, parent=None):
        super().__init__(parent)
        self.themes = themes
        self.setWindowTitle("Settings")
        self.setMinimumWidth(560)
        lay = QVBoxLayout(self)

        look = QGroupBox("Appearance")
        form = QFormLayout(look)
        self.theme_box = QComboBox()
        for key, theme in themes.themes.items():
            self.theme_box.addItem(theme.name, key)
        self.theme_box.setCurrentIndex(max(self.theme_box.findData(theme_key), 0))
        self.theme_box.currentIndexChanged.connect(lambda _: self.theme_chosen.emit(self.theme_box.currentData()))
        form.addRow("Theme", self.theme_box)
        lay.addWidget(look)

        box = QGroupBox("Third-party tools")
        tools_lay = QVBoxLayout(box)
        note = QLabel("tradebot doesn't include these tools. Install them yourself; leave a path empty to find "
                      "the tool on PATH.", objectName="Muted", wordWrap=True)
        tools_lay.addWidget(note)
        self.paths: dict[str, QLineEdit] = {}
        self.status: dict[str, QLabel] = {}
        form = QFormLayout()
        for t in tools.TOOLS:
            edit = QLineEdit(tools.configured_path(t.key), placeholderText=f"{t.command} (on PATH)")
            edit.setToolTip(f"Used for: {t.used_for}")
            edit.editingFinished.connect(lambda k=t.key: self._save_path(k))
            browse = QPushButton("Browse…")
            browse.clicked.connect(lambda _, k=t.key: self._browse(k))
            row = QHBoxLayout()
            row.addWidget(edit, 1)
            row.addWidget(browse)
            status = QLabel(openExternalLinks=True, wordWrap=True)
            self.paths[t.key], self.status[t.key] = edit, status
            form.addRow(t.name, row)
            form.addRow("", status)
            self._show_status(t.key)
        tools_lay.addLayout(form)
        lay.addWidget(box)

        themes.changed.connect(self._refresh)  # the links take the new theme's accent colour
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

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
