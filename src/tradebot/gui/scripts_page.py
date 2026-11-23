# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""The Scripts page (#191): your indicator and strategy scripts in a list, with the editor beside it.

The editor is the one the Scanner uses (:class:`tradebot.gui.script_editor.ScriptEditor`); this page just gives
the files a home of their own, with a New button, so writing a script doesn't start from a scan.
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QToolButton, QVBoxLayout, QWidget

from tradebot import scripts
from tradebot.gui.icons import bind_icon
from tradebot.gui.script_editor import ScriptEditor
from tradebot.gui.theme import ThemeManager


class ScriptsView(QWidget):
    status = pyqtSignal(str)
    #: a reload succeeded: the names of the indicators the script defines
    reloaded = pyqtSignal(list)

    def __init__(self, themes: ThemeManager, sample=None):
        super().__init__()
        self.themes = themes
        self.list = QListWidget(objectName="ScriptList")
        self.list.currentItemChanged.connect(self._picked)
        self.new_btn = QToolButton(toolTip="New script")
        bind_icon(self.new_btn, "plus", themes, off="text")
        self.new_btn.clicked.connect(self.new_script)
        self.editor = ScriptEditor(themes, sample)
        self.editor.close_btn.hide()  # a page, not a pane: nothing to close
        self.editor.reloaded.connect(self._reloaded)
        self.editor.status.connect(self.status)
        self.where = QLabel(objectName="Muted", wordWrap=True)

        head = QHBoxLayout()
        head.addWidget(QLabel("Indicators and strategies", objectName="SidebarTitle"), 1)
        head.addWidget(self.new_btn)
        left = QFrame(objectName="Card")
        left.setFixedWidth(260)
        col = QVBoxLayout(left)
        col.setContentsMargins(12, 10, 12, 10)
        col.addLayout(head)
        col.addWidget(self.list, 1)
        col.addWidget(self.where)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(12)
        lay.addWidget(left)
        lay.addWidget(self.editor, 1)
        self.refresh()

    def refresh(self, select=None):
        """Re-read the scripts folder; keep ``select`` (a path) or the open script highlighted."""
        folder = scripts.plugin_dir()
        self.where.setText(f"Scripts live in {folder}")
        keep = select or self.editor.path
        self.list.blockSignals(True)
        self.list.clear()
        for path in sorted(folder.glob("*.py")) if folder.is_dir() else []:
            item = QListWidgetItem(path.name)
            item.setData(Qt.ItemDataRole.UserRole, str(path))
            self.list.addItem(item)
            if keep is not None and path == keep:
                self.list.setCurrentItem(item)
        self.list.blockSignals(False)

    def showEvent(self, event):
        super().showEvent(event)
        self.refresh()

    def _picked(self, item, _previous):
        if item is None:
            return
        path = Path(item.data(Qt.ItemDataRole.UserRole))
        if path != self.editor.path and not self.editor.open_script(path):
            self.refresh()  # the user cancelled because of unsaved changes: back to the open script

    def new_script(self):
        path, source = scripts.new_script()
        self.editor.open_script(path, source, saved=False)

    def _reloaded(self, names: list):
        self.refresh()
        self.reloaded.emit(names)
