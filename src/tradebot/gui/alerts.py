# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Background alerts in the app (#156): the timer that runs :func:`tradebot.alerts.run_due` and the list dialog."""

from __future__ import annotations

from PyQt6.QtCore import QObject, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from tradebot import alerts
from tradebot.gui.common import run_async

POLL_MS = 30_000


class AlertWatcher(QObject):
    """Checks the stored alerts every 30 s in a worker thread. Nothing runs until :meth:`start`."""

    fired = pyqtSignal(str)  #: a log line per alert that fired or failed

    def __init__(self, parent=None, poll_ms: int = POLL_MS):
        super().__init__(parent)
        self._busy = False
        self.timer = QTimer(self)
        self.timer.setInterval(poll_ms)
        self.timer.timeout.connect(self.tick)

    def start(self):
        self.timer.start()
        self.tick()

    def stop(self):
        self.timer.stop()

    def tick(self):
        if self._busy or not alerts.load():
            return
        self._busy = True

        def done(lines):
            self._busy = False
            for line in lines:
                self.fired.emit(line)

        run_async(alerts.run_due, done, lambda err: setattr(self, "_busy", False))


class AlertsDialog(QDialog):
    """The stored alerts: switch on/off, remove."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Alerts")
        self.resize(640, 360)
        lay = QVBoxLayout(self)
        self.hint = QLabel("Add price alerts from the chart (right-click a level); indicator alerts and scheduled "
                           "scans with `tradebot alert`. They are checked while the app or `tradebot watch` runs.")
        self.hint.setWordWrap(True)
        lay.addWidget(self.hint)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Kind", "Condition", "Every", "State"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        lay.addWidget(self.table, 1)
        row = QHBoxLayout()
        self.toggle_btn = QPushButton("Enable / disable")
        self.toggle_btn.clicked.connect(self.toggle_selected)
        self.remove_btn = QPushButton("Remove")
        self.remove_btn.clicked.connect(self.remove_selected)
        row.addWidget(self.toggle_btn)
        row.addWidget(self.remove_btn)
        row.addStretch()
        lay.addLayout(row)
        self.refresh()

    def refresh(self):
        self.items = alerts.load()
        self.table.setRowCount(len(self.items))
        for r, a in enumerate(self.items):
            state = "off" if not a.enabled else "armed" if a.armed else "fired (waiting)"
            for c, text in enumerate((a.kind, a.describe(), str(a.every), state)):
                self.table.setItem(r, c, QTableWidgetItem(text))

    def _selected(self):
        rows = self.table.selectionModel().selectedRows()
        return self.items[rows[0].row()] if rows else None

    def toggle_selected(self):
        a = self._selected()
        if a:
            alerts.update(a.id, enabled=not a.enabled, armed=True, last_run=0.0)
            self.refresh()

    def remove_selected(self):
        a = self._selected()
        if a:
            alerts.remove(a.id)
            self.refresh()
