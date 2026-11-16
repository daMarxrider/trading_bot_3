# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Journal tab (#159): the orders the bots placed, realised and unrealised P&L, notes and tags, CSV export."""

from __future__ import annotations

import pandas as pd
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from tradebot.gui.common import DataFrameModel, card, run_async
from tradebot.gui.theme import ThemeManager
from tradebot.journal import Journal, fifo_lots, unrealised

COLUMNS = ["id", "time", "mode", "symbol", "side", "amount", "price", "fee", "pnl", "tags", "note"]
MODES = [("Live and paper", "real"), ("Live", "live"), ("Paper", "paper"), ("Dry runs", "dry"), ("All", "")]


class JournalView(QWidget):
    status = pyqtSignal(str)

    def __init__(self, engine, themes: ThemeManager, journal_factory=Journal):
        super().__init__()
        self.engine, self.themes, self._factory = engine, themes, journal_factory
        self.mode = QComboBox(toolTip="Which orders to show: dry runs are only intended fills, nothing was sent")
        for label, key in MODES:
            self.mode.addItem(label, key)
        self.mode.currentIndexChanged.connect(self.refresh)
        self.tag = QLineEdit(placeholderText="filter by tag")
        self.tag.editingFinished.connect(self.refresh)
        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        self._base = ""
        self.model = DataFrameModel(formats={
            "price": lambda v: f"{v:,.6g}", "amount": lambda v: f"{v:,.6g}", "fee": lambda v: f"{v:,.4g}",
            "pnl": lambda v: f"{v:+,.2f}", "time": lambda v: str(v)[:16].replace("T", " ")})
        self.table = QTableView()
        self.table.setModel(self.model)
        self.table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.clicked.connect(self._selected)
        self.note = QLineEdit(placeholderText="note for the selected trade")
        self.tags = QLineEdit(placeholderText="tags, comma-separated")
        save = QPushButton("Save note and tags")
        save.clicked.connect(self.save_annotation)
        trades_csv = QPushButton("Export trades (CSV)")
        trades_csv.clicked.connect(lambda: self.export(False))
        lots_csv = QPushButton("Export FIFO tax lots (CSV)", toolTip="One row per closed piece, with the year it "
                                                                   "closed in, cost, proceeds, fees and profit")
        lots_csv.clicked.connect(lambda: self.export(True))
        refresh = QPushButton("Refresh", objectName="Primary")
        refresh.clicked.connect(self.refresh)
        top = QHBoxLayout()
        for w in (QLabel("Show"), self.mode, self.tag):
            top.addWidget(w)
        top.addStretch()
        for w in (trades_csv, lots_csv, refresh):
            top.addWidget(w)
        edit = QHBoxLayout()
        edit.addWidget(self.note, 3)
        edit.addWidget(self.tags, 1)
        edit.addWidget(save)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 16)
        title = QLabel("Journal")
        title.setObjectName("Title")
        lay.addWidget(title)
        lay.addWidget(card(_row(top), self.summary))
        lay.addWidget(self.table, 1)
        lay.addWidget(card(_row(edit)))
        self.refresh()

    def _mode(self) -> str | None:
        return self.mode.currentData() or None

    def refresh(self, *_):
        j = self._factory()
        try:
            rows = j.trades(self._mode(), tag=self.tag.text().strip() or None)
            s = j.summary(self._mode())
            opens = fifo_lots(j.fills(self._mode()))[1]
        finally:
            j.close()
        df = pd.DataFrame(rows, columns=COLUMNS + ["broker", "bot", "order_id", "status"])[COLUMNS] if rows \
            else pd.DataFrame(columns=COLUMNS)
        self.model.set_frame(df.iloc[::-1])
        wr = "–" if s["win_rate"] is None else f"{s['win_rate']:.0%}"
        self._base = (f"{s['fills']} orders · {s['lots']} closed lots · win rate {wr} · realised "
                      f"<b>{s['realised']:+,.2f}</b> · fees {s['fees']:,.2f}")
        self.summary.setText(self._base + (f" · {s['open']} open piece(s), valuing …" if opens else ""))
        symbols = sorted({o["symbol"] for o in opens})
        if not symbols:
            return

        def work():
            prices = {}
            for sym in symbols:
                try:
                    prices[sym] = self.engine.latest_price(sym)
                except Exception:
                    pass
            return prices

        run_async(work, lambda prices: self._valued(opens, prices),
                  lambda err: self.summary.setText(self._base + f" · unrealised: {err}"))

    def _valued(self, opens, prices):
        un = unrealised(opens, prices)
        self.summary.setText(self._base + f" · unrealised <b>{sum(un.values()):+,.2f}</b> "
                             f"({len(un)} of {len({o['symbol'] for o in opens})} symbols priced)")

    def _selected(self, index):
        row = self.model.df.iloc[index.row()]
        self.note.setText("" if pd.isna(row["note"]) else str(row["note"]))
        self.tags.setText("" if pd.isna(row["tags"]) else str(row["tags"]))

    def save_annotation(self):
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            self.status.emit("journal: select a trade first")
            return
        fill_id = int(self.model.df.iloc[rows[0].row()]["id"])
        j = self._factory()
        try:
            j.annotate(fill_id, self.note.text(), self.tags.text())
        finally:
            j.close()
        self.refresh()
        self.status.emit(f"journal: trade #{fill_id} updated")

    def export(self, lots: bool):
        path, _ = QFileDialog.getSaveFileName(self, "Export", "tax_lots.csv" if lots else "trades.csv", "CSV (*.csv)")
        if not path:
            return
        j = self._factory()
        try:
            n = j.export_lots(path, self._mode()) if lots else j.export_trades(path, self._mode())
        finally:
            j.close()
        self.status.emit(f"journal: {n} rows written to {path}")


def _row(layout) -> QWidget:
    w = QWidget()
    layout.setContentsMargins(0, 0, 0, 0)
    w.setLayout(layout)
    return w
