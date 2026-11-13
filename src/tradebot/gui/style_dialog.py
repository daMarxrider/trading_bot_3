# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Dialog to change how an indicator's lines are drawn (#52): colour, width and dash per output."""

from __future__ import annotations

from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (QColorDialog, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QGridLayout, QLabel,
                             QPushButton, QVBoxLayout)

DASH_CHOICES = (("Solid", None), ("Dashed", "dash"), ("Dotted", "dot"))


class StyleDialog(QDialog):
    """``rows`` is ``[(output, current colour "#rrggbb", width, dash)]``; :meth:`overrides` returns what the user
    changed as ``{output: {"color": .., "width": .., "dash": ..}}`` (only the changed fields)."""

    def __init__(self, title: str, rows, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self._initial = {out: (color, width, dash) for out, color, width, dash in rows}
        self._colors = {out: color for out, color, _, _ in rows}
        self.buttons: dict[str, QPushButton] = {}
        self.widths: dict[str, QDoubleSpinBox] = {}
        self.dashes: dict[str, QComboBox] = {}
        lay = QVBoxLayout(self)
        grid = QGridLayout()
        lay.addLayout(grid)
        for r, (out, color, width, dash) in enumerate(rows):
            btn = QPushButton()
            btn.setFixedWidth(70)
            btn.clicked.connect(lambda _, o=out: self._pick(o))
            self.buttons[out] = btn
            spin = QDoubleSpinBox(minimum=0.5, maximum=8.0, singleStep=0.5, decimals=1, value=width)
            box = QComboBox()
            for text, value in DASH_CHOICES:
                box.addItem(text, value)
            box.setCurrentIndex([v for _, v in DASH_CHOICES].index(dash))
            self.widths[out], self.dashes[out] = spin, box
            for c, w in enumerate((QLabel(out), btn, spin, box)):
                grid.addWidget(w, r, c)
            self._paint(out)
        reset = QPushButton("Reset to defaults")
        reset.clicked.connect(self._reset)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(reset)
        lay.addWidget(bb)
        self.reset_requested = False

    def _paint(self, out):
        color = self._colors[out]
        text = "#000" if QColor(color).lightness() > 128 else "#fff"
        self.buttons[out].setText(color)
        self.buttons[out].setStyleSheet(f"background: {color}; color: {text};")

    def _pick(self, out):
        c = QColorDialog.getColor(QColor(self._colors[out]), self, "Line colour")
        if c.isValid():
            self._colors[out] = c.name()
            self._paint(out)

    def _reset(self):
        self.reset_requested = True
        self.accept()

    def overrides(self) -> dict[str, dict]:
        if self.reset_requested:
            return {}
        result = {}
        for out, (color, width, dash) in self._initial.items():
            new = {}
            if self._colors[out].lower() != color.lower():
                new["color"] = self._colors[out]
            if self.widths[out].value() != width:
                new["width"] = self.widths[out].value()
            if self.dashes[out].currentData() != dash:
                new["dash"] = self.dashes[out].currentData() or "solid"
            if new:
                result[out] = new
        return result
