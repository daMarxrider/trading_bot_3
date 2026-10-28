# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Auto-generated parameter forms for indicators and algorithms."""

from __future__ import annotations

from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout, QSpinBox, QVBoxLayout, QWidget

from tradebot.core.registry import Param


class ParamForm(QWidget):
    """One spin box per :class:`Param`, honouring its range and integer flag."""

    def __init__(self, space: dict[str, Param], values: dict | None = None, parent=None):
        super().__init__(parent)
        self.form = QFormLayout(self)
        self.form.setContentsMargins(0, 0, 0, 0)
        self.editors: dict[str, QSpinBox | QDoubleSpinBox] = {}
        self.set_space(space, values)

    def set_space(self, space: dict[str, Param], values: dict | None = None):
        while self.form.rowCount():
            self.form.removeRow(0)
        self.editors.clear()
        values = values or {}
        for key, spec in space.items():
            if spec.integer:
                box = QSpinBox()
                box.setRange(int(spec.low), int(spec.high))
            else:
                box = QDoubleSpinBox()
                box.setDecimals(4 if spec.high - spec.low < 10 else 2)
                box.setRange(spec.low, spec.high)
                box.setSingleStep((spec.high - spec.low) / 100 or 0.1)
            box.setValue(values.get(key, spec.default))
            if spec.description:
                box.setToolTip(spec.description)
            self.editors[key] = box
            self.form.addRow(key.replace("_", " "), box)

    def values(self) -> dict:
        return {k: box.value() for k, box in self.editors.items()}


class ParamDialog(QDialog):
    def __init__(self, title: str, space: dict[str, Param], values: dict | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        lay = QVBoxLayout(self)
        self.form = ParamForm(space, values)
        lay.addWidget(self.form)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

    @classmethod
    def ask(cls, title, space, values=None, parent=None) -> dict | None:
        if not space:
            return {}
        dlg = cls(title, space, values, parent)
        return dlg.form.values() if dlg.exec() == QDialog.DialogCode.Accepted else None
