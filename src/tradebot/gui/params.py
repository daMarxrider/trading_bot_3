# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Auto-generated parameter forms for indicators and algorithms."""

from __future__ import annotations

from PyQt6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout, QSpinBox, QVBoxLayout,
                             QWidget)

from tradebot.core.registry import Param
from tradebot.gui.common import tooltip


class ParamForm(QWidget):
    """One spin box per :class:`Param`, honouring its range and integer flag."""

    def __init__(self, space: dict[str, Param], values: dict | None = None, parent=None):
        super().__init__(parent)
        self.form = QFormLayout(self)
        self.form.setContentsMargins(0, 0, 0, 0)
        self.form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)  # a wide font wraps the field under its label
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
            box.setMinimumWidth(min(box.minimumSizeHint().width(), 90))  # a wide font must not widen the side panel
            self.editors[key] = box
            self.form.addRow(key.replace("_", " "), box)
            self.form.labelForField(box).setWordWrap(True)  # a long name wraps instead of widening the side panel
            if spec.description:
                for widget in (box, self.form.labelForField(box)):
                    widget.setToolTip(tooltip(spec.description))

    def values(self) -> dict:
        return {k: box.value() for k, box in self.editors.items()}


#: what the timeframe box offers (#161): None = the chart's own candles
TIMEFRAME_CHOICES = (None, "15m", "30m", "1h", "2h", "4h", "1d", "1w", "1mo")


class ParamDialog(QDialog):
    def __init__(self, title: str, space: dict[str, Param], values: dict | None = None, parent=None,
                 timeframe: bool = False, tf: str | None = None):
        super().__init__(parent)
        self.setWindowTitle(title)
        lay = QVBoxLayout(self)
        self.form = ParamForm(space, values)
        lay.addWidget(self.form)
        self.tf_box = None
        if timeframe:  # a higher timeframe to compute on (#161)
            self.tf_box = QComboBox()
            for choice in TIMEFRAME_CHOICES:
                self.tf_box.addItem(choice or "chart's own", choice)
            self.tf_box.setCurrentIndex(max(0, self.tf_box.findData(tf)))
            self.tf_box.setToolTip(tooltip(
                "Compute the indicator on longer candles than the chart's, e.g. the daily SMA on an hourly chart. "
                "Each candle sees the last closed longer candle only (no look-ahead), so the line steps."))
            self.form.form.addRow("timeframe", self.tf_box)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

    def values(self) -> dict:
        """The parameters, plus ``"tf"`` when a higher timeframe was picked."""
        tf = self.tf_box.currentData() if self.tf_box is not None else None
        return {**self.form.values(), **({"tf": tf} if tf else {})}

    @classmethod
    def ask(cls, title, space, values=None, parent=None, timeframe: bool = False, tf: str | None = None) -> dict | None:
        """The values entered, or None if cancelled. ``timeframe``: also offer a higher timeframe (``"tf"``)."""
        if not space and not timeframe:
            return {}
        dlg = cls(title, space, values, parent, timeframe, tf)
        return dlg.values() if dlg.exec() == QDialog.DialogCode.Accepted else None
