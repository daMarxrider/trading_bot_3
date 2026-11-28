# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Auto-generated parameter forms for indicators and algorithms."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
                             QFormLayout, QFrame, QGridLayout, QLabel, QScrollArea, QSpinBox, QVBoxLayout, QWidget)

from tradebot.core.registry import Param
from tradebot.gui.common import tooltip


def _spin(spec: Param, value) -> QSpinBox | QDoubleSpinBox:
    if spec.integer:
        box = QSpinBox()
        box.setRange(int(spec.low), int(spec.high))
    else:
        box = QDoubleSpinBox()
        box.setDecimals(4 if spec.high - spec.low < 10 else 2)
        box.setRange(spec.low, spec.high)
        box.setSingleStep((spec.high - spec.low) / 100 or 0.1)
    box.setValue(value)
    if spec.description:
        box.setToolTip(tooltip(spec.description))
    return box


def is_switch(key: str, spec: Param) -> bool:
    """A 0/1 parameter that turns a part on or off (``use.<part>`` in a pool): a check box, not a number."""
    return key.startswith("use.") and spec.integer and spec.low == 0 and spec.high == 1


def grouped(space: dict[str, Param]) -> bool:
    """Whether ``space`` is a set of parts (``part.setting`` names, e.g. a pool or a combo of strategies): such a
    form is laid out per part instead of as one long column."""
    return sum("." in k for k in space) >= 6


class GroupedParamForm(QWidget):
    """The parameters of a strategy made of parts (#230): the plain ones first, then a section per part, its
    ``use.<part>`` switch as a check box in the heading (off dims its settings) and its settings two to a row, in one
    grid so the columns line up from part to part."""

    COLUMNS = 2  #: label + editor pairs per row

    def __init__(self, space: dict[str, Param], values: dict | None = None, parent=None):
        super().__init__(parent)
        values = values or {}
        self.editors: dict[str, QSpinBox | QDoubleSpinBox | QCheckBox] = {}
        self.grid = QGridLayout(self)
        self.grid.setContentsMargins(0, 0, 8, 0)  # room for the scroll bar
        self.grid.setHorizontalSpacing(12)
        self.grid.setVerticalSpacing(6)
        parts: dict[str, list[str]] = {}
        plain = []
        for key in space:
            if key.startswith("use."):
                parts.setdefault(key[4:], [])
            elif "." in key:
                parts.setdefault(key.split(".", 1)[0], []).append(key)
            else:
                plain.append(key)
        self._row = 0
        if plain:
            self._settings(None, plain, space, values)
        for part, keys in parts.items():
            self._settings(part, keys, space, values)
        for c in range(self.COLUMNS):
            self.grid.setColumnStretch(c * 2, 1)
        self.grid.setRowStretch(self._row, 1)

    def _settings(self, part: str | None, keys: list[str], space, values):
        editors: list[QWidget] = []
        if part is not None:
            switch_key, title = f"use.{part}", part.replace("_", " ")
            if switch_key in space and is_switch(switch_key, space[switch_key]):
                head = QCheckBox(title)
                head.setChecked(bool(values.get(switch_key, space[switch_key].default)))
                head.setToolTip(tooltip(space[switch_key].description or f"Use {title}"))
                head.toggled.connect(lambda on: [e.setEnabled(on) for e in editors])
                self.editors[switch_key] = head
            else:
                head = QLabel(title)
            font = head.font()
            font.setWeight(font.Weight.DemiBold)
            head.setFont(font)
            if self._row:
                self.grid.setRowMinimumHeight(self._row, 10)  # a gap above every part but the first
                self._row += 1
            self.grid.addWidget(head, self._row, 0, 1, 2 * self.COLUMNS)
            self._row += 1
        for i, key in enumerate(keys):
            spec = space[key]
            label = QLabel(key.split(".", 1)[-1].replace("_", " "))
            editor = _spin(spec, values.get(key, spec.default))
            if spec.description:
                label.setToolTip(tooltip(spec.description))
            self.editors[key] = editor
            editors.extend([label, editor])
            r, c = self._row + i // self.COLUMNS, (i % self.COLUMNS) * 2
            self.grid.addWidget(label, r, c)
            self.grid.addWidget(editor, r, c + 1)
        self._row += -(-len(keys) // self.COLUMNS)
        switch = self.editors.get(f"use.{part}") if part is not None else None
        if isinstance(switch, QCheckBox) and not switch.isChecked():
            for e in editors:
                e.setEnabled(False)

    def values(self) -> dict:
        return {k: int(e.isChecked()) if isinstance(e, QCheckBox) else e.value() for k, e in self.editors.items()}


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
            box = _spin(spec, values.get(key, spec.default))
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
        if grouped(space):  # a strategy of parts: sections, scrolling inside a dialog that fits the screen (#230)
            self.form = GroupedParamForm(space, values)
            scroll = QScrollArea(widgetResizable=True, frameShape=QFrame.Shape.NoFrame, objectName="Pane")
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            scroll.setWidget(self.form)
            self.form.setAutoFillBackground(False)  # setWidget() turns it on: the palette's light window colour
            lay.addWidget(scroll, 1)
            screen = (parent.screen() if parent is not None else None) or QApplication.primaryScreen()
            room = screen.availableGeometry() if screen is not None else None
            hint = self.form.sizeHint()
            width = max(560, hint.width() + 40)
            height = hint.height() + 90
            if room is not None:
                width, height = min(width, int(room.width() * 0.9)), min(height, int(room.height() * 0.85))
            self.resize(width, height)
        else:
            self.form = ParamForm(space, values)
            lay.addWidget(self.form)
        self.tf_box = None
        if timeframe and isinstance(self.form, ParamForm):  # a higher timeframe to compute on (#161)
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
