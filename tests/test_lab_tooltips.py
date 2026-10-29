# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Backtest lab: every input, choice and result explains itself on hover, and on a long press by touch (#70)."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

from PyQt6.QtCore import QEvent, QPoint, QPointF, Qt  # noqa: E402
from PyQt6.QtGui import QInputDevice, QMouseEvent, QPointingDevice  # noqa: E402
from PyQt6.QtTest import QTest  # noqa: E402
from PyQt6.QtWidgets import QApplication, QFormLayout, QLabel, QToolTip  # noqa: E402

from tradebot.algorithms import build_algorithm  # noqa: E402
from tradebot.backtest import backtest  # noqa: E402
from tradebot.glossary import METRICS, OBJECTIVES, STRATEGIES, TERMS  # noqa: E402
from tradebot.gui.theme import ThemeManager  # noqa: E402
from tradebot.gui.touch import tooltip_at  # noqa: E402
from tradebot.gui.views import BacktestView  # noqa: E402

from .conftest import make_ohlc  # noqa: E402

TIP = Qt.ItemDataRole.ToolTipRole


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def lab(app, tmp_path):
    themes = ThemeManager(user_dir=tmp_path)
    themes.apply("dark")
    view = BacktestView(None, themes, ["TEST"])
    view.resize(1200, 800)
    view.show()
    app.processEvents()
    yield view
    view.close()  # windows of earlier tests would otherwise cover this one's
    view.deleteLater()
    app.processEvents()


def test_every_form_row_explains_itself(lab):
    forms = lab.findChildren(QFormLayout)
    rows = [(form.itemAt(i, QFormLayout.ItemRole.LabelRole), form.itemAt(i, QFormLayout.ItemRole.FieldRole))
            for form in forms for i in range(form.rowCount())]
    assert len(rows) >= 10  # setup, optimizer and the strategy's parameters
    for label, field in rows:
        assert label.widget().toolTip(), label.widget().text()
        assert field.widget().toolTip(), label.widget().text()
    assert "0.1%" in lab.fee.toolTip()


def test_every_choice_explains_itself(lab):
    for i in range(lab.objective.count()):
        key = lab.objective.itemText(i)
        assert OBJECTIVES[key].split(".")[0] in lab.objective.itemData(i, TIP)
        assert OBJECTIVES[key].split(".")[0] in lab.objective.toolTip()  # all choices, also for touch
    assert all(lab.method.itemData(i, TIP) for i in range(lab.method.count()))
    builtin = [i for i in range(lab.algo.count()) if lab.algo.itemText(i) in STRATEGIES]  # other tests add more
    assert len(builtin) >= 7 and all(lab.algo.itemData(i, TIP) for i in builtin)


def test_results_explain_each_metric_and_column(lab):
    df = make_ohlc(400)
    res = backtest(df, build_algorithm("sma_cross"))
    lab._show((df, {"in-sample": res, "out-of-sample": res}))
    model = lab.metrics
    for row in range(model.rowCount()):
        metric = model.df.iat[row, 0]
        for col in range(model.columnCount()):
            assert METRICS[metric].split(".")[0] in model.data(model.index(row, col), TIP)
    headers = [model.headerData(c, Qt.Orientation.Horizontal, TIP) for c in range(model.columnCount())]
    assert "never saw" in headers[2] and "tuned" in headers[1] and headers[0]
    assert lab.verdict.toolTip() and "Fragile" in lab.verdict.toolTip()


def test_tooltips_wrap(lab):
    assert lab.population.toolTip().startswith("<qt>")  # rich text: Qt wraps it instead of one wide line


def touch_press(widget, pos, kind=QEvent.Type.MouseButtonPress):
    device = QPointingDevice("finger", 1, QInputDevice.DeviceType.TouchScreen, QPointingDevice.PointerType.Finger,
                             QInputDevice.Capability.Position, 1, 0)
    local = QPointF(pos)
    event = QMouseEvent(kind, local, QPointF(widget.mapToGlobal(pos)), Qt.MouseButton.LeftButton,
                        Qt.MouseButton.LeftButton if kind == QEvent.Type.MouseButtonPress else Qt.MouseButton.NoButton,
                        Qt.KeyboardModifier.NoModifier, device)
    QApplication.sendEvent(widget, event)


def test_long_press_by_touch_shows_the_tooltip(lab):
    field = lab.population
    center = field.rect().center()
    touch_press(field, center)
    QTest.qWait(700)
    assert QToolTip.isVisible() and QToolTip.text() == field.toolTip()
    QToolTip.hideText()

    touch_press(field, center)  # a short tap shows nothing
    touch_press(field, center, QEvent.Type.MouseButtonRelease)
    QTest.qWait(700)
    assert not QToolTip.isVisible()


def test_tooltip_at_finds_table_cells_headers_and_labels(lab):
    df = make_ohlc(400)
    lab._show((df, {"backtest": backtest(df, build_algorithm("sma_cross"))}))
    QApplication.processEvents()
    table = lab.metrics_view
    cell = table.visualRect(lab.metrics.index(0, 1)).center()
    metric = lab.metrics.df.iat[0, 0]
    assert METRICS[metric].split(".")[0] in tooltip_at(table.viewport(), table.viewport().mapToGlobal(cell))
    header = table.horizontalHeader()
    x = header.sectionViewportPosition(1) + 5
    assert "replayed" in tooltip_at(header.viewport(), header.viewport().mapToGlobal(QPoint(x, 5)))
    title = next(w for w in lab.findChildren(QLabel) if w.text() == "Parameters")
    assert tooltip_at(title, title.mapToGlobal(QPoint(1, 1))) == title.toolTip()
    assert TERMS["parameters"].split(".")[0] in title.toolTip()
