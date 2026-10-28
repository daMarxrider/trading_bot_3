# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Crosshair across all panes of the chart view (#46)."""

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

from PyQt6.QtCore import QPointF  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from tradebot.gui.chart import ChartView, nearest_bar  # noqa: E402
from tradebot.gui.theme import ThemeManager  # noqa: E402
from tradebot.indicators.builtin import MACD, RSI, Bollinger  # noqa: E402

from .conftest import make_ohlc  # noqa: E402


def test_nearest_bar():
    x = np.array([0.0, 10.0, 20.0])
    assert [nearest_bar(x, t) for t in (-5, 4, 6, 14, 16, 99)] == [0, 0, 1, 1, 2, 2]
    assert nearest_bar(np.array([5.0]), 100) == 0


@pytest.fixture
def chart(tmp_path):
    app = QApplication.instance() or QApplication([])
    themes = ThemeManager(user_dir=tmp_path)
    themes.apply("dark")
    c = ChartView(None, themes, ["TEST"])
    c.resize(1100, 800)
    c.df = make_ohlc(300)
    c.indicators = [Bollinger(), RSI(), MACD()]
    c.show()
    c.redraw(autorange=True)
    app.processEvents()
    return c


def scene_pos(plot, x, y):
    return plot.vb.mapViewToScene(QPointF(x, y))


def lines(chart):
    return {plot: chart._cross[plot] for plot in [chart.price, *chart.panes]}


@pytest.mark.parametrize("hovered", [0, 1, 2])  # price pane, RSI pane, MACD pane
def test_hovering_any_pane_draws_the_vertical_line_in_every_pane(chart, hovered):
    plots = [chart.price, *chart.panes]
    assert len(plots) == 3
    plot = plots[hovered]
    x = chart._x
    y = float(np.mean(plot.vb.viewRange()[1]))
    t = x[120] + (x[121] - x[120]) * 0.3  # between two bars, nearer the first
    ranges = [p.vb.viewRange() for p in plots]
    statuses = []
    chart.status.connect(statuses.append)
    chart._mouse_moved(scene_pos(plot, t, y))

    for p, (vline, hline) in lines(chart).items():
        assert vline.value() == x[120]  # same bar in every pane, snapped to its time
        assert hline.isVisible() == (p is plot)  # horizontal line only where the cursor is
    assert lines(chart)[plot][1].value() == pytest.approx(y, rel=1e-3)
    assert [p.vb.viewRange() for p in plots] == ranges  # the crosshair never moves the view
    assert statuses and f"{chart.df.index[120]:%Y-%m-%d}" in statuses[-1]  # OHLC readout of that bar


def test_tap_moves_the_crosshair_too(chart):
    (rsi_pane, _) = chart.panes
    x = chart._x
    chart._tap(scene_pos(rsi_pane, x[200], 50.0))  # what a touch tap does
    assert all(v.value() == x[200] for v, _ in lines(chart).values())
    assert lines(chart)[rsi_pane][1].isVisible() and not chart.hline.isVisible()


def test_crosshair_survives_redraws_and_is_themed(chart):
    chart.indicators = [RSI()]
    chart.redraw()
    assert set(chart._cross) == {chart.price, *chart.panes}  # removed panes' lines are gone, new ones exist
    vline, hline = chart._cross[chart.panes[0]]
    assert vline in chart.panes[0].items and hline in chart.panes[0].items
    chart.themes.apply("light")  # redraws: the panes and their lines are recreated
    colors = {line.pen.color().name() for pair in chart._cross.values() for line in pair}
    assert colors == {"#667089"}  # every crosshair line uses the light theme's muted colour


def test_outside_the_panes_nothing_changes(chart):
    before = [(v.value(), h.isVisible()) for v, h in lines(chart).values()]
    chart._mouse_moved(QPointF(-50, -50))
    assert [(v.value(), h.isVisible()) for v, h in lines(chart).values()] == before
