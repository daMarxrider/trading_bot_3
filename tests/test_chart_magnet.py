"""The chart magnet: toggle, persistence, snapping of clicks and handle drags (#197)."""

import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

import pyqtgraph as pg  # noqa: E402
from PyQt6.QtCore import QPointF  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from tradebot.config import Settings  # noqa: E402
from tradebot.gui.chart import ChartView  # noqa: E402
from tradebot.gui.theme import ThemeManager  # noqa: E402


def candles(n=200):
    close = np.linspace(100, 200, n)
    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99, "close": close,
                         "volume": 1.0}, index=idx)


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def themes(app, tmp_path):
    t = ThemeManager(user_dir=tmp_path)
    t.apply("dark")
    return t


def make(themes, path):
    c = ChartView(None, themes, ["TEST"], Settings(path))
    c.resize(1000, 700)
    c.df = candles()
    c.show()
    c.redraw(autorange=True)
    c._switch_drawings("AAA")
    c._loaded_symbol = "AAA"
    QApplication.processEvents()
    return c


@pytest.fixture
def chart(themes, tmp_path):
    c = make(themes, tmp_path / "s.json")
    yield c
    c.close()
    c.deleteLater()


def view_pt(chart, i, dy=0.0):
    return QPointF(chart._x[i], chart.df["close"].iloc[i] + dy)


def test_toggle_persists(themes, tmp_path):
    path = tmp_path / "s.json"
    c = make(themes, path)
    try:
        assert not c.magnet_btn.isChecked()
        c.magnet_btn.setChecked(True)
    finally:
        c.close()
        c.deleteLater()
    c2 = make(themes, path)
    try:
        assert c2.magnet_btn.isChecked()
    finally:
        c2.close()
        c2.deleteLater()


def test_snap_on_off_and_radius(chart):
    i = 100
    dy = chart.price.vb.viewPixelSize()[1]
    near = view_pt(chart, i, 3 * dy)
    far = view_pt(chart, i, 80 * dy)
    pt, which = chart.snap_view_point(near)
    assert which is None and pt == near  # off
    chart.magnet_btn.setChecked(True)
    pt, which = chart.snap_view_point(near)
    assert which in ("close", "open") and pt.x() == chart._x[i] and pt.y() == chart.df["close"].iloc[i]
    assert chart.snap_view_point(far)[1] is None


def test_click_places_snapped_point(chart):
    chart.magnet_btn.setChecked(True)
    chart.tools["hline"].setChecked(True)
    i = 100
    dy = chart.price.vb.viewPixelSize()[1]
    chart._click(chart.price.vb.mapViewToScene(view_pt(chart, i, 2 * dy)))
    assert chart.price_at(chart.drawings[-1].value()) == pytest.approx(chart.df["close"].iloc[i])


def test_handle_drag_snaps(chart):
    chart.magnet_btn.setChecked(True)
    x = chart._x
    roi = pg.LineSegmentROI([[x[10], 100], [x[100], 130]])
    chart._add_drawing(roi, "trend")
    roi.sigRegionChangeStarted.emit(roi)
    h = roi.getHandles()[1]
    dy = chart.price.vb.viewPixelSize()[1]
    roi.movePoint(h, view_pt(chart, 120, 2 * dy), finish=False)
    end = roi.mapToParent(h.pos())
    assert end.x() == pytest.approx(x[120]) and end.y() == pytest.approx(chart.df["close"].iloc[120])
