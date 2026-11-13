# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Chart drawings stored per symbol, Fibonacci, notes, vertical line, measure, edit/delete/colour (#160)."""

import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

import pyqtgraph as pg  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from tradebot.config import Settings  # noqa: E402
from tradebot.gui.chart import ChartView  # noqa: E402
from tradebot.gui.drawings import FibItem, MeasureItem, NoteItem  # noqa: E402
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


def make(themes, path, symbol="AAA"):
    c = ChartView(None, themes, ["TEST"], Settings(path))
    c.resize(1000, 700)
    c.df = candles()
    c.show()
    c.redraw(autorange=True)
    c._switch_drawings(symbol)
    c._loaded_symbol = symbol
    QApplication.processEvents()
    return c


@pytest.fixture
def chart(themes, tmp_path):
    c = make(themes, tmp_path / "s.json")
    yield c
    c.close()
    c.deleteLater()


def add_all(chart):
    x = chart._x
    chart._add_level(150)
    chart._add_vline(x[50])
    chart._add_note(x[20], 120, "support?")
    chart._add_drawing(chart._two_point("fib", [x[10], 100], [x[100], 180], "#ff8800"), "fib")
    chart._add_drawing(chart._two_point("measure", [x[10], 100], [x[60], 150], "#00ff00"), "measure")
    chart._add_drawing(pg.LineSegmentROI([[x[0], 100], [x[100], 130]]), "trend")
    chart._add_drawing(pg.RectROI([x[30], 110], [86400 * 5, 20]), "rect")


def test_tools_have_tips(chart):
    for key in ("fib", "vline", "text", "measure"):
        assert chart.tools[key].toolTip()


def test_fib_levels_and_measure_label(chart):
    add_all(chart)
    fib = next(d for d in chart.drawings if isinstance(d, FibItem))
    assert [t.toPlainText() for t in fib._texts[:3]] == ["0  (100)", "0.236  (118.88)", "0.382  (130.56)"]
    assert fib._texts[6].toPlainText() == "1  (180)"
    assert fib.shape().boundingRect().height() > 0
    m = next(d for d in chart.drawings if isinstance(d, MeasureItem))
    assert m._texts[0].toPlainText().startswith("+50 (+50.00%) · 50d")


def test_fib_is_log_aware(chart):
    add_all(chart)
    fib = next(d for d in chart.drawings if isinstance(d, FibItem))
    chart.log_btn.setChecked(True)
    assert fib.prices() == pytest.approx((100, 180))
    assert fib._texts[3].toPlainText() == "0.5  (140)"
    note = next(d for d in chart.drawings if isinstance(d, NoteItem))
    assert note.pos().y() == pytest.approx(np.log10(120))
    vline = next(d for d in chart.drawings if d._kind == "vline")
    assert vline.value() == pytest.approx(chart._x[50])  # a time, untouched


def test_drawings_saved_per_symbol_and_restored(themes, tmp_path):
    path = tmp_path / "s.json"
    chart = make(themes, path)
    add_all(chart)
    specs = chart.drawing_specs()
    assert len(specs) == 7
    chart._remove_drawings()
    chart._switch_drawings("BBB")  # another symbol: none of AAA's
    chart._loaded_symbol = "BBB"
    assert chart.drawings == []
    chart._add_level(111)
    chart._switch_drawings("AAA")
    chart._loaded_symbol = "AAA"
    again = chart.drawing_specs()
    assert [s["kind"] for s in again] == [s["kind"] for s in specs]
    assert again[0]["price"] == pytest.approx(150) and again[2]["text"] == "support?"
    assert again[3]["color"] == "#ff8800"
    chart.close()
    chart.deleteLater()
    fresh = make(themes, path)  # across app restarts, from the settings file
    assert len(fresh.drawings) == 7
    fresh._switch_drawings("BBB")
    fresh._loaded_symbol = "BBB"
    assert [s["price"] for s in fresh.drawing_specs()] == pytest.approx([111])
    fresh.close()
    fresh.deleteLater()


def test_restored_on_log_scale_and_broken_entries_skipped(themes, tmp_path):
    path = tmp_path / "s.json"
    s = Settings(path)
    s.set("chart.drawings", {"AAA": [{"kind": "bogus"}, {"kind": "hline"}, {"kind": "hline", "price": 150}]})
    chart = make(themes, path)
    assert len(chart.drawings) == 1 and chart.drawings[0].value() == 150
    chart.close()
    chart.deleteLater()


def test_delete_recolour_edit_persist(chart):
    add_all(chart)
    stored = lambda: chart.settings.get("chart.drawings")["AAA"]  # noqa: E731
    level = chart.drawings[0]
    chart.set_drawing_color(level, "#123456")
    assert stored()[0]["color"] == "#123456"
    note = next(d for d in chart.drawings if isinstance(d, NoteItem))
    chart.set_drawing_color(note, "#abcdef")
    assert next(s for s in stored() if s["kind"] == "text")["color"] == "#abcdef"
    chart._ask_text = lambda current: "new text"
    chart._edit_note(note)
    assert next(s for s in stored() if s["kind"] == "text")["text"] == "new text"
    chart.delete_drawing(level)
    assert level not in chart.drawings and len(stored()) == 6
    chart.clear_drawings()
    assert "AAA" not in chart.settings.get("chart.drawings")


def test_dragging_saves(chart):
    chart._add_level(150)
    chart.drawings[0].setValue(160)
    chart.drawings[0].sigPositionChangeFinished.emit(chart.drawings[0])
    assert chart.settings.get("chart.drawings")["AAA"][0]["price"] == pytest.approx(160)


def test_drawing_at_and_context_menu(chart):
    chart._add_level(150)
    pt = chart.price.vb.mapViewToScene(pg.Point(float(chart._x[100]), 150.0))
    assert chart.drawing_at(pt) is chart.drawings[0]
    far = chart.price.vb.mapViewToScene(pg.Point(float(chart._x[100]), 190.0))
    assert chart.drawing_at(far) is None
    chart._context_menu(pt, chart.mapToGlobal(chart.rect().center()))
    labels = [a.text() for a in chart._menu.actions()]
    assert "Delete drawing" in labels and "Change colour …" in labels
    next(a for a in chart._menu.actions() if a.text() == "Delete drawing").trigger()
    assert chart.drawings == []
    chart._menu.close()


def test_tools_place_drawings(chart):
    vp = chart.canvas.viewport()
    scene = chart.canvas.mapToScene(vp.rect().center())
    chart.tools["vline"].setChecked(True)
    chart._click(scene)
    chart._ask_text = lambda current: "hello"
    chart.tools["text"].setChecked(True)
    chart._click(scene)
    for tool in ("fib", "measure"):
        chart.tools[tool].setChecked(True)
        chart._click(scene)
        chart._click(chart.canvas.mapToScene(vp.rect().center() + vp.rect().topLeft() + pg.QtCore.QPoint(40, 40)))
    assert [d._kind for d in chart.drawings] == ["vline", "text", "fib", "measure"]
