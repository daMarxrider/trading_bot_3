# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""The indicator and signal dialog (#230): one place to add, edit and remove the chart's studies, opened with /."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

from PyQt6.QtCore import QEvent, Qt  # noqa: E402
from PyQt6.QtGui import QKeyEvent  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

import tradebot.gui.chart as chart_mod  # noqa: E402
from tradebot.algorithms import algorithm_names  # noqa: E402
from tradebot.config import Settings  # noqa: E402
from tradebot.gui.chart import ChartView  # noqa: E402
from tradebot.gui.study_dialog import NAME, ON, REMOVE, StudyRow, rank  # noqa: E402
from tradebot.gui.theme import ThemeManager  # noqa: E402
from tradebot.indicators import INDICATORS  # noqa: E402
from tradebot.indicators.base import CATEGORIES, category_of  # noqa: E402
from tradebot.indicators.builtin import RSI  # noqa: E402

from .conftest import make_ohlc  # noqa: E402


@pytest.fixture
def chart(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    asked = []  # (title, values offered) of every parameter dialog; the answer is ``answer`` or the offered values

    def ask(cls, title, space, values=None, parent=None, **kw):
        asked.append((title, dict(values or {})))
        return ask.answer if ask.answer is not None else dict(values or {})

    ask.answer = None
    monkeypatch.setattr(chart_mod.ParamDialog, "ask", classmethod(ask))
    themes = ThemeManager(user_dir=tmp_path)
    themes.apply("dark")
    c = ChartView(None, themes, ["TEST"], Settings(tmp_path / "s.json"))
    c.resize(1200, 800)
    c.show()
    c.df = make_ohlc(300)
    c.redraw()
    c.asked, c.ask = asked, ask
    yield c
    if c._studies is not None:
        c._studies.reject()
    c.close()
    c.deleteLater()
    app.processEvents()


def press(dialog, key, text=""):
    QApplication.sendEvent(dialog.query, QKeyEvent(QEvent.Type.KeyPress, key, Qt.KeyboardModifier.NoModifier, text))


def search(dialog, text):
    dialog.query.setText(text)
    dialog._refresh()


def names(dialog):
    return [r.label or r.name for r in dialog.model.rows]


def test_every_builtin_indicator_has_a_category_and_every_signal_is_listed(chart):
    d = chart.open_studies()
    assert {category_of(c) for c in INDICATORS} <= set(CATEGORIES) - {"script"}  # none falls back to "Scripts"
    rows = d.catalogue()
    assert {r.name for r in rows if r.kind == "indicator"} == {c.name for c in INDICATORS}
    assert [r.name for r in rows if r.kind == "signal"] == algorithm_names()
    assert "cat:script" not in d._counts  # no user scripts: no empty category
    d.set_source("sig:combined")
    assert {"pool", "vote", "combo"} <= set(names(d)) and "sma_cross" not in names(d)


def test_search_ranks_names_before_descriptions():
    rsi = StudyRow("indicator", "rsi", "Relative Strength Index (Wilder)", "oscillator", cls=RSI)
    stoch = StudyRow("indicator", "ssw", "Stochastic Short Waves: smoothed fast and slow RSI", "oscillator", cls=RSI)
    assert rank("rsi", rsi) == 0 and rank("rsi", stoch) == 2
    assert rank("relative strength", rsi) == 2 and rank("strength relative", rsi) == 2  # every word, any order
    assert rank("oscill", rsi) is None  # categories are the rail's job, not the search's
    assert rank("macd", rsi) is None and rank("", rsi) == 0


def test_enter_adds_the_highlighted_indicator_and_the_dialog_stays_open(chart):
    d = chart.open_studies()
    search(d, "rsi")
    assert names(d)[0] == "rsi"
    press(d, Qt.Key.Key_Return)
    assert [i.name for i in chart.indicators] == ["rsi"] and len(chart.panes) == 1
    assert d.isVisible() and "Added rsi 14" in d.status.text()
    assert d.model.rows[0].on_chart and d.model.data(d.model.index(0, ON), Qt.ItemDataRole.DecorationRole) is not None
    assert d.current().key[2] is None  # still on the catalogue row: Enter again adds a second one
    press(d, Qt.Key.Key_Return)
    assert [i.name for i in chart.indicators] == ["rsi", "rsi"]


def test_cancelling_the_parameters_adds_nothing(chart, monkeypatch):
    d = chart.open_studies()
    search(d, "macd")
    monkeypatch.setattr(chart_mod.ParamDialog, "ask", classmethod(lambda cls, *a, **k: None))  # Cancel
    press(d, Qt.Key.Key_Return)
    assert chart.indicators == [] and d.status.text() == ""


def test_a_signal_replaces_the_last_one_and_delete_removes_it(chart):
    d = chart.open_studies()
    search(d, "rsi_threshold")
    press(d, Qt.Key.Key_Return)
    assert chart.algorithm.name == "rsi_threshold"
    search(d, "sma_cross")
    press(d, Qt.Key.Key_Return)
    assert chart.algorithm.name == "sma_cross"
    search(d, "")
    d.set_source("chart")
    assert [r.kind for r in d.model.rows] == ["signal"] and names(d)[0].startswith("sma_cross")
    press(d, Qt.Key.Key_Delete)
    assert chart.algorithm is None and "Removed sma_cross" in d.status.text()
    assert not d.table.isVisibleTo(d) and "Nothing on the chart yet" in d.empty.text()


def test_rows_on_the_chart_come_first_and_edit_in_place(chart):
    first = RSI()
    chart.indicators = [first]
    chart.redraw()
    slot = chart._slots[id(first)]
    d = chart.open_studies()
    assert names(d)[0] == "rsi 14" and d.model.rows[0].live is first  # on the chart, above the catalogue
    chart.ask.answer = {"period": 21}
    press(d, Qt.Key.Key_Return)  # an on-chart row: change its parameters
    (new,) = chart.indicators
    assert new is not first and new.values == {"period": 21} and chart._slots[id(new)] == slot  # same colours
    assert chart.asked[-1] == ("rsi", {"period": 14})  # the dialog offered its current values
    assert names(d)[d.current_row()] == "rsi 21"  # the highlight stays on it


def test_the_remove_column_removes_that_row(chart):
    chart.indicators = [RSI(), RSI(period=7)]
    chart.redraw()
    d = chart.open_studies()
    d._clicked(d.model.index(1, REMOVE))
    assert [chart.study_label(i) for i in chart.indicators] == ["rsi 14"]
    d._clicked(d.model.index(len(chart.indicators), REMOVE))  # a catalogue row has nothing to remove: it adds
    assert len(chart.indicators) == 2


def test_the_dialog_is_reused_and_reset(chart):
    d = chart.open_studies()
    search(d, "macd")
    d.set_source("chart")
    d.reject()
    again = chart.open_studies()
    assert again is d and again.query.text() == "" and again.source == "all" and names(again)


@pytest.mark.parametrize("mods", [Qt.KeyboardModifier.NoModifier,
                                  Qt.KeyboardModifier.ShiftModifier,  # Shift+7 on a German keyboard
                                  Qt.KeyboardModifier.GroupSwitchModifier,  # AltGr on Linux
                                  Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.AltModifier])  # Windows
def test_slash_over_the_chart_opens_the_dialog_on_any_layout(chart, monkeypatch, mods):
    opened, symbols = [], []
    monkeypatch.setattr(chart, "open_studies", lambda initial="": opened.append(1))
    monkeypatch.setattr(chart.symbols, "open_dialog", symbols.append)
    monkeypatch.setattr(chart.canvas, "underMouse", lambda: True)
    monkeypatch.setattr(chart.window(), "isActiveWindow", lambda: True)
    monkeypatch.setattr(QApplication, "focusWidget", staticmethod(lambda: chart.canvas))
    key = Qt.Key.Key_7 if mods == Qt.KeyboardModifier.ShiftModifier else Qt.Key.Key_Slash
    assert chart._type_to_search(QKeyEvent(QEvent.Type.KeyPress, key, mods, "/"))
    assert opened == [1] and symbols == []
    assert not chart._type_to_search(QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_E, Qt.KeyboardModifier.ControlModifier
                                               | Qt.KeyboardModifier.AltModifier, "e"))  # Ctrl+Alt+E stays a shortcut


def test_one_toolbar_button_replaces_the_menu_and_the_signal_box(chart):
    assert not hasattr(chart, "algo_box") and not hasattr(chart, "ind_menu")
    assert "/" in chart.studies_btn.toolTip()
    chart.studies_btn.click()
    assert chart._studies is not None and chart._studies.isVisible()


def test_user_scripts_get_their_own_category(chart, monkeypatch):
    class Mine(RSI):
        name = "mine_test"
        category = ""  # a script that sets none

    INDICATORS.register(Mine)
    try:
        chart._studies = None  # the rail is built with the dialog
        d = chart.open_studies()
    finally:
        INDICATORS.unregister("mine_test")
    d = chart.open_studies()
    INDICATORS.register(Mine)
    try:
        d.set_source("cat:script")
        assert names(d) == ["mine_test"] and d.model.data(d.model.index(0, NAME)) == "mine_test"
    finally:
        INDICATORS.unregister("mine_test")


def test_a_pool_of_strategies_gets_a_section_per_part_that_fits_the_screen(chart):
    from PyQt6.QtWidgets import QCheckBox

    from tradebot.algorithms import build_algorithm
    from tradebot.gui.params import GroupedParamForm, ParamDialog

    pool = build_algorithm("pool")
    values = {**pool.values, "use.donchian": 0}
    d = ParamDialog("pool parameters", pool.param_space(), values, chart)
    assert isinstance(d.form, GroupedParamForm)
    switch = d.form.editors["use.donchian"]
    assert isinstance(switch, QCheckBox) and not switch.isChecked()
    assert not d.form.editors["donchian.period"].isEnabled()  # its settings are dimmed while it is off
    switch.setChecked(True)
    assert d.form.editors["donchian.period"].isEnabled()
    got = d.values()
    assert set(got) == set(pool.param_space()) and got["use.donchian"] == 1  # every parameter, switches as 0/1
    assert got["threshold"] == values["threshold"] and got["donchian.period"] == values["donchian.period"]
    room = chart.screen().availableGeometry()
    assert d.height() <= room.height() * 0.85 + 1  # scrolls instead of running off the screen
    d.deleteLater()


def test_keys_typed_while_a_picker_opens_reach_its_search(chart, monkeypatch):
    d = chart.open_studies()
    monkeypatch.setattr(d, "isActiveWindow", lambda: False)  # the window manager hasn't given it the keyboard yet
    d.query.clear()
    for ch in "macd":
        QApplication.sendEvent(chart.canvas, QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_A, Qt.KeyboardModifier.NoModifier,
                                                       ch))
    assert d.query.text() == "macd" and names(d)[0] == "macd"  # handed over and searched, none lost
    monkeypatch.setattr(d, "isActiveWindow", lambda: True)
    assert not chart_mod.forward_key(d, QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_X, Qt.KeyboardModifier.NoModifier,
                                                  "x"))  # once it has the keyboard, keys go to it directly


def test_a_picker_closes_when_the_main_window_is_clicked_but_not_for_its_own_dialogs(chart, monkeypatch):
    d = chart.open_studies()
    owner = chart.window()
    monkeypatch.setattr(QApplication, "activeWindow", staticmethod(lambda: d))  # e.g. a parameter dialog of its own
    d._dismiss_if_left()
    assert d.isVisible()
    monkeypatch.setattr(QApplication, "activeWindow", staticmethod(lambda: owner))
    d._dismiss_if_left()
    assert not d.isVisible()


def test_drawing_tools_sit_in_a_rail_on_the_chart_not_in_the_toolbar(chart, tmp_path):
    from PyQt6.QtWidgets import QMainWindow

    window = QMainWindow()
    view = ChartView(None, chart.themes, ["TEST"], Settings(tmp_path / "rail.json"))
    window.setCentralWidget(view)
    window.show()
    window.resize(1440, 900)
    QApplication.processEvents()
    try:
        rail = view.tool_rail
        for btn in [*view.tools.values(), view.magnet_btn]:
            assert btn.parent() is rail and btn.width() == btn.height() == 32  # icons; the names are in the tooltips
            assert btn.text().lower() in btn.toolTip().lower()
        assert rail.geometry().right() < view.canvas.geometry().left()  # beside the plot, on the same surface
        groups = (view.symbol.parentWidget(), view.loadout_btn.parentWidget())  # the toolbar: what and how
        assert view.studies_btn.parentWidget() is groups[1]
        assert not any(g.isAncestorOf(b) for g in groups for b in view.tools.values())  # no tool row above the chart
        view.tools["trend"].click()
        assert view._active_tool() == "trend"
    finally:
        window.close()
        window.deleteLater()
        QApplication.processEvents()
