# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Chart loadouts and the remembered chart session (#54)."""

import os

import pytest

from tradebot.algorithms import build_algorithm
from tradebot.config import Settings
from tradebot.indicators.builtin import RSI, Bollinger, StochasticShortWaves
from tradebot.loadouts import LoadoutStore, capture, restore

from .conftest import make_ohlc


def test_round_trip_with_vote_members_and_params():
    algo = build_algorithm("vote", members=["rsi_threshold", "macd_cross"]).with_params(threshold=0.6)
    data = capture([Bollinger(period=30), StochasticShortWaves(period=10)], algo, line=True)
    indicators, signal, line, skipped = restore(data)
    assert [(i.name, i.values) for i in indicators] == [("bbands", {"period": 30, "width": 2.0}),
                                                         ("ssw", {"period": 10, "first_drop": 66, "ath": 83, "lower": 35})]
    assert signal.name == "vote" and [m.name for m in signal.members] == ["rsi_threshold", "macd_cross"]
    assert signal.values == algo.values and line is True and skipped == []


def test_restore_skips_missing_and_repairs_params():
    data = {"indicators": [{"name": "gone_plugin", "params": {}},
                           {"name": "rsi", "params": {"period": 9999, "unknown": 1, "bad": "x"}}],
            "signal": {"name": "vote", "params": {}, "members": ["rsi_threshold", "removed_algo"]}}
    indicators, signal, line, skipped = restore(data)
    assert [i.name for i in indicators] == ["rsi"] and indicators[0].values == {"period": 100}  # clipped to range
    assert signal is None and skipped == ["gone_plugin", "removed_algo"] and line is False


def test_store_in_memory_and_in_settings(tmp_path):
    for store in (LoadoutStore(), LoadoutStore(Settings(tmp_path / "s.json"))):
        store.save("b", {"indicators": []})
        store.save("a", {"indicators": []})
        store.save("b", {"indicators": [{"name": "rsi"}]})  # overwrite keeps the position
        assert list(store.all()) == ["b", "a"] and store.all()["b"]["indicators"] == [{"name": "rsi"}]
        store.delete("b")
        assert list(store.all()) == ["a"]
    assert list(LoadoutStore(Settings(tmp_path / "s.json")).all()) == ["a"]  # persisted


# -- in the chart -------------------------------------------------------------------------------------
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QApplication = None  # marks this as a GUI module for the conftest's GC handling


@pytest.fixture
def gui(tmp_path, monkeypatch):
    pytest.importorskip("PyQt6")
    from PyQt6.QtWidgets import QApplication as App

    from tradebot.gui import chart as chart_module
    from tradebot.gui.theme import ThemeManager

    app = App.instance() or App([])
    themes = ThemeManager(user_dir=tmp_path / "themes")
    themes.apply("dark")
    settings = Settings(tmp_path / "settings.json")
    # applying a loadout must never ask for parameters
    monkeypatch.setattr(chart_module.ParamDialog, "ask", lambda *a, **k: pytest.fail("parameter dialog opened"))

    def make():
        c = chart_module.ChartView(None, themes, ["AAPL"], settings)
        c.resize(1100, 700)
        return c

    return app, make, chart_module, settings


def answer(monkeypatch, module, name=None, confirm=None):
    if name is not None:
        monkeypatch.setattr(module.QInputDialog, "getText", lambda *a, **k: (name, True))
    if confirm is not None:
        yes, no = module.QMessageBox.StandardButton.Yes, module.QMessageBox.StandardButton.No
        monkeypatch.setattr(module.QMessageBox, "question", lambda *a, **k: yes if confirm else no)


def menu_rows(chart):
    from PyQt6.QtWidgets import QPushButton, QToolButton, QWidgetAction

    chart._fill_loadout_menu()
    rows = []
    for action in chart.loadout_menu.actions():
        if isinstance(action, QWidgetAction):
            w = action.defaultWidget()
            rows.append((w.findChild(QPushButton).text(), w.findChild(QToolButton).text()))
        elif action.text():
            rows.append((action.text(), None))
    return rows


@pytest.mark.parametrize("width", [1200, 1700])  # wrapped onto two rows / one row
def test_button_sits_top_right_next_to_clear(gui, width):
    app, make, _, _ = gui
    chart = make()
    chart.resize(width, 700)
    chart.show()
    app.processEvents()
    group = chart.loadout_btn.parentWidget()  # the tools group: … Clear, Loadouts
    lay = group.layout()
    widgets = [lay.itemAt(i).widget() for i in range(lay.count())]
    assert widgets[-2].text() == "✕ Clear" and widgets[-1] is chart.loadout_btn
    right_edge = chart.loadout_btn.mapTo(chart, chart.loadout_btn.rect().topRight()).x()
    assert right_edge >= chart.width() - 30  # flush with the right margin, whether it wrapped or not
    assert group.y() < 120  # in the toolbar at the top


def test_menu_lists_loadouts_with_delete_and_save_last(gui, monkeypatch):
    _, make, module, _ = gui
    chart = make()
    assert menu_rows(chart) == [("No saved loadouts", None), ("Save Current", None)]
    chart.indicators = [RSI()]
    answer(monkeypatch, module, name="Swing")
    chart.save_current_loadout()
    answer(monkeypatch, module, name="Scalp")
    chart.save_current_loadout()
    assert menu_rows(chart) == [("Swing", "✕"), ("Scalp", "✕"), ("Save Current", None)]


def test_save_suggests_a_name_and_asks_before_replacing(gui, monkeypatch):
    _, make, module, _ = gui
    chart = make()
    seen = {}

    def get_text(parent, title, label, echo, text):
        seen["suggestion"] = text
        return "  ", True  # blank name: nothing saved

    monkeypatch.setattr(module.QInputDialog, "getText", get_text)
    chart.save_current_loadout()
    assert seen["suggestion"] == "Loadout 1" and chart.loadouts.all() == {}
    chart.indicators = [RSI()]
    answer(monkeypatch, module, name="A")
    chart.save_current_loadout()
    chart.indicators = [Bollinger()]
    answer(monkeypatch, module, name="A", confirm=False)
    chart.save_current_loadout()
    assert chart.loadouts.all()["A"]["indicators"][0]["name"] == "rsi"  # declined: unchanged
    answer(monkeypatch, module, name="A", confirm=True)
    chart.save_current_loadout()
    assert chart.loadouts.all()["A"]["indicators"][0]["name"] == "bbands"


def test_delete_needs_confirmation(gui, monkeypatch):
    _, make, module, _ = gui
    chart = make()
    chart.loadouts.save("A", capture([RSI()], None, False))
    answer(monkeypatch, module, confirm=False)
    chart.delete_loadout("A")
    assert "A" in chart.loadouts.all()
    answer(monkeypatch, module, confirm=True)
    chart.delete_loadout("A")
    assert chart.loadouts.all() == {}


def test_apply_replaces_indicators_and_signal_without_dialogs(gui):
    app, make, _, _ = gui
    chart = make()
    chart.df = make_ohlc(200)
    chart.indicators = [RSI()]
    chart.redraw()
    chart.loadouts.save("Swing", capture([Bollinger(), StochasticShortWaves()],
                                         build_algorithm("rsi_threshold").with_params(lower=25), True))
    chart.apply_loadout("Swing")
    app.processEvents()
    assert [i.name for i in chart.indicators] == ["bbands", "ssw"] and len(chart.panes) == 1
    assert chart.algorithm.values["lower"] == 25 and chart.algo_box.currentText() == "rsi_threshold"
    assert chart.style_btn.isChecked()


def test_last_session_is_restored_on_the_next_start(gui):
    app, make, _, settings = gui
    first = make()
    first.indicators = [Bollinger(period=30), RSI()]
    first.algorithm = build_algorithm("sma_cross")
    first.tf.blockSignals(True)
    first.tf.setCurrentText("4h")
    first.tf.blockSignals(False)
    first._loaded("BTC/USDT", make_ohlc(200))  # loading the data remembers the setup, saved or not
    session = settings.get("chart.session")
    assert session["symbol"] == "BTC/USDT" and session["tf"] == "4h"
    second = make()  # the next start
    assert second.symbol.text() == "BTC/USDT" and second.tf.currentText() == "4h"
    assert [(i.name, i.values) for i in second.indicators] == [("bbands", {"period": 30, "width": 2.0}),
                                                                ("rsi", {"period": 14})]
    assert second.algorithm.name == "sma_cross" and second.algo_box.currentText() == "sma_cross"
    assert second.loadouts.all() == {}  # the session isn't a loadout


def test_session_with_a_removed_plugin_reports_it_on_first_load(gui):
    _, make, _, settings = gui
    settings.set("chart.session", {"symbol": "AAPL", "tf": "1d", "bars": "500", "line": False, "signal": None,
                                   "indicators": [{"name": "gone_plugin", "params": {}}, {"name": "rsi", "params": {}}]})
    chart = make()
    messages = []
    chart.status.connect(messages.append)
    chart._loaded("AAPL", make_ohlc(100))
    assert [i.name for i in chart.indicators] == ["rsi"]
    assert any("gone_plugin" in m for m in messages)
