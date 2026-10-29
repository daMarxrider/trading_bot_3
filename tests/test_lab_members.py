# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Choosing the vote ensemble's members in the backtest lab (#78)."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from tradebot.algorithms import ALGORITHMS  # noqa: E402
from tradebot.gui.theme import ThemeManager  # noqa: E402
from tradebot.gui.views import BacktestView  # noqa: E402

DEFAULT = ["rsi_threshold", "macd_cross", "ichimoku_cloud"]


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def lab(app, tmp_path):
    themes = ThemeManager(user_dir=tmp_path)
    themes.apply("dark")
    view = BacktestView(None, themes, ["TEST"])
    view.show()
    view.algo.setCurrentText("vote")
    QApplication.processEvents()
    return view


def members(lab):
    return [m.name for m in lab._algorithm().members]


def test_shown_only_for_the_vote(lab):
    assert lab.members_card.isVisible()
    lab.algo.setCurrentText("sma_cross")
    assert not lab.members_card.isVisible()


def test_every_algorithm_but_the_vote_itself(lab):
    assert list(lab.members.boxes) == [n for n in ALGORITHMS.names() if n != "vote"]


def test_default_members_unchanged(lab):
    assert sorted(members(lab)) == sorted(DEFAULT)
    assert not lab.members.all.isChecked()


def test_all_checks_every_algorithm_and_unchecking_goes_back(lab):
    lab.members.all.click()
    assert members(lab) == list(lab.members.boxes)
    assert {k.split(".")[0] for k in lab.params.values() if "." in k} >= {"sma_cross", "elliott_wave"} & set(
        lab.members.boxes)  # their parameters are in the form
    lab.members.all.click()
    assert sorted(members(lab)) == sorted(DEFAULT)


def test_checking_every_box_by_hand_checks_all(lab):
    for box in lab.members.boxes.values():
        box.setChecked(True)
    assert lab.members.all.isChecked()


def test_choosing_members_updates_the_parameters_and_keeps_set_values(lab):
    key = "rsi_threshold.period"
    lab.params.editors[key].setValue(21)
    lab.members.boxes["macd_cross"].setChecked(False)
    lab.members.boxes["sma_cross"].setChecked(True)
    keys = lab.params.values()
    assert not any(k.startswith("macd_cross.") for k in keys) and any(k.startswith("sma_cross.") for k in keys)
    assert keys[key] == 21
    assert sorted(members(lab)) == sorted(["rsi_threshold", "ichimoku_cloud", "sma_cross"])
    assert lab._algorithm().values[key] == 21


def test_the_last_member_stays_checked(lab):
    for name in DEFAULT:
        lab.members.boxes[name].setChecked(False)
    assert len(members(lab)) == 1


def test_show_on_chart_gets_the_chosen_members(lab):
    from PyQt6.QtWidgets import QPushButton

    lab.members.boxes["sma_cross"].setChecked(True)
    sent = []
    lab.show_on_chart.connect(lambda _symbol, algo: sent.append(algo))
    (button,) = [b for b in lab.findChildren(QPushButton) if b.text() == "Show on chart"]
    button.click()
    assert "sma_cross" in [m.name for m in sent[0].members]
