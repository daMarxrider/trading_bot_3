# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""The pool in the backtest lab: candidates, All, limits and the selection card (#21)."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from tradebot.algorithms.pool import PoolEnsemble, default_pool  # noqa: E402
from tradebot.gui.theme import ThemeManager  # noqa: E402
from tradebot.gui.views import BacktestView  # noqa: E402

from .conftest import make_ohlc  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def lab(app, tmp_path):
    themes = ThemeManager(user_dir=tmp_path)
    themes.apply("dark")
    view = BacktestView(None, themes, ["TEST"])
    view.show()
    view.algo.setCurrentText("pool")
    QApplication.processEvents()
    return view


def test_pool_card_replaces_the_vote_members(lab):
    assert lab.pool_card.isVisible() and not lab.members_card.isVisible()
    lab.algo.setCurrentText("vote")
    assert lab.members_card.isVisible() and not lab.pool_card.isVisible()


def test_all_candidates_checked_by_default(lab):
    assert lab.pool.all.isChecked()
    assert list(lab.pool.boxes) == default_pool()[0]
    assert lab.pool.boxes["sma"].text() == "sma · filter" and lab.pool.boxes["sma_cross"].text() == "sma_cross"
    algo = lab._algorithm()
    assert isinstance(algo, PoolEnsemble) and [m.name for m in algo.pool] == default_pool()[0]


def test_custom_pool_and_limits(lab):
    lab.pool.set_selected(["macd_cross", "sma"])
    lab.min_members.setValue(2)
    lab.max_members.setValue(1)  # below the minimum: raised to it
    algo = lab._algorithm()
    assert [m.name for m in algo.pool] == ["macd_cross", "sma"] and not lab.pool.all.isChecked()
    assert algo.min_members == 2 and algo.max_members == 2
    assert "use.sma" in lab.params.values()
    lab.max_members.setValue(0)
    assert lab._algorithm().max_members is None  # "no limit"


def test_last_strategy_stays_checked(lab):
    lab.pool.set_selected(["macd_cross", "sma", "rsi"])
    lab.pool.boxes["macd_cross"].setChecked(False)  # filters alone can't trade
    assert lab.pool.boxes["macd_cross"].isChecked()
    lab.pool.boxes["sma"].setChecked(False)
    assert lab.pool.selected() == ["macd_cross", "rsi"]


def test_selection_card(lab):
    pool = PoolEnsemble(["macd_cross", "sma_cross", "sma", "rsi"])
    base = pool.canonical(pool.values)
    ga = pool.canonical({**base, "use.sma_cross": 0, "use.rsi": 0})
    pso = pool.canonical({**base, "use.sma_cross": 0, "use.sma": 0})
    lab._show_selection(pool, {"GA": ga, "PSO": pso})
    text = lab.selection.text()
    assert lab.selection_card.isVisible() and "agree" in text
    assert "GA uses 2 of 4" in text and "Dropped by both: sma_cross" in text
    lab._show_selection(pool, {"GA": ga})
    assert "agree" not in lab.selection.text() and "Dropped: sma_cross, rsi" in lab.selection.text()
    lab._show_selection(None, {})
    assert not lab.selection_card.isVisible()


def test_optimize_pool_shows_selection(lab, monkeypatch):
    monkeypatch.setattr(lab, "_data", lambda: make_ohlc(500, seed=4))
    lab.pool.set_selected(["macd_cross", "sma_cross", "sma"])
    lab.population.setValue(16)
    lab.generations.setValue(2)
    results = {}
    monkeypatch.setattr("tradebot.gui.views.run_async", lambda work, done, fail: results.update(payload=work()))
    lab.run_optimizer()
    lab._show(results["payload"])
    assert lab.selection_card.isVisible() and "Selection" in lab.selection.text()
