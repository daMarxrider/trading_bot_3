# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Walk-forward in the backtest lab (#89): the options, the folds table, the overview plot, a fold's curves."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

import pyqtgraph as pg  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from tradebot.gui.theme import ThemeManager  # noqa: E402
from tradebot.gui.views import BacktestView  # noqa: E402

from .conftest import make_ohlc  # noqa: E402


class Engine:
    def get_ohlc(self, symbol, tf, start=None, end=None, bars=None):
        return make_ohlc(900, seed=sum(map(ord, symbol)))


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def lab(app, tmp_path, monkeypatch):
    themes = ThemeManager(user_dir=tmp_path)
    themes.apply("dark")
    view = BacktestView(Engine(), themes, ["AAPL"])
    view.show()
    view.population.setValue(12)
    view.generations.setValue(2)
    view.algo.setCurrentText("sma_cross")
    monkeypatch.setattr("tradebot.gui.views.run_async", lambda work, done, fail: done(work()))
    QApplication.processEvents()
    return view


def regions(lab):
    return [i for i in lab.plot.getPlotItem().items if isinstance(i, pg.LinearRegionItem)]


def test_options_only_apply_with_walk_forward(lab):
    assert not lab.wf_train.isEnabled() and not lab.holdout.isEnabled() and not lab.wf_anchored.isEnabled()
    lab.wf_folds.setValue(3)
    assert lab.wf_train.isEnabled() and lab.wf_test.isEnabled() and lab.holdout.isEnabled()
    assert "unseen" in lab.wf_folds.toolTip() or "never saw" in lab.wf_folds.toolTip()


def test_walk_forward_one_symbol(lab):
    lab.wf_folds.setValue(3)
    lab.holdout.setValue(100)
    lab.run_optimizer()
    assert lab.folds_card.isVisible() and not lab.per_symbol_card.isVisible()
    assert list(lab.folds.df["fold"]) == ["1", "2", "3", "all", "holdout"]
    assert "Walk-forward" in lab.folds_title.text() and "param" in lab.folds_note.text()
    names = [c[0] for c in lab._curves]
    assert {"walk-forward", "holdout", "buy & hold", "monthly investing"} <= set(names)
    assert len(regions(lab)) == 3  # one shaded test stretch per fold
    walk, hold = (dict((c[0], c) for c in lab._curves)[n] for n in ("walk-forward", "holdout"))
    assert hold[2][0] == pytest.approx(walk[2][-1])  # the holdout continues the walk-forward curve
    assert "walk-forward" in lab.metrics.df.columns and "holdout" in lab.metrics.df.columns

    lab.folds_view.clicked.emit(lab.folds.index(1, 0))  # fold 2: its own in- and out-of-sample curves
    assert {"in-sample", "out-of-sample"} <= {c[0] for c in lab._curves} and not regions(lab)
    wf, _ = lab._wf
    assert lab.params.values()["fast"] == wf.folds[1].params["fast"]
    lab.folds_view.clicked.emit(lab.folds.index(3, 0))  # all: back to the overview
    assert "walk-forward" in {c[0] for c in lab._curves} and len(regions(lab)) == 3

    lab.wf_folds.setValue(0)  # a normal run hides the folds again
    lab.run_optimizer()
    assert not lab.folds_card.isVisible() and not regions(lab)


def test_walk_forward_several_symbols_and_both(lab):
    lab.symbol.setText("AAPL, MSFT")
    lab.method.setCurrentIndex(lab.method.findData("both"))
    lab.wf_folds.setValue(2)
    lab.run_optimizer()
    assert lab.folds_card.isVisible() and lab.per_symbol_card.isVisible()
    assert list(lab.per_symbol.df["symbol"]) == ["AAPL", "MSFT"]
    assert "beats b&h on" in lab.folds.df.columns and list(lab.folds.df["fold"]) == ["1", "2", "all"]
    lab.per_symbol_view.clicked.emit(lab.per_symbol.index(1, 0))
    assert lab._shown_symbol == "MSFT"
    lab.folds_view.clicked.emit(lab.folds.index(0, 0))  # fold 1 of MSFT, with the GA/PSO verdict
    assert lab.verdict_card.isVisible()
