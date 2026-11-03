# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Meta-search in the backtest lab (#90): the option, the combinations table, a combination's folds, the winner."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from tradebot.gui.theme import ThemeManager  # noqa: E402
from tradebot.gui.views import BacktestView  # noqa: E402

from .conftest import make_ohlc  # noqa: E402


class Engine:
    def get_ohlc(self, symbol, tf, start=None, end=None, bars=None):
        return make_ohlc(1000, seed=sum(map(ord, symbol)))


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


def test_meta_search_needs_walk_forward_and_is_off_by_default(lab):
    assert lab.meta.value() == 0 and lab.meta.isEnabled() and not lab.meta_compute.isEnabled()
    lab.meta.setValue(3)
    assert lab.meta_compute.isEnabled()
    lab.wf_folds.setValue(0)
    assert not lab.meta.isEnabled() and not lab.meta_compute.isEnabled()
    assert "holdout" in lab.meta.toolTip()


def test_meta_search_shows_combinations_the_winner_and_its_holdout(lab):
    lab.symbol.setText("AAPL, MSFT")
    lab.wf_folds.setValue(2)
    lab.meta.setValue(3)
    lab.run_optimizer()
    res, _, views = lab._meta
    assert lab.meta_card.isVisible() and lab.folds_card.isVisible() and not lab.stop_btn.isVisible()
    table = lab.meta_model.df
    assert len(table) == len(res.trials) == 3 and table["#"].iloc[0].endswith("★")
    assert {"samples", "sample_length", "min_symbols_beating", "sharpe out", "beats b&h in"} <= set(table.columns)
    assert "holdout" in lab.meta_note.text() and "scored once" in lab.meta_note.text()
    # the winner is shown with its holdout
    assert list(lab.folds.df["fold"])[-1] == "holdout" and lab._wf[0] is res.final
    other = next(i for i, label in enumerate(table["#"]) if not label.endswith("★"))
    lab.meta_view.clicked.emit(lab.meta_model.index(other, 0))  # another combination: its folds, no holdout
    assert "holdout" not in list(lab.folds.df["fold"]) and lab._wf[0] is views[int(table["#"].iloc[other].split()[0]) - 1][0]

    lab.samples.setValue(0)
    lab.meta_apply.click()  # the winner's settings into the Optimizer card, the meta-search off
    best = res.best.settings
    assert lab.samples.value() == best["samples"] and lab.min_beating.value() == best["min_symbols_beating"]
    assert lab.meta.value() == 0

    lab.run_optimizer()  # a plain walk-forward hides the meta card
    assert not lab.meta_card.isVisible() and lab.folds_card.isVisible()


def test_stop_ends_the_search_early(lab, monkeypatch):
    lab.wf_folds.setValue(2)
    lab.meta.setValue(5)
    from tradebot.optimize import meta

    real = meta.walk_forward

    def first_only(*args, **kwargs):  # the user presses Stop during the first combination
        lab._stop_meta()
        return real(*args, **kwargs)

    monkeypatch.setattr(meta, "walk_forward", first_only)
    lab.run_optimizer()
    res = lab._meta[0]
    assert res.stopped == "stopped" and len(res.trials) == 1 and res.final is not None
    assert "Stopped early" in lab.meta_note.text()
