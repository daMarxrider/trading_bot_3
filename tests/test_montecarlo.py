# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Monte Carlo robustness test (#151): resampling, bands, probabilities, the CLI table and the lab."""

import os

import numpy as np
import pytest
from typer.testing import CliRunner

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tradebot.analysis import montecarlo as mc  # noqa: E402

from .conftest import make_ohlc  # noqa: E402

TRADES = np.array([0.05, -0.02, 0.03, -0.04, 0.06, 0.01, -0.01, 0.02])


@pytest.mark.parametrize("method", mc.METHODS)
def test_resample_shape(method):
    out = mc.resample(TRADES, 50, method, rng=np.random.default_rng(0))
    assert out.shape == (50, len(TRADES))
    assert set(np.unique(out)) <= set(TRADES)


def test_shuffle_keeps_final_return_and_changes_drawdown():
    r = mc.monte_carlo(TRADES, 300, "shuffle", seed=1)
    assert np.allclose(r.final_return, r.actual_return)
    assert r.max_drawdown.min() < r.max_drawdown.max()
    assert r.prob_loss == 0.0


def test_bootstrap_varies_and_is_seeded():
    a, b = mc.monte_carlo(TRADES, 500, seed=2), mc.monte_carlo(TRADES, 500, seed=2)
    assert np.array_equal(a.final_return, b.final_return)
    lo, mid, hi = a.return_band
    assert lo < mid < hi
    assert 0 < a.prob_loss < 1


def test_block_keeps_neighbours():
    r = mc.resample(np.arange(20.0), 30, "block", block_length=5, rng=np.random.default_rng(0))
    assert r.shape == (30, 20)
    assert (np.diff(r[:, :5], axis=1) == 1).all()


def test_probabilities_and_fan():
    r = mc.monte_carlo([-0.1] * 10, 100, dd_threshold=0.2, seed=0)
    assert r.prob_loss == 1.0 and r.prob_drawdown == 1.0
    fan = r.fan()
    assert fan["low"].shape == (11,) and (fan["low"] <= fan["high"]).all()


def test_errors():
    with pytest.raises(ValueError):
        mc.monte_carlo([], 10)
    with pytest.raises(ValueError):
        mc.resample(TRADES, 10, "nope")


def test_from_backtest_sources():
    from tradebot.algorithms import build_algorithm
    from tradebot.backtest import backtest

    res = backtest(make_ohlc(600, seed=4), build_algorithm("sma_cross"))
    t = mc.from_backtest(res, 200, seed=0)
    assert t.items == len(res.trades) and t.source == "trades"
    b = mc.from_backtest(res, 200, source="returns", method="block", seed=0)
    assert b.items == len(res.returns) and b.block_length
    assert abs(t.actual_return - (np.prod(1 + res.trades["return"]) - 1)) < 1e-9


def test_cli_table(monkeypatch):
    from tradebot.cli import main

    monkeypatch.setattr(main, "load_data", lambda *a, **k: make_ohlc(600, seed=4))
    out = CliRunner().invoke(main.app, ["montecarlo", "X", "sma_cross", "--runs", "200", "--seed", "1"])
    assert out.exit_code == 0, out.output
    assert "final return" in out.output and "probability of a loss" in out.output
    bad = CliRunner().invoke(main.app, ["montecarlo", "X", "sma_cross", "--method", "x"])
    assert bad.exit_code != 0


def test_lab_fan(tmp_path, monkeypatch):
    pytest.importorskip("PyQt6")
    pytest.importorskip("pyqtgraph")
    from PyQt6.QtWidgets import QApplication

    from tradebot.gui.theme import ThemeManager
    from tradebot.gui.views import BacktestView

    class Engine:
        def get_ohlc(self, symbol, tf, start=None, end=None, bars=None):
            return make_ohlc(600, seed=4)

    app = QApplication.instance() or QApplication([])
    themes = ThemeManager(user_dir=tmp_path)
    themes.apply("dark")
    view = BacktestView(Engine(), themes, ["AAPL"])
    try:
        monkeypatch.setattr("tradebot.gui.views.run_async", lambda work, done, fail: done(work()))
        view.algo.setCurrentText("sma_cross")
        view.mc_runs.setValue(100)
        view.run_monte_carlo()
        app.processEvents()
        assert len(view.plot.getPlotItem().listDataItems()) > 10
        assert "probability" in view.metrics.df.columns
    finally:
        view.close()
        view.deleteLater()
