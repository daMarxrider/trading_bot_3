# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Portfolio backtest tab (#148)."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from tests.conftest import make_ohlc  # noqa: E402
from tradebot.gui.portfolio_backtest import PortfolioBacktestView  # noqa: E402
from tradebot.gui.theme import ThemeManager  # noqa: E402


class FakeEngine:
    def get_ohlc(self, symbol, tf, bars=None):
        if symbol == "BAD":
            raise KeyError(symbol)
        return make_ohlc(300, seed={"A": 1, "B": 2, "C": 3}[symbol])


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def view(app, tmp_path, monkeypatch):
    # run the work at once on the GUI thread
    def now(fn, done, failed=None):
        try:
            result = fn()
        except Exception as e:  # noqa: BLE001
            failed(str(e))
            return
        done(result)

    monkeypatch.setattr("tradebot.gui.portfolio_backtest.run_async", now)
    themes = ThemeManager(user_dir=tmp_path / "themes")
    themes.apply("dark")
    v = PortfolioBacktestView(FakeEngine(), themes, ["A", "B", "C", "BAD"])
    yield v
    v.close()
    v.deleteLater()


def test_runs_and_shows_portfolio_and_basket(view):
    messages = []
    view.status.connect(messages.append)
    view.max_positions.setValue(2)
    view.allocation.setCurrentText("volatility")
    view.run()
    assert view.result is not None and view.result.settings["max_positions"] == 2
    assert list(view.metrics_model.df.columns) == ["metric", "portfolio", "buy & hold basket"]
    assert list(view.symbols_model.df["symbol"]) == ["A", "B", "C"]
    assert "skipped" in view.summary.text() and "BAD" in view.summary.text()
    assert len(view.plot.getPlotItem().listDataItems()) == 2
    assert "trades" in messages[-1]


def test_reports_missing_symbols(view):
    messages = []
    view.status.connect(messages.append)
    view.symbols.setText("BAD")
    view.run()
    assert view.result is None and "none of the symbols" in messages[-1]
