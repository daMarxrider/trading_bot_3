# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Several symbols in the backtest lab (#84): the watchlist button, per-symbol results, completion."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from tradebot.gui.theme import ThemeManager  # noqa: E402
from tradebot.gui.views import BacktestView  # noqa: E402
from tradebot.symbols import SymbolMatch, SymbolSearch  # noqa: E402

from .conftest import make_ohlc  # noqa: E402

LISTS = {"Main": ["AAPL", "MSFT"], "Crypto": ["BTC/USDT", "AAPL"]}


class Engine:
    def get_ohlc(self, symbol, tf, start=None, end=None, bars=None):
        if symbol == "BAD":
            raise LookupError("unknown symbol")
        return make_ohlc(150 if symbol == "NEW" else 500, seed=sum(map(ord, symbol)))


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def lab(app, tmp_path, monkeypatch):
    themes = ThemeManager(user_dir=tmp_path)
    themes.apply("dark")
    view = BacktestView(Engine(), themes, ["AAPL"], watchlists_fn=lambda: LISTS)
    view.show()
    view.population.setValue(12)
    view.generations.setValue(2)
    monkeypatch.setattr("tradebot.gui.views.run_async", lambda work, done, fail: done(work()))
    QApplication.processEvents()
    return view


def test_watchlist_menu_fills_the_symbols(lab):
    lab._fill_lists_menu()
    actions = [a for a in lab.lists_menu.actions() if not a.isSeparator()]
    assert [a.text() for a in actions] == ["Main (2)", "Crypto (2)", "All watchlists (3)"]
    actions[2].trigger()
    assert lab._symbol_list() == ["AAPL", "MSFT", "BTC/USDT"]
    actions[1].trigger()
    assert lab.symbol.text() == "BTC/USDT, AAPL"


def test_symbol_list_parsing(lab):
    lab.symbol.setText(" AAPL,MSFT  SAP.DE, AAPL ,")
    assert lab._symbol_list() == ["AAPL", "MSFT", "SAP.DE"]


def test_single_symbol_has_no_per_symbol_card(lab):
    lab.run_optimizer()
    assert not lab.per_symbol_card.isVisible() and lab.metrics.rowCount() > 0


def test_optimize_several_symbols(lab):
    lab.symbol.setText("AAPL, MSFT, NEW, BAD")
    lab.algo.setCurrentText("sma_cross")
    lab.run_optimizer()
    assert lab.per_symbol_card.isVisible()
    df = lab.per_symbol.df
    assert list(df["symbol"]) == ["AAPL", "MSFT", "mean"]  # NEW skipped (short), BAD failed to load
    assert "Skipped: NEW" in lab.per_symbol_note.text() and "beats buy & hold on" in lab.per_symbol_note.text()
    assert lab._shown_symbol == "AAPL" and lab._chart_symbol() == "AAPL"
    lab.per_symbol_view.clicked.emit(lab.per_symbol.index(1, 0))
    assert lab._shown_symbol == "MSFT" and lab._chart_symbol() == "MSFT"
    lab.per_symbol_view.clicked.emit(lab.per_symbol.index(2, 0))  # the aggregate row: nothing to show
    assert lab._shown_symbol == "MSFT"
    assert "in-sample" in lab.metrics.df.columns and lab._curves


def test_optimize_several_symbols_both_methods(lab):
    lab.symbol.setText("AAPL, MSFT")
    lab.algo.setCurrentText("sma_cross")
    lab.method.setCurrentIndex(lab.method.findData("both"))
    lab.aggregate.setCurrentText("min")
    lab.run_optimizer()
    assert lab.per_symbol_card.isVisible() and lab.verdict_card.isVisible()
    assert list(lab.per_symbol.df["symbol"]) == ["AAPL", "MSFT", "min"]
    assert any(c.startswith("GA") for c in lab.metrics.df.columns)


def test_backtest_several_symbols(lab):
    lab.symbol.setText("AAPL, MSFT")
    lab.run_backtest()
    assert list(lab.per_symbol.df["symbol"]) == ["AAPL", "MSFT"]
    assert "beats buy & hold on" in lab.per_symbol_note.text()
    lab.symbol.setText("AAPL")
    lab.run_backtest()
    assert not lab.per_symbol_card.isVisible()


def test_completion_replaces_only_the_last_symbol(lab):
    lab.symbols.search = SymbolSearch(lambda: [], remote=[lambda q: [SymbolMatch("SOL/USDT", "Solana", "", "crypto")]])
    lab.symbol.setText("AAPL, so")
    assert lab.symbols._query() == "so"
    lab.symbols._chosen("SOL/USDT")
    assert lab.symbol.text() == "AAPL, SOL/USDT"
