# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Symbol search for autocompletion (no network: fake sources)."""

import sys
import types

import pytest

from tradebot.symbols import CryptoPairs, SymbolMatch, SymbolSearch, rank, yahoo_search

APPLE = SymbolMatch("AAPL", "Apple Inc.", "NASDAQ", "stock")
HOSPITALITY = SymbolMatch("APLE", "Apple Hospitality REIT, Inc.", "NYSE", "stock")
SOL = SymbolMatch("SOL/USDT", "SOL / USDT", "binance", "crypto")


def test_rank_orders_exact_prefix_name_contains():
    assert rank("aapl", APPLE) == 0
    assert rank("sol", SOL) == 0  # the base of a pair counts as the symbol
    assert rank("aa", APPLE) == 1
    assert rank("apple", APPLE) == 2 and rank("hosp", HOSPITALITY) == 2  # any word of the name
    assert rank("pl", APPLE) == 3
    assert rank("xyz", APPLE) is None and rank("  ", APPLE) is None


def test_label():
    assert APPLE.label == "AAPL — Apple Inc. · NASDAQ · stock"
    assert SymbolMatch("SAP.DE", kind="watchlist").label == "SAP.DE · watchlist"


def test_local_first_ranked_and_deduplicated():
    remote = [HOSPITALITY, APPLE, SymbolMatch("AAPL.TO", "Apple Inc.", "Toronto", "stock")]
    s = SymbolSearch(lambda: ["MSFT", "AAPL", "AAPL"], remote=[lambda q: remote])
    out = [m.symbol for m in s.search("aapl")]
    assert out == ["AAPL", "AAPL.TO"]  # watchlist AAPL once, the remote duplicate dropped
    assert s.search("aapl")[0].kind == "watchlist"
    assert [m.symbol for m in s.search("apple")] == ["APLE", "AAPL", "AAPL.TO"]  # name matches, source order


def test_failing_source_falls_back_to_local():
    def offline(query):
        raise ConnectionError("no network")

    s = SymbolSearch(lambda: ["SAP.DE"], remote=[offline, lambda q: [SOL]])
    assert [m.symbol for m in s.search("sa")] == ["SAP.DE"]
    assert [m.symbol for m in s.search("sol")] == ["SOL/USDT"]


def test_limit():
    many = [SymbolMatch(f"AB{i}") for i in range(50)]
    assert len(SymbolSearch(remote=[lambda q: many], limit=7).search("ab")) == 7


def test_yahoo_search_filters_types_and_maps_crypto(monkeypatch):
    quotes = [
        {"symbol": "AAPL", "longname": "Apple Inc.", "exchDisp": "NASDAQ", "quoteType": "EQUITY"},
        {"symbol": "SAAPL=F", "shortname": "Apple futures", "exchDisp": "CME", "quoteType": "FUTURE"},
        {"symbol": "SOL-USD", "shortname": "Solana USD", "exchDisp": "CCC", "quoteType": "CRYPTOCURRENCY"},
    ]
    fake = types.SimpleNamespace(Search=lambda query, max_results, news_count: types.SimpleNamespace(quotes=quotes))
    monkeypatch.setitem(sys.modules, "yfinance", fake)
    assert yahoo_search("apple") == [APPLE, SymbolMatch("SOL/USD", "Solana USD", "CCC", "crypto")]


def test_crypto_pairs_fall_back_sort_and_load_once(monkeypatch):
    loads = []

    class Down:
        def load_markets(self):
            raise ConnectionError("geo-blocked")

    class Up:
        def load_markets(self):
            loads.append(1)
            return {
                "SOL/BTC": {"symbol": "SOL/BTC", "base": "SOL", "quote": "BTC", "spot": True},
                "SOL/USDT": {"symbol": "SOL/USDT", "base": "SOL", "quote": "USDT", "spot": True},
                "SOL/USDT:USDT": {"symbol": "SOL/USDT:USDT", "base": "SOL", "quote": "USDT", "spot": False},
                "OLD/USDT": {"symbol": "OLD/USDT", "base": "OLD", "quote": "USDT", "spot": True, "active": False},
                "SOL/TRY": {"symbol": "SOL/TRY", "base": "SOL", "quote": "TRY", "spot": True},
            }

    monkeypatch.setitem(sys.modules, "ccxt", types.SimpleNamespace(first=Down, second=Up))
    pairs = CryptoPairs(["first", "second"])
    assert [m.symbol for m in pairs.load()] == ["SOL/USDT", "SOL/BTC"]  # spot, active, common quotes; USDT first
    pairs.load()
    assert loads == [1]  # cached for the session


# -- the completer in the GUI -----------------------------------------------------------------------
@pytest.fixture
def qt_app():
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PyQt6")
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


QApplication = None  # marks this as a GUI module for the conftest's GC handling


def wait_until(app, condition, timeout=3.0):
    import time

    from PyQt6.QtCore import QThreadPool

    end = time.monotonic() + timeout
    while time.monotonic() < end and not condition():
        QThreadPool.globalInstance().waitForDone(20)
        app.processEvents()
    return condition()


def test_completer_shows_local_then_remote_and_inserts_symbol(qt_app, tmp_path):
    from PyQt6.QtTest import QTest

    from tradebot.gui.chart import ChartView
    from tradebot.gui.theme import ThemeManager

    themes = ThemeManager(user_dir=tmp_path)
    themes.apply("dark")
    queries = []

    def remote(query):
        queries.append(query)
        return [APPLE, HOSPITALITY]

    chart = ChartView(None, themes, ["AAPL"], symbol_search=SymbolSearch(lambda: ["AAPL"], remote=[remote]))
    loads = []
    chart.load = lambda: loads.append(chart.symbol.text())
    chart.show()
    chart.symbol.setFocus()
    chart.symbol.clear()
    QTest.keyClicks(chart.symbol, "ap")
    model = chart.symbols.model
    assert [model.item(i).text() for i in range(model.rowCount())] == ["AAPL · watchlist"]  # instant, local
    QTest.keyClicks(chart.symbol, "ple")
    assert wait_until(qt_app, lambda: model.rowCount() == 2)
    assert queries == ["apple"]  # debounced: one search after the pause, not one per key
    assert model.item(1).text() == "APLE — Apple Hospitality REIT, Inc. · NYSE · stock"  # equal rank: source order

    chart.symbols.completer.activated[str].emit(model.item(0).data(chart.symbols.SYMBOL_ROLE))
    assert chart.symbol.text() == "AAPL"  # the plain symbol, not the label
    assert loads == ["AAPL"]  # choosing a suggestion loads the chart


def test_stale_remote_results_are_dropped(qt_app, tmp_path):
    from PyQt6.QtWidgets import QLineEdit

    from tradebot.gui.common import SymbolCompleter

    edit = QLineEdit()
    edit.show()
    edit.setFocus()
    completer = SymbolCompleter(edit, SymbolSearch(remote=[]))
    edit.setText("appl")
    completer._pending = "appl"
    completer._remote_done("app", [APPLE])  # an older search finished after the user typed on
    assert completer.model.rowCount() == 0
    completer._remote_done("appl", [APPLE])
    assert completer.model.rowCount() == 1
    completer.model.clear()
    edit.editingFinished.emit()  # Enter pressed / field left before the search finished
    completer._remote_done("appl", [APPLE])
    assert completer.model.rowCount() == 0


def test_symbol_name_for_stocks_and_crypto_pairs():
    from tradebot.symbols import symbol_name

    names = {"AAPL": [APPLE, HOSPITALITY], "BTC-USD": [SymbolMatch("BTC/USD", "Bitcoin USD", "CCC", "crypto")],
             "ETH-USD": [SymbolMatch("ETH/USD", "Ethereum USD", "CCC", "crypto")]}
    search = lambda q: names.get(q, [])  # noqa: E731
    assert symbol_name("AAPL", search) == "Apple Inc."  # the exact ticker, not the first result
    assert symbol_name("BTC/USDT", search) == "Bitcoin / TetherUS"  # common quotes from the table
    assert symbol_name("BTC/ETH", search) == "Bitcoin / Ethereum"
    assert symbol_name("BTC/XYZ", search) == "Bitcoin / XYZ"  # unknown quote: its ticker
    assert symbol_name("NOPE", search) is None and symbol_name("NOPE/USDT", search) is None

    def offline(q):
        raise ConnectionError

    assert symbol_name("MSFT", offline) is None  # no name, no error
