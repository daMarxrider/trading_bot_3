# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""The symbol dialog (#220): categories, lists, ratios, prices, and how it opens."""

import os

import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")

from PyQt6.QtCore import QEvent, QPointF, Qt  # noqa: E402
from PyQt6.QtGui import QKeyEvent, QMouseEvent  # noqa: E402
from PyQt6.QtWidgets import QApplication, QLineEdit, QMainWindow  # noqa: E402

import tradebot.gui.symbol_dialog as sd  # noqa: E402
from tradebot.config import Settings  # noqa: E402
from tradebot.gui.common import SymbolCompleter  # noqa: E402
from tradebot.gui.symbol_dialog import DAY, DIVIDE, SymbolContext, SymbolDialog, can_divide  # noqa: E402
from tradebot.gui.theme import ThemeManager  # noqa: E402
from tradebot.symbols import SymbolMatch, SymbolSearch, category, rank  # noqa: E402

REMOTE = [
    SymbolMatch("ETH/USDT", "ETH / USDT", "binance", "crypto"),
    SymbolMatch("ETH/BTC", "ETH / BTC", "binance", "crypto"),
    SymbolMatch("ETHA", "iShares Ethereum Trust ETF", "NasdaqGM", "ETF"),
    SymbolMatch("ETD", "Ethan Allen Interiors", "NYSE", "stock"),
    SymbolMatch("ETH=F", "Ether Futures", "CME", "future"),
    SymbolMatch("BTC/USD", "Bitcoin USD", "CCC", "crypto"),
    SymbolMatch("AAPL", "Apple Inc.", "NASDAQ", "stock"),
]
LISTS = {"Main": ["BTC/USDT", "AAPL"], "Dividends": ["KO", "PEP"]}


class Engine:
    def __init__(self):
        self.calls = []

    def peek(self, symbol, tf, bars=None):
        return None

    def get_ohlc(self, symbol, tf, bars=None, **kw):
        self.calls.append(symbol)
        if symbol == "BAD":
            raise LookupError("unknown")
        return pd.DataFrame({"close": [100.0, 110.0]}, index=pd.date_range("2026-11-25", periods=2, tz="UTC"))


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(app, tmp_path, monkeypatch):
    monkeypatch.setattr(sd, "run_async", lambda fn, done, fail=None: done(fn()))
    themes = ThemeManager(user_dir=tmp_path)
    themes.apply("graphite")
    settings = Settings(tmp_path / "settings.json")
    settings.set(sd.RECENT_KEY, ["ETH/USDT", "^GDAXI"])
    win = QMainWindow()
    win.resize(1200, 800)
    win.symbol_context = SymbolContext(Engine(), settings, themes, lambda: LISTS)
    win.search = SymbolSearch(lambda: ["BTC/USDT", "AAPL", "KO", "PEP"], remote=[lambda q: REMOTE])
    win.show()
    yield win
    win.close()
    win.deleteLater()
    QApplication.processEvents()


def dialog(window, initial="", multi=False) -> SymbolDialog:
    d = SymbolDialog(window.search, window, initial=initial, multi=multi)
    d.show()
    if initial:
        d._search_remote()
    QApplication.processEvents()
    return d


def typed(d: SymbolDialog, text: str):
    d.query.setText(text)
    d._edited(text)
    d._search_remote()


def key(d: SymbolDialog, k, text=""):
    d._key(QKeyEvent(QEvent.Type.KeyPress, k, Qt.KeyboardModifier.NoModifier, text))


def shown(d: SymbolDialog) -> list[str]:
    return [m.symbol for m in d.model.rows]


def test_categories():
    assert category(SymbolMatch("SPY", kind="ETF")) == "stocks"  # ETFs and funds count as stocks
    assert category(SymbolMatch("VTSAX", kind="fund")) == "stocks"
    assert category(SymbolMatch("HYG", "iShares High Yield Corporate Bond ETF", kind="ETF")) == "stocks"
    assert category(SymbolMatch("^TNX", "Treasury Yield 10 Years", kind="index")) == "bonds"
    assert category(SymbolMatch("^GDAXI", kind="index")) == "indices"
    assert category(SymbolMatch("EURUSD=X", kind="forex")) == "forex"
    assert category(SymbolMatch("GC=F", kind="future")) == "futures"
    # a watchlist symbol has no kind: its spelling decides
    assert [category(SymbolMatch(s, kind="watchlist")) for s in ("BTC/USDT", "EURUSD=X", "GC=F", "^N225", "^TYX", "SAP.DE")] \
        == ["crypto", "forex", "futures", "indices", "bonds", "stocks"]


def test_rank_ignores_the_pair_slash():
    assert rank("btcusd", SymbolMatch("BTC/USD")) == 0  # typed the Scalable way
    assert rank("btcus", SymbolMatch("BTC/USDT")) == 1


def test_can_divide():
    assert can_divide("BTC/USDT", "ETH/USDT") and can_divide("KO", "PEP") and can_divide("EURUSD=X", "GC=F")
    assert not can_divide("BTC/USDT", "AAPL")  # mixed ratios need #219
    assert not can_divide("KO", "KO") and not can_divide("BTC/USDT", "BTC/USDT:USDT")


def test_empty_query_shows_recent_then_own_symbols(window):
    d = dialog(window)
    assert shown(d) == ["ETH/USDT", "^GDAXI", "BTC/USDT", "AAPL", "KO", "PEP"]
    d.close()


def test_search_filters_by_category_with_counts(window):
    d = dialog(window, "eth")
    assert shown(d) == ["ETH/USDT", "ETH/BTC", "ETHA", "ETH=F", "ETD"]  # recent ETH/USDT first, then by rank
    assert d.model.rows[0].name == "ETH / USDT"  # the user's own symbol gets the source's details
    assert d._counts["cat:all"].text() == "5" and d._counts["cat:stocks"].text() == "2"
    d.set_source("cat:stocks")
    assert shown(d) == ["ETHA", "ETD"]  # the ETF sits with the stocks
    key(d, Qt.Key.Key_Tab)
    assert d.source == "cat:forex" and shown(d) == [] and "No symbol matches “eth”" in d.empty.text()
    key(d, Qt.Key.Key_Backtab)
    key(d, Qt.Key.Key_Backtab)
    assert d.source == "cat:all"
    d.close()


def test_lists_in_the_rail(window):
    d = dialog(window)
    d.set_source("list:Dividends")
    assert shown(d) == ["KO", "PEP"]
    typed(d, "pe")
    assert shown(d) == ["PEP"]  # the query filters within the list
    d.set_source("list:")
    typed(d, "")
    assert shown(d) == ["ETH/USDT", "^GDAXI"]
    d.close()


def test_enter_picks_and_remembers(window):
    d = dialog(window, "eth")
    picked = []
    d.picked.connect(picked.append)
    key(d, Qt.Key.Key_Down)
    key(d, Qt.Key.Key_Return)
    assert picked == ["ETH/BTC"] and not d.isVisible()
    assert window.symbol_context.recent()[0] == "ETH/BTC"


def test_ratio_with_slash_and_backspace(window):
    d = dialog(window, "btcusd")
    picked = []
    d.picked.connect(picked.append)
    assert shown(d)[0] == "BTC/USD"
    key(d, Qt.Key.Key_Slash, "/")  # "BTCUSD/" names the highlighted row: divide it
    assert d.numerator == "BTC/USD" and d.token.isVisible() and d.query.text() == ""
    typed(d, "a")
    assert "AAPL" not in shown(d)  # a crypto pair can't be divided by a stock yet
    typed(d, "")
    key(d, Qt.Key.Key_Backspace)
    assert d.numerator is None and d.query.text() == "BTC/USD"
    key(d, Qt.Key.Key_Slash, "/")
    typed(d, "eth")
    key(d, Qt.Key.Key_Return)
    assert picked == ["BTC/USD:ETH/USDT"]  # the engine's spread symbol (#117)


def test_slash_inside_a_pair_is_typed(window):
    d = dialog(window, "eth")
    key(d, Qt.Key.Key_Slash, "/")  # "eth" isn't a whole symbol: the slash belongs to "ETH/…"
    assert d.numerator is None
    d.close()


def test_spaced_slash_and_divide_column(window):
    d = dialog(window)
    typed(d, "ETH/USDT / ")
    assert d.numerator == "ETH/USDT"
    d.clear_ratio()
    typed(d, "eth")
    d._clicked(d.model.index(1, DIVIDE))
    assert d.numerator == "ETH/BTC"
    d.close()


def test_multi_stays_open_and_collects(window):
    d = dialog(window, "eth", multi=True)
    picked = []
    d.picked.connect(picked.append)
    key(d, Qt.Key.Key_Return)
    d._clicked(d.model.index(2, 1))
    assert picked == ["ETH/USDT", "ETHA"] and d.isVisible() and d.status.text() == "Added ETH/USDT, ETHA"
    d.close()


def test_prices_load_for_visible_rows(window):
    d = dialog(window, "eth")
    d._load_visible()
    engine = window.symbol_context.engine
    assert set(shown(d)) <= set(engine.calls)
    assert d.model.index(0, DAY).data() == "▲ +10.00%"  # colour and a triangle
    calls = len(engine.calls)
    d._load_visible()
    assert len(engine.calls) == calls  # fresh prices aren't fetched again
    d.close()


def test_completer_opens_the_dialog(window, monkeypatch):
    edit = QLineEdit("AAPL", window)
    comp = SymbolCompleter(edit, window.search)
    chosen = []
    comp.chosen.connect(chosen.append)
    edit.show()
    click = QMouseEvent(QEvent.Type.MouseButtonPress, QPointF(5, 5), QPointF(5, 5), Qt.MouseButton.LeftButton,
                        Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(edit, click)
    assert comp.dialog is not None and comp.dialog.isVisible()
    comp.dialog.pick("KO")
    assert edit.text() == "KO" and chosen == ["KO"]

    multi = QLineEdit("AAPL, MSFT", window)
    comp = SymbolCompleter(multi, window.search, multi=True)
    QApplication.sendEvent(multi, click)
    assert comp.dialog is None  # a click edits the list; the search icon opens the dialog
    comp.dialog_action.trigger()
    comp.dialog.pick("KO")
    comp.dialog.pick("MSFT")
    assert multi.text() == "AAPL, MSFT, KO"
    comp.dialog.close()


def test_typing_over_the_chart_opens_the_dialog(window, monkeypatch, tmp_path):
    pytest.importorskip("pyqtgraph")
    from tradebot.gui.chart import ChartView

    chart = ChartView(None, window.symbol_context.themes, ["AAPL"], Settings(tmp_path / "c.json"),
                      symbol_search=window.search)
    window.setCentralWidget(chart)
    QApplication.processEvents()
    opened = []
    monkeypatch.setattr(chart.symbols, "open_dialog", opened.append)
    monkeypatch.setattr(chart.canvas, "underMouse", lambda: True)
    monkeypatch.setattr(window, "isActiveWindow", lambda: True)
    monkeypatch.setattr(QApplication, "focusWidget", staticmethod(lambda: chart.canvas))  # the chart was clicked

    def press(k, text, mods=Qt.KeyboardModifier.NoModifier):
        return chart._type_to_search(QKeyEvent(QEvent.Type.KeyPress, k, mods, text))

    assert press(Qt.Key.Key_E, "e") and opened == ["E"]
    QApplication.sendEvent(chart.canvas, QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_K, Qt.KeyboardModifier.NoModifier, "k"))
    assert opened == ["E", "K"]  # a real key press on the focused chart goes the same way
    opened.clear()
    opened.append("E")
    assert not press(Qt.Key.Key_E, "e", Qt.KeyboardModifier.ControlModifier)  # shortcuts keep their meaning
    assert not press(Qt.Key.Key_Space, " ")
    monkeypatch.setattr(QApplication, "focusWidget", staticmethod(lambda: chart.symbol))
    assert not press(Qt.Key.Key_E, "e")  # a text field has the keyboard
    assert opened == ["E"]



def test_the_dialog_is_reused_and_reset(window):
    from tradebot.gui.symbol_dialog import open_symbol_dialog

    edit = QLineEdit(window)
    first, second = [], []
    d = open_symbol_dialog(window.search, edit, first.append, "btcusd")
    d._search_remote()
    key(d, Qt.Key.Key_Slash, "/")
    assert d.numerator == "BTC/USD"
    d.reject()
    LISTS["Swing"] = ["NVDA"]
    try:
        again = open_symbol_dialog(window.search, edit, second.append, "k")
        assert again is d and again.numerator is None and not again.token.isVisible() and again.query.text() == "k"
        assert "list:Swing" in again._counts  # a watchlist made meanwhile is in the rail
        QApplication.processEvents()  # the rows come right after the window
        assert shown(again)[0] == "KO"
        key(again, Qt.Key.Key_Return)
        assert first == [] and second == ["KO"]  # only the field it was opened for gets the pick
    finally:
        del LISTS["Swing"]


def test_prices_show_loading_until_they_come(window, monkeypatch):
    pending = []
    monkeypatch.setattr(sd, "run_async", lambda fn, done, fail=None: pending.append((fn, done, fail)))
    d = SymbolDialog(window.search, window, initial="")
    d.show()
    d._load_visible()
    assert d.model.index(0, DAY).data() == "…"
    for fn, done, fail in pending:
        try:
            done(fn())
        except Exception as exc:
            fail and fail(str(exc))
    assert d.model.index(0, DAY).data() == "▲ +10.00%"
    d.close()
