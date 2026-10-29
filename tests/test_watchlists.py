# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Named watchlists (#67): storage in config.toml with the migration of the old ``watchlist``."""

import pytest

from tradebot.config import (
    DEFAULT_WATCHLIST,
    delete_watchlist,
    load_config,
    save_config_value,
    save_watchlist,
    watchlist,
    watchlists,
)


@pytest.fixture
def config_file(tmp_path, monkeypatch):
    path = tmp_path / "config.toml"
    monkeypatch.setenv("TRADEBOT_CONFIG", str(path))
    return path


OLD = """\
# my config
watchlist = ["AAPL", "BTC/USDT"]   # keep these
theme = "glass"

[brokers.binance]
api_key = "abc"
"""


def test_without_a_config_the_default_is_main(config_file):
    assert watchlists() == {"Main": DEFAULT_WATCHLIST}


def test_the_old_watchlist_is_read_as_main_and_moved_on_the_first_change(config_file):
    config_file.write_text(OLD)
    assert watchlists() == {"Main": ["AAPL", "BTC/USDT"]}
    assert config_file.read_text() == OLD  # reading changes nothing
    assert save_watchlist("Crypto", ["ETH/USDT"])
    text = config_file.read_text()
    assert text == ('# my config\ntheme = "glass"\n\n[brokers.binance]\napi_key = "abc"\n\n'
                    '[watchlists]\nMain = ["AAPL", "BTC/USDT"]\nCrypto = ["ETH/USDT"]\n')
    assert watchlists() == {"Main": ["AAPL", "BTC/USDT"], "Crypto": ["ETH/USDT"]}


def test_all_lists_combined(config_file):
    config_file.write_text('[watchlists]\nMain = ["AAPL", "BTC/USDT"]\nCrypto = ["BTC/USDT", "ETH/USDT"]\n')
    assert watchlist() == ["AAPL", "BTC/USDT", "ETH/USDT"]
    assert watchlist("Crypto") == ["BTC/USDT", "ETH/USDT"]
    with pytest.raises(KeyError, match="Main, Crypto"):
        watchlist("nope")


def test_names_with_spaces_are_quoted_and_rename_keeps_the_place(config_file):
    config_file.write_text('[watchlists]\nMain = ["AAPL"]\nCrypto = ["BTC/USDT"]  # coins\nLast = []\n')
    assert save_watchlist("Long term", ["SAP.DE"])
    assert save_watchlist("My coins", ["BTC/USDT"], rename="Crypto")
    assert list(watchlists()) == ["Main", "My coins", "Last", "Long term"]
    assert '"My coins" = ["BTC/USDT"]  # coins\n' in config_file.read_text()
    assert save_watchlist("My coins", ["BTC/USDT", "ETH/USDT"])  # the quoted key is found again
    assert delete_watchlist("Last")
    assert watchlists() == {"Main": ["AAPL"], "My coins": ["BTC/USDT", "ETH/USDT"], "Long term": ["SAP.DE"]}


def test_hand_written_multi_line_arrays_are_replaced_whole(config_file):
    config_file.write_text('[watchlists]\nMain = [\n  "AAPL",  # apple, [not a bracket]\n  "MS#FT",\n]\n'
                           'Other = [\'X\']\n\n[brokers.x]\nkey = "v"\n')
    assert save_watchlist("Main", ["NVDA"])
    assert config_file.read_text() == ('[watchlists]\nMain = ["NVDA"]\nOther = [\'X\']\n\n'
                                       '[brokers.x]\nkey = "v"\n')
    assert load_config()["brokers"] == {"x": {"key": "v"}}


def test_scalars_keep_working(config_file):
    config_file.write_text('theme = "a # b"  # note\nflag = true\n')
    assert save_config_value("theme", "dark") and save_config_value("flag", False)
    assert config_file.read_text() == 'theme = "dark"  # note\nflag = false\n'


def test_cli_watchlist_option(config_file, monkeypatch):
    from typer.testing import CliRunner

    import tradebot.news
    from tradebot.cli.main import app

    config_file.write_text('[watchlists]\nMain = ["AAPL"]\nCrypto = ["BTC/USDT"]\n')
    asked = []
    monkeypatch.setattr(tradebot.news, "fetch_news", lambda symbols, limit=50: asked.append(symbols) or [])
    runner = CliRunner()
    assert runner.invoke(app, ["news", "--watchlist", "Crypto"]).exit_code == 0
    assert runner.invoke(app, ["news"]).exit_code == 0
    assert asked == [["BTC/USDT"], ["AAPL", "BTC/USDT"]]
    result = runner.invoke(app, ["news", "-w", "nope"])
    assert result.exit_code != 0 and "no watchlist" in result.output


# -- the chart sidebar's panel ------------------------------------------------------------------------
import os  # noqa: E402

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PyQt6.QtWidgets")
pytest.importorskip("pyqtgraph")
QApplication = QtWidgets.QApplication

from PyQt6.QtCore import QModelIndex, QThreadPool  # noqa: E402

from tradebot.config import Settings  # noqa: E402
from tradebot.gui import watchlists as wl  # noqa: E402
from tradebot.gui.chart import ChartView  # noqa: E402
from tradebot.gui.theme import ThemeManager  # noqa: E402

from .conftest import make_ohlc  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def themes(app, tmp_path):
    t = ThemeManager(user_dir=tmp_path / "themes")
    t.apply("dark")
    return t


class FakeEngine:
    """Daily candles: the cache has yesterday's close 100, the network today's 110."""

    def __init__(self):
        self.calls = []

    def peek(self, symbol, timeframe="1d", bars=None, **kw):
        self.calls.append(("peek", symbol))
        return make_ohlc(5).assign(close=[90, 90, 90, 90, 100])

    def get_ohlc(self, symbol, timeframe="1d", bars=None, **kw):
        self.calls.append(("get", symbol))
        return make_ohlc(5).assign(close=[90, 90, 90, 100, 110])


def settle():
    QThreadPool.globalInstance().waitForDone(5000)
    QApplication.processEvents()


def symbols_shown(panel):
    return [panel.list.item(i).data(wl.Qt.ItemDataRole.UserRole) for i in range(panel.list.count())]


@pytest.fixture
def panel(themes, config_file, tmp_path):
    config_file.write_text('[watchlists]\nMain = ["AAPL", "MSFT"]\nCrypto = ["BTC/USDT"]\n')
    p = wl.WatchlistPanel(FakeEngine(), themes, Settings(tmp_path / "s.json"))
    p.resize(340, 600)
    return p


def test_lists_and_the_remembered_selection(panel, tmp_path, themes):
    assert [panel.selector.itemText(i) for i in range(panel.selector.count())] == ["Main", "Crypto"]
    assert symbols_shown(panel) == ["AAPL", "MSFT"]
    panel.select("Crypto")
    assert symbols_shown(panel) == ["BTC/USDT"]
    again = wl.WatchlistPanel(None, themes, Settings(tmp_path / "s.json"))
    assert again.current == "Crypto"


def test_add_remove_reorder_are_saved(panel):
    changed = []
    panel.changed.connect(lambda: changed.append(1))
    panel.field.setText(" NVDA ")
    panel.add_btn.click()
    assert symbols_shown(panel) == ["AAPL", "MSFT", "NVDA"] and panel.field.text() == ""
    panel.add_symbol("AAPL")  # already there: nothing changes
    panel.set_symbol("SAP.DE")
    panel.add_current_btn.click()
    panel.remove_symbol("MSFT")
    panel.move_symbol("SAP.DE", -1)
    assert watchlists()["Main"] == symbols_shown(panel) == ["AAPL", "SAP.DE", "NVDA"]
    assert len(changed) == 4 and panel.all_symbols() == ["AAPL", "SAP.DE", "NVDA", "BTC/USDT"]
    panel.list.model().moveRow(QModelIndex(), 2, QModelIndex(), 0)  # what a drop does
    panel.list.reordered.emit()
    assert watchlists()["Main"] == symbols_shown(panel) == ["NVDA", "AAPL", "SAP.DE"]
    assert all(panel.list.itemWidget(panel.list.item(i)) for i in range(3))  # rows are rebuilt


def test_delete_key_removes_the_current_row(panel):
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QKeyEvent

    panel.list.setCurrentRow(1)
    panel.list.keyPressEvent(QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Delete, Qt.KeyboardModifier.NoModifier))
    assert watchlists()["Main"] == ["AAPL"]


def test_new_rename_delete(panel, monkeypatch):
    answers = iter([("Long term", True), ("Stocks", True), ("Crypto", True)])
    monkeypatch.setattr(wl.QInputDialog, "getText", lambda *a, **k: next(answers))
    monkeypatch.setattr(wl.QMessageBox, "information", lambda *a: None)
    panel.new_list()
    assert panel.current == "Long term" and panel.list.count() == 0 and panel.body.currentWidget() is panel.empty
    panel.select("Main")
    panel.rename_list()
    assert list(watchlists()) == ["Stocks", "Crypto", "Long term"] and panel.current == "Stocks"
    panel.rename_list()  # "Crypto" is taken
    assert list(watchlists()) == ["Stocks", "Crypto", "Long term"]
    monkeypatch.setattr(wl.QMessageBox, "question", lambda *a: wl.QMessageBox.StandardButton.No)
    panel.delete_list()
    assert "Stocks" in watchlists()
    monkeypatch.setattr(wl.QMessageBox, "question", lambda *a: wl.QMessageBox.StandardButton.Yes)
    panel.delete_list()
    panel.delete_list()
    assert list(watchlists()) == ["Long term"] and not panel.delete_btn.isEnabled()
    panel.delete_list()  # the last list stays
    assert list(watchlists()) == ["Long term"]


def test_the_old_watchlist_shows_as_main_and_is_migrated(themes, config_file):
    config_file.write_text('watchlist = ["AAPL"]\n')
    p = wl.WatchlistPanel(None, themes)
    assert p.current == "Main" and symbols_shown(p) == ["AAPL"]
    p.add_symbol("MSFT")
    assert load_config() == {"watchlists": {"Main": ["AAPL", "MSFT"]}}


def test_prices_load_only_while_visible_cache_first(panel, monkeypatch):
    panel.set_visible(False)
    assert panel.engine.calls == []
    panel.set_visible(True)
    assert panel.timer.isActive() and panel.timer.interval() == 60_000
    settle()
    assert ("peek", "AAPL") in panel.engine.calls and ("get", "MSFT") in panel.engine.calls
    last, change, fresh = panel.quotes["AAPL"]
    assert (last, round(change, 3), fresh) == (110, 0.1, True)  # the cached 100 was replaced
    row = panel.list.itemWidget(panel.list.item(0))
    assert "110.00" in row.quote.text() and "+10.00%" in row.quote.text()
    assert panel.themes.current.up in row.quote.text()
    panel._quoted("AAPL", panel.engine.peek("AAPL"), False)  # a late cached price doesn't win
    assert panel.quotes["AAPL"][0] == 110
    panel.set_visible(False)
    assert not panel.timer.isActive()


def test_names_are_shown(panel, monkeypatch):
    monkeypatch.setattr(wl, "symbol_name", lambda s: {"AAPL": "Apple Inc."}.get(s))
    panel.set_visible(True)
    settle()
    assert "Apple Inc." in panel.list.itemWidget(panel.list.item(0)).label.text()


def test_click_charts_the_symbol_and_the_charted_row_is_highlighted(themes, config_file, tmp_path):
    config_file.write_text('[watchlists]\nMain = ["AAPL", "MSFT"]\n')
    chart = ChartView(None, themes, ["TEST"], Settings(tmp_path / "s.json"))
    chart.resize(1200, 700)
    chart.show()
    opened = []
    chart.load = lambda: opened.append(chart.symbol.text())
    panel = chart.watchlists
    panel.list.itemClicked.emit(panel.list.item(1))
    assert opened == ["MSFT"]
    chart._loaded("MSFT", make_ohlc(50))
    assert panel.list.currentRow() == 1 and panel.list.item(1).isSelected()
    chart._loaded("XYZ", make_ohlc(50))
    assert not panel.list.selectedItems()


def test_chart_context_menu_adds_to_a_list(themes, config_file, tmp_path):
    config_file.write_text('[watchlists]\nMain = ["AAPL"]\nCrypto = []\n')
    chart = ChartView(None, themes, ["TEST"], Settings(tmp_path / "s.json"))
    chart.resize(1200, 700)
    chart.show()
    chart._loaded("ETH/USDT", make_ohlc(50))
    chart._context_menu(chart.price.vb.mapViewToScene(chart.price.vb.viewRect().center()), chart.mapToGlobal(chart.rect().center()))
    sub = next(a.menu() for a in chart._menu.actions() if a.text() == "Add to watchlist")
    assert [a.text() for a in sub.actions()] == ["Main", "Crypto"]
    sub.actions()[1].trigger()
    assert watchlists()["Crypto"] == ["ETH/USDT"]


def test_sidebar_category_below_news_and_prices_only_while_shown(themes, config_file, tmp_path):
    config_file.write_text('[watchlists]\nMain = ["AAPL"]\n')
    engine = FakeEngine()
    chart = ChartView(None, themes, ["TEST"], Settings(tmp_path / "s.json"))
    chart.watchlists.engine = engine
    chart.resize(1200, 700)
    chart.show()
    QApplication.processEvents()
    assert chart.sidebar.keys == ["news", "watchlists"]
    settle()
    assert engine.calls == []  # News is the open category
    chart.sidebar.show_category("watchlists")
    settle()
    assert ("get", "AAPL") in engine.calls
    chart.sidebar.toggle.click()  # collapsed
    assert not chart.watchlists.timer.isActive()


@pytest.mark.parametrize("touch", [False, True])
def test_touch_rows_are_finger_sized_and_long_press_opens_the_menu(app, tmp_path, config_file, touch):
    config_file.write_text('[watchlists]\nMain = ["AAPL"]\n')
    themes = ThemeManager(user_dir=tmp_path, touch=touch)
    themes.apply("light")
    p = wl.WatchlistPanel(None, themes)
    p.resize(340, 600)
    p.show()
    QApplication.processEvents()
    row = p.list.itemWidget(p.list.item(0))
    assert row.remove.isVisibleTo(p) == touch  # no hover on a touchscreen: the ✕ is always there
    assert p.list.visualItemRect(p.list.item(0)).height() >= (48 if touch else 30)
    p._row_menu(p.list.viewport().mapToGlobal(p.list.visualItemRect(p.list.item(0)).center()))
    assert [a.text() for a in p._menu.actions() if a.text()] == ["Chart AAPL", "Move up", "Move down", "Remove AAPL"]
    assert p._long_press.on_long_press == p._row_menu


@pytest.mark.parametrize("v, text", [(187.4, "187.40"), (64123.5, "64,123.50"), (0.0873412, "0.08734")])
def test_fmt_price(v, text):
    assert wl.fmt_price(v) == text
