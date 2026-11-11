# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""The Scanner page (#111-#114)."""

import os
import threading
import time

import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

from PyQt6.QtCore import QThreadPool  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from tradebot import scanner  # noqa: E402
from tradebot.config import Settings  # noqa: E402
from tradebot.gui import scanner as gs  # noqa: E402
from tradebot.gui.theme import ThemeManager  # noqa: E402
from tradebot.indicators import INDICATORS  # noqa: E402

LISTS = {"Main": ["AAPL", "MSFT"], "Crypto": ["BTC/USDT", "ETH/USDT", "SOL/USDT"]}


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def themes(app, tmp_path):
    t = ThemeManager(user_dir=tmp_path)
    t.apply("dark")
    return t


def wait_until(cond, timeout=5):
    end = time.time() + timeout
    while not cond() and time.time() < end:
        QApplication.processEvents()
        QThreadPool.globalInstance().waitForDone(20)
    QApplication.processEvents()
    assert cond()


def fake_scan(symbols, indicator, params, timeframe, filters, progress=None, cancel=None):
    fake_scan.calls.append((list(symbols), indicator, params, timeframe, filters))
    cls = INDICATORS.get(indicator)
    table = pd.DataFrame({"symbol": symbols, **{o: [1.0] * len(symbols) for o in cls.outputs}})
    for a in cls.alerts:
        table[a] = True
    return scanner.ScanResult(table, len(symbols), len(symbols), {}, 0.1)


fake_scan.calls = []


@pytest.fixture
def make(themes, tmp_path):
    fake_scan.calls.clear()
    lists = dict(LISTS)
    views = []

    def build(settings=None, **kw):
        settings = settings or Settings(tmp_path / "settings.json")
        v = gs.ScannerView(None, themes, lambda: lists, settings, scan_fn=kw.pop("scan_fn", fake_scan),
                           index_fn=kw.pop("index_fn", lambda key: ["A", "B", "C"]))
        views.append(v)
        return v

    build.lists = lists
    return build


def test_highlight_and_pretty():
    assert gs.highlight("S&P 500", "p 5") == "S&amp;<b>P 5</b>00"
    assert gs.highlight("Dow", "DOW") == "<b>Dow</b>" and gs.highlight("a", "") == "a"
    assert gs.pretty("ma_cross") == "MA Cross"


def test_defaults_first_start(make):
    v = make()
    assert v.source == ("watchlist", "Main") and v.cls.name == "ma_cross" and v.timeframe == "1d"
    assert v.source_btn.text() == "⭐ Main · 2 symbols"
    assert v.indicator_btn.text() == "ƒ MA Cross (9, 21, 0)"
    assert list(v.model.df.columns) == ["symbol", "fast", "slow"] and list(v.model.df.symbol) == ["AAPL", "MSFT"]
    assert [v.tf.itemData(i) for i in range(v.tf.count())] == list(gs.TF_LABELS)
    assert [c["alert"] for c in v.chips] == ["cross_up", "cross_down"] and v.active_filters() == []


def test_source_search_and_tabs(make, app):
    v = make()
    dlg = gs.SourceDialog(v.lists(), {"sp500": 503})
    assert dlg.watch.visible_keys() == ["Main", "Crypto"] and "sp500" in dlg.index.visible_keys()
    dlg.watch.search.setText("CRY")
    assert dlg.watch.visible_keys() == ["Crypto"]
    dlg.watch.search.setText("zzz")
    assert "No watchlists match ‘zzz’." in dlg.watch.empty.text() and not dlg.watch.empty.isHidden()
    dlg.index.search.setText("^gspc")  # the ticker matches, and so does part of the title
    assert dlg.index.visible_keys() == ["sp500"]
    dlg.index.search.setText("jones")
    assert dlg.index.visible_keys() == ["dow30"]
    dlg.tabs.setCurrentIndex(1)
    dlg.tabs.setCurrentIndex(0)
    assert dlg.watch.search.text() == "" and dlg.index.search.text() == ""
    dlg.index.chosen.emit("dax40")
    assert dlg.choice == ("index", "dax40")


def test_selecting_an_index_loads_its_symbols_and_is_saved(make, tmp_path):
    v = make()
    v._set_source("index", "sp500")
    wait_until(lambda: v.symbols == ["A", "B", "C"])
    assert v.source_btn.text() == "🇺🇸 S&P 500 · 3" and list(v.model.df.symbol) == ["A", "B", "C"]
    again = make(Settings(tmp_path / "settings.json"))
    assert again.source == ("index", "sp500")


def test_vanished_source_falls_back_to_the_default_list_with_a_message(make, tmp_path):
    s = Settings(tmp_path / "settings.json")
    s.set("scanner.source", {"kind": "watchlist", "key": "Swing"})
    v = make(s)
    assert v.source == ("watchlist", "Main") and v.notice == "Scanner: ‘Swing’ no longer exists, using Main"
    s.set("scanner.source", {"kind": "index", "key": "gone"})
    assert make(s).source == ("watchlist", "Main")


def test_scanned_watchlist_follows_the_sidebar(make):
    v = make()
    make.lists["Main"] = ["AAPL", "MSFT", "NVDA"]
    v.watchlists_changed()
    assert v.symbols == ["AAPL", "MSFT", "NVDA"] and v.source_btn.text().endswith("3 symbols")
    del make.lists["Main"]
    v.watchlists_changed()
    assert v.source == ("watchlist", "Crypto")


def test_indicator_picker_groups_search_and_hidden_footer(app):
    classes, hidden = gs.screenable_split()
    assert "elliott" in hidden and "elliott" not in {c.name for c in classes} and "ma_cross" in {c.name for c in classes}
    dlg = gs.IndicatorDialog(classes, hidden)
    assert "ma_cross" in dlg.picker.visible_keys()
    heads = [dlg.picker.list.item(i).text() for i in range(dlg.picker.list.count())
             if dlg.picker.list.item(i).data(gs.Qt.ItemDataRole.UserRole) is None]
    assert heads[0] == "Built-in"  # no scripts, no community plugins here
    dlg.picker.search.setText("cross")  # the description matches too
    assert "ma_cross" in dlg.picker.visible_keys() and "rsi" not in dlg.picker.visible_keys()
    dlg.picker.search.setText("zzz")
    assert "No indicators match" in dlg.picker.empty.text()


def test_indicator_choice_params_and_filters_are_remembered_per_indicator(make, tmp_path):
    v = make()
    v._set_indicator("rsi")
    assert v.columns == ["value"] and [c["alert"] for c in v.chips] == ["cross_up", "cross_down"]
    v._remember("params", {"period": 7})
    v.chips.append({"output": "value", "op": "<", "value": 30.0})
    v._filters_changed()
    v.set_column("cross_up", True)
    again = make(Settings(tmp_path / "settings.json"))
    assert again.cls.name == "rsi" and again.params == {"period": 7} and again.columns == ["value", "cross_up"]
    assert again.active_filters() == [{"output": "value", "op": "<", "value": 30.0}]
    again._set_indicator("ma_cross")  # back: its own state, not rsi's
    assert again.params["fast"] == 9 and again.active_filters() == []


def test_a_missing_indicator_falls_back_to_ma_cross(make, tmp_path):
    s = Settings(tmp_path / "settings.json")
    s.set("scanner.indicator", "uninstalled")
    assert make(s).cls.name == "ma_cross"
    s.set("scanner.indicator", "elliott")  # not screenable
    assert make(s).cls.name == "ma_cross"


def test_templates_toggle_and_feed_the_scan(make):
    v = make()
    v._chip_clicked(v.chips[0])
    assert v.active_filters() == [{"alert": "cross_up", "state": True}]
    assert v.chip_widgets[0][1].text().startswith("☑")
    v.start_scan()
    wait_until(lambda: v.result is not None)
    assert fake_scan.calls[-1][4] == [{"alert": "cross_up", "state": True}]
    v._remove_chip(v.chips[0])
    assert [c["alert"] for c in v.chips] == ["cross_down"]


def test_filter_dialog_builds_value_and_alert_chips(make, app):
    v = make()
    dlg = gs.FilterDialog(v.cls)
    dlg.output.setCurrentText("fast")
    dlg.op.setCurrentIndex(dlg.op.findData(">="))
    dlg.rhs.setCurrentText("slow")
    assert dlg.chip() == {"output": "fast", "op": ">=", "value": "slow"}
    dlg.rhs.setCurrentText("number")
    dlg.number.setValue(12.5)
    assert dlg.chip()["value"] == 12.5
    dlg.kind.setCurrentIndex(1)
    dlg.state.setCurrentIndex(1)
    assert dlg.chip() == {"alert": "cross_up", "state": False}
    again = gs.FilterDialog(v.cls, {"output": "slow", "op": "!=", "value": 3.0})
    assert again.chip() == {"output": "slow", "op": "!=", "value": 3.0}
    assert v.chip_text({"output": "fast", "op": ">=", "value": "slow"}) == "fast ≥ slow"


def test_columns_are_the_indicators_outputs_and_sort_puts_nan_last(make):
    v = make()
    assert "close" not in v.columns and v.columns == ["fast", "slow"]
    menu = v.column_menu()
    assert [a.text() for a in menu.actions()] == ["fast", "slow", "cross_up (alert)", "cross_down (alert)"]
    v.set_column("cross_up", True)
    assert v.columns == ["fast", "slow", "cross_up"]
    macd = make()
    macd._set_indicator("bbands")
    assert len(gs.plot_outputs(macd.cls)) == 3
    frame = pd.DataFrame({"symbol": ["a", "b", "c"], "fast": [2.0, float("nan"), 1.0]})
    v.model.set_frame(frame)
    v.model.sort(1, gs.Qt.SortOrder.AscendingOrder)
    assert list(v.model.df.symbol) == ["c", "a", "b"]
    v.model.sort(1, gs.Qt.SortOrder.DescendingOrder)
    assert list(v.model.df.symbol) == ["a", "c", "b"]


def test_number_formats_follow_the_unit(app):
    assert gs.formatter(INDICATORS.get("rsi"), "value")(55.123) == "55.1"
    assert gs.formatter(INDICATORS.get("rsi"), "cross_up")(True) == "✓"
    assert gs.formatter(INDICATORS.get("sma"), "value")(1234.5) == "1,234.50"


def test_double_click_charts_with_the_scan_timeframe_and_indicator(make):
    v = make()
    got = []
    v.open_chart.connect(lambda *a: got.append(a))
    v.tf.setCurrentIndex(v.tf.findData("1h"))
    v._row_opened(v.model.index(1, 0))
    assert got == [("MSFT", "1h", "ma_cross", v.params)]


def test_scan_summary_and_stale_prompt_that_undoing_hides(make):
    v = make()
    v.start_scan()
    wait_until(lambda: v.result is not None)
    assert v.summary.text().startswith("2 scanned · 2 match") and not v.summary_bar.isHidden()
    assert not v.scanning and v.scan_btn.isEnabled() and v.notification.isHidden()
    v.tf.setCurrentIndex(v.tf.findData("4h"))  # any input changes the signature
    assert not v.notification.isHidden() and "color" in v.table.styleSheet()
    assert list(v.model.df.symbol) == ["AAPL", "MSFT"]  # marked stale, not cleared
    v.tf.setCurrentIndex(v.tf.findData("1d"))  # undone
    assert v.notification.isHidden() and v.table.styleSheet() == ""
    v.chips[0]["on"] = True
    v._filters_changed()
    assert not v.notification.isHidden()
    v.rescan_btn.click()
    wait_until(lambda: v.notification.isHidden() and not v.scanning)
    assert fake_scan.calls[-1][4] == [{"alert": "cross_up", "state": True}]


def test_changing_source_or_indicator_loads_new_rows_and_offers_a_rescan(make):
    v = make()
    v.start_scan()
    wait_until(lambda: v.result is not None)
    v._set_source("watchlist", "Crypto")
    assert v.result is None and list(v.model.df.symbol) == ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    assert not v.notification.isHidden()
    v._set_indicator("rsi")
    assert list(v.model.df.columns) == ["symbol", "value"]


def test_stop_cancels_and_reports(make):
    started = threading.Event()

    def slow(symbols, indicator, params, timeframe, filters, progress=None, cancel=None):
        started.set()
        for i in range(100):
            if cancel():
                break
            time.sleep(0.01)
        table = pd.DataFrame({"symbol": symbols[:1], "fast": [1.0], "slow": [1.0]})
        return scanner.ScanResult(table, 1, 1, {"X": "no data"}, 0.5, cancelled=True)

    v = make(scan_fn=slow)
    v.start_scan()
    assert started.wait(5) and not v.scan_bar.isHidden() and not v.scan_btn.isEnabled()
    v.stop_btn.click()
    wait_until(lambda: not v.scanning)
    assert v.summary.text().startswith("stopped after 2 of 2") and v.skipped_btn.text() == "1 skipped"


def test_progress_bar_only_after_two_seconds_and_busy_is_refused(make, monkeypatch):
    monkeypatch.setattr(gs, "PROGRESS_AFTER_MS", 50)
    release = threading.Event()

    def slow(symbols, *a, progress=None, cancel=None):
        progress(1, len(symbols))
        release.wait(5)
        return fake_scan(symbols, *a)

    v = make(scan_fn=slow)
    v.start_scan()
    assert v.progress.isHidden()
    wait_until(lambda: not v.progress.isHidden())
    assert v.progress.value() == 1 and v.progress.maximum() == 2
    release.set()
    wait_until(lambda: not v.scanning)

    def busy(*a, **k):
        raise scanner.ScanBusy("another")

    b = make(scan_fn=busy)
    b.start_scan()
    wait_until(lambda: not b.scanning)
    assert b.scan_label.text() == "Another scan is running" and b.result is None


def test_main_window_has_the_scanner_page_after_the_screener(app, themes, tmp_path, monkeypatch):
    from tradebot.gui.app import MainWindow

    monkeypatch.setenv("TRADEBOT_CONFIG", str(tmp_path / "config.toml"))
    win = MainWindow(themes, "dark", Settings(tmp_path / "s.json"))
    names = [b.text() for b in win.findChildren(gs.QPushButton) if b.objectName() == "Nav"]
    assert names.index("📡  Scanner") == names.index("🔎  Screener") + 1
    win.scanner.open_chart.emit("AAPL", "1h", "ma_cross", {"fast": 5})
    assert win.chart.tf.currentText() == "1h" and any(i.name == "ma_cross" for i in win.chart.indicators)
    QThreadPool.globalInstance().waitForDone(3000)
