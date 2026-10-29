# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""The disk cache of candles and symbol lists (#63)."""

import os
import time

import numpy as np
import pandas as pd
import pytest

from tradebot.config import load_config, os_cache_dir
from tradebot.core.models import AssetClass
from tradebot.data import MarketDataEngine
from tradebot.data import cache as cache_mod
from tradebot.data.cache import DAY, DiskCache, human_size
from tradebot.data.providers import BaseDataProvider

NOW = pd.Timestamp.now(tz="UTC").floor("D")


def daily(n=300, end=NOW, scale=1.0) -> pd.DataFrame:
    idx = pd.date_range(end=end - pd.Timedelta(days=1), periods=n, freq="D", tz="UTC")
    close = np.linspace(100, 200, n) * scale
    return pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99, "close": close,
                         "volume": 1.0}, index=idx)


class Provider(BaseDataProvider):
    """Serves ``df``, records each request; ``down`` makes it fail like an outage."""

    name = "fake"
    asset_classes = frozenset({AssetClass.CRYPTO, AssetClass.STOCK})
    timeframes = frozenset({"1d", "1h"})

    def __init__(self, df):
        self.df, self.calls, self.down = df, [], False

    def fetch(self, symbol, timeframe, start, end):
        self.calls.append((start, end))
        if self.down:
            raise ConnectionError("offline")
        return self.df[(self.df.index >= start) & (self.df.index < end)]


@pytest.fixture
def disk():
    return DiskCache()


def engine(provider, disk):
    return MarketDataEngine([provider], disk=disk)  # a new engine is a new start: nothing in memory


# -- DiskCache -----------------------------------------------------------------------------------------
def test_round_trip_keeps_candles_unit_and_listing(disk):
    df = daily(10)
    df.index = df.index.as_unit("us")
    disk.save("BTC/USDT", "1d", df, listed=df.index[0])
    out, listed = disk.load("BTC/USDT", "1d")
    pd.testing.assert_frame_equal(out, df, check_freq=False)
    assert listed == df.index[0]
    assert disk.path("BTC/USDT", "1d").parent.name == "BTC%2FUSDT"  # safe on every OS
    assert disk.load("ETH/USDT", "1d") is None


def test_corrupt_file_is_dropped(disk):
    path = disk.path("X", "1d")
    path.parent.mkdir(parents=True)
    path.write_bytes(b"not a cache file")
    assert disk.load("X", "1d") is None and not path.exists()


def test_keeps_only_the_newest_candles(disk, monkeypatch):
    monkeypatch.setattr(cache_mod, "MAX_BARS", 50)
    df = daily(120)
    disk.save("X", "1d", df)
    out, _ = disk.load("X", "1d")
    assert len(out) == 50 and out.index[-1] == df.index[-1]


def age(path, days):
    t = time.time() - days * DAY
    os.utime(path, (t, t))


def test_cleanup_deletes_unused_series(disk):
    for symbol, tf, days in (("OLD", "1d", 61), ("RECENT", "1d", 59), ("MIN", "5m", 15), ("MIN2", "5m", 13)):
        disk.save(symbol, tf, daily(5))
        age(disk.path(symbol, tf), days)
    (disk.folder / "crypto_pairs.json").write_text("[]")  # from before #63
    assert disk.cleanup() == 2
    left = {p.parent.name for p in disk.ohlc_dir.glob("*/*.npz")}
    assert left == {"RECENT", "MIN2"}  # 1m–30m candles go after 14 days, the rest after 60
    assert not (disk.ohlc_dir / "OLD").exists()  # its empty folder too
    assert not (disk.folder / "crypto_pairs.json").exists()


def test_cleanup_over_the_cap_drops_least_recently_used(disk):
    for i in range(10):
        disk.save(f"S{i}", "1d", daily(2000))
        age(disk.path(f"S{i}", "1d"), 10 - i)  # S0 used longest ago
    size = disk.path("S0", "1d").stat().st_size
    disk.max_bytes = int(size * 5.5)
    disk.cleanup()
    left = sorted(p.parent.name for p in disk.ohlc_dir.glob("*/*.npz"))
    assert left == ["S6", "S7", "S8", "S9"]  # down to 80 % of the cap, oldest use first
    disk.load("S6", "1d")  # reading counts as use
    disk.max_bytes = int(size * 3.5)  # room for 2.8 after the cleanup
    disk.cleanup()
    assert sorted(p.parent.name for p in disk.ohlc_dir.glob("*/*.npz")) == ["S6", "S9"]  # S7, S8 were older


def test_cleanup_runs_at_most_once_a_day(disk, monkeypatch):
    runs = []
    monkeypatch.setattr(DiskCache, "cleanup", lambda self: runs.append(1) or 0)
    disk.maybe_cleanup()
    disk.maybe_cleanup()
    assert runs == [1]
    age(disk.folder / ".last_cleanup", 1.1)
    disk.maybe_cleanup()
    assert runs == [1, 1]


def test_info_and_clear(disk):
    disk.save("A", "1d", daily(10))
    disk.save("A", "1h", daily(10))
    disk.symbols_dir.mkdir(parents=True)
    (disk.symbols_dir / "names.json").write_text("{}")
    info = disk.info()
    assert info.series == 2 and info.ohlc_bytes > 0 and info.symbol_bytes == 2
    assert info.total_bytes == info.ohlc_bytes + 2 and info.folder == os_cache_dir()
    disk.clear(symbols=False)
    assert disk.info().series == 0 and disk.info().symbol_bytes == 2
    disk.clear()
    assert disk.info().total_bytes == 0


def test_config_switches_it_off_or_sets_the_cap(tmp_path, monkeypatch):
    config = tmp_path / "config.toml"
    monkeypatch.setenv("TRADEBOT_CONFIG", str(config))
    assert DiskCache.from_config().max_bytes == 500 * 2**20  # on by default
    config.write_text("[cache]\nmax_mb = 50\n")
    assert DiskCache.from_config().max_bytes == 50 * 2**20
    config.write_text("[cache]\nenabled = false\n")
    assert DiskCache.from_config() is None
    assert MarketDataEngine([]).disk is None


def test_human_size():
    assert [human_size(n) for n in (0, 1536, 5 * 2**20, 3 * 2**30)] == ["0 B", "1.5 KB", "5.0 MB", "3.0 GB"]


# -- the engine with a disk cache ------------------------------------------------------------------------
def test_second_start_fetches_only_the_newest_candles(disk):
    df = daily()
    provider = Provider(df)
    first = engine(provider, disk).get_ohlc("BTC/USDT", "1d", bars=200)
    assert len(first) == 200 and disk.load("BTC/USDT", "1d") is not None
    provider.calls.clear()
    again = engine(provider, disk).get_ohlc("BTC/USDT", "1d", bars=200)
    pd.testing.assert_frame_equal(again, first, check_freq=False)
    assert len(provider.calls) == 1 and provider.calls[0][0] == df.index[-2]  # the last two candles again


def test_the_newest_candle_is_updated(disk):
    df = daily()
    provider = Provider(df)
    engine(provider, disk).get_ohlc("X", "1d", bars=100)
    provider.df = df.copy()
    provider.df.iloc[-1, provider.df.columns.get_loc("close")] = 999.0  # the running candle moved
    out = engine(provider, disk).get_ohlc("X", "1d", bars=100)
    assert out["close"].iloc[-1] == 999.0 and disk.load("X", "1d")[0]["close"].iloc[-1] == 999.0


def test_new_candles_are_appended(disk):
    old = daily(end=NOW - pd.Timedelta(days=10))
    provider = Provider(old)
    engine(provider, disk).get_ohlc("X", "1d", bars=100, end=NOW - pd.Timedelta(days=10))
    provider.df = daily(400)
    out = engine(provider, disk).get_ohlc("X", "1d", bars=100)
    assert out.index[-1] == provider.df.index[-1]
    cached = disk.load("X", "1d")[0]
    assert cached.index[0] <= old.index[-100] and cached.index[-1] == provider.df.index[-1]
    assert (cached.index.to_series().diff().dropna() == pd.Timedelta(days=1)).all()  # one run, no hole


def test_adjusted_history_is_fetched_again(disk):
    provider = Provider(daily())
    engine(provider, disk).get_ohlc("AAPL", "1d", bars=200)
    provider.df = daily(scale=0.5)  # a 2:1 split: every past price halves
    provider.calls.clear()
    out = engine(provider, disk).get_ohlc("AAPL", "1d", bars=200)
    pd.testing.assert_frame_equal(out, provider.df.iloc[-200:], check_freq=False)
    assert len(provider.calls) >= 2  # the short refresh found the change, then the full fetch
    cached = disk.load("AAPL", "1d")[0]
    pd.testing.assert_frame_equal(cached, provider.df.loc[cached.index], check_freq=False)  # no old prices left


def test_offline_returns_the_cached_candles(disk):
    provider = Provider(daily())
    first = engine(provider, disk).get_ohlc("X", "1d", bars=100)
    provider.down = True
    out = engine(provider, disk).get_ohlc("X", "1d", bars=100)
    pd.testing.assert_frame_equal(out, first, check_freq=False)
    assert out.attrs.get("offline") is True


def test_no_older_candles_are_requested_before_the_listing(disk):
    listed = daily(50)  # only 50 days of history
    provider = Provider(listed)
    out = engine(provider, disk).get_ohlc("NEW", "1d", start=NOW - pd.Timedelta(days=400))
    assert len(out) == 50 and disk.load("NEW", "1d")[1] == listed.index[0]
    provider.calls.clear()
    engine(provider, disk).get_ohlc("NEW", "1d", start=NOW - pd.Timedelta(days=400))
    assert all(start >= listed.index[0] for start, _ in provider.calls)  # only the newest candles


def test_a_range_not_touching_the_cache_replaces_it(disk):
    provider = Provider(daily(2000))
    engine(provider, disk).get_ohlc("X", "1d", start=NOW - pd.Timedelta(days=1999), end=NOW - pd.Timedelta(days=1900))
    engine(provider, disk).get_ohlc("X", "1d", bars=50)
    cached = disk.load("X", "1d")[0]
    assert cached.index[0] > NOW - pd.Timedelta(days=1800)  # the old range is gone: never two runs with a hole
    assert (cached.index.to_series().diff().dropna() == pd.Timedelta(days=1)).all()


def test_memory_only_engine_writes_nothing(disk):
    MarketDataEngine([Provider(daily())], disk=False).get_ohlc("X", "1d", bars=50)
    assert disk.info().series == 0


def test_peek_reads_the_disk_without_network(disk):
    provider = Provider(daily())
    e = engine(provider, disk)
    assert e.peek("X", "1d", bars=50) is None
    e.get_ohlc("X", "1d", bars=100)
    provider.calls.clear()
    peek = engine(provider, disk).peek("X", "1d", bars=50)
    assert len(peek) == 50 and peek.index[-1] == provider.df.index[-1] and provider.calls == []
    start = provider.df.index[-30]
    assert len(engine(provider, disk).peek("X", "1d", start=start)) == 30


# -- symbol lists ------------------------------------------------------------------------------------------
def test_instrument_names_are_kept_on_disk(monkeypatch):
    from tradebot import symbols
    from tradebot.symbols import SymbolMatch, symbol_name

    calls = []

    def search(query):
        calls.append(query)
        return [SymbolMatch("RHM.DE", "Rheinmetall AG")]

    symbol_name.cache_clear()
    assert symbol_name("RHM.DE", search) == "Rheinmetall AG"
    symbol_name.cache_clear()  # a new start
    assert symbol_name("RHM.DE", search) == "Rheinmetall AG" and calls == ["RHM.DE"]
    assert (DiskCache().symbols_dir / "names.json").is_file()
    symbol_name.cache_clear()


def test_symbol_search_uses_the_shared_disk_files(tmp_path):
    from tradebot.symbols import SymbolMatch, disk_symbol_search

    calls = []

    def yahoo(query):
        calls.append(query)
        return [SymbolMatch("NVDA", "NVIDIA")]

    folder = DiskCache().symbols_dir
    search = disk_symbol_search(list, folder, yahoo=yahoo)
    search.remote[0] = lambda q: []  # no exchange
    assert search.search("nvd")[0].symbol == "NVDA"
    again = disk_symbol_search(list, folder, yahoo=yahoo)
    again.remote[0] = lambda q: []
    assert again.search("nvd")[0].symbol == "NVDA" and calls == ["nvd"]
    assert (folder / "yahoo_search.json").is_file()


# -- CLI ---------------------------------------------------------------------------------------------------
def test_cli_cache_info_and_clear(disk):
    from typer.testing import CliRunner

    from tradebot.cli import main as cli

    disk.save("A", "1d", daily(10))
    res = CliRunner().invoke(cli.app, ["cache", "info"])
    assert res.exit_code == 0 and "1 series" in res.output
    res = CliRunner().invoke(cli.app, ["cache", "clear", "--symbols"])
    assert disk.info().series == 1
    res = CliRunner().invoke(cli.app, ["cache", "clear"])
    assert res.exit_code == 0 and disk.info().series == 0


# -- settings dialog --------------------------------------------------------------------------------------
QApplication = None  # marks this as a GUI module for the conftest's GC handling


@pytest.fixture
def dialog(tmp_path, monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PyQt6")
    from PyQt6.QtWidgets import QApplication as App

    from tradebot.gui.settings import SettingsDialog
    from tradebot.gui.theme import ThemeManager

    dialog.app = App.instance() or App([])  # keep a reference
    monkeypatch.setenv("TRADEBOT_CONFIG", str(tmp_path / "config.toml"))
    themes = ThemeManager(user_dir=tmp_path / "themes")
    themes.apply("dark")
    return SettingsDialog(themes, "dark")


def test_settings_show_the_cache_and_clear_it(dialog, disk):
    disk.save("A", "1d", daily(500))
    disk.save("B", "1d", daily(500))
    dialog._show_cache_usage()
    assert "2 series" in dialog.cache_usage["ohlc"].text() and "KB" in dialog.cache_usage["ohlc"].text()
    assert "of at most 500 MB" in dialog.cache_usage["total"].text()
    changed = []
    dialog.cache_changed.connect(lambda: changed.append(1))
    dialog.clear_cache("ohlc", confirm=False)
    assert disk.info().series == 0 and "0 series" in dialog.cache_usage["ohlc"].text() and changed


def test_clearing_asks_first(dialog, disk, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox

    disk.save("A", "1d", daily(10))
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.No)
    dialog.clear_cache("all")
    assert disk.info().series == 1  # No keeps it
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    dialog.clear_cache("all")
    assert disk.info().series == 0


def test_settings_switch_the_cache_and_set_its_size(dialog):
    dialog.cache_enabled.setChecked(False)
    assert load_config()["cache"]["enabled"] is False
    dialog.cache_max.setValue(1200)
    dialog.cache_max.editingFinished.emit()
    assert load_config()["cache"] == {"enabled": False, "max_mb": 1200}
    assert DiskCache.from_config() is None
    dialog.cache_enabled.setChecked(True)
    assert DiskCache.from_config().max_bytes == 1200 * 2**20


class PeekEngine:
    """Has ``cached`` on disk and ``fresh`` online."""

    def __init__(self, cached, fresh):
        self.cached, self.fresh = cached, fresh

    def peek(self, *args, **kwargs):
        return self.cached

    def get_ohlc(self, *args, **kwargs):
        return self.fresh


def test_chart_shows_the_cache_first_then_refreshes_in_place(dialog, tmp_path):
    from PyQt6.QtCore import QThreadPool
    from PyQt6.QtWidgets import QApplication as App

    from tradebot.gui.chart import ChartView
    from tradebot.gui.theme import ThemeManager

    cached = daily(300, end=NOW - pd.Timedelta(days=1))
    fresh = daily(301)
    fresh.attrs["offline"] = True
    themes = ThemeManager(user_dir=tmp_path / "t")
    themes.apply("dark")
    chart = ChartView(PeekEngine(cached, fresh), themes, ["X"])
    chart.resize(900, 600)
    chart.show()
    messages = []
    chart.status.connect(messages.append)
    chart.load()
    assert chart.df is cached and "from the cache" in messages[-1]  # drawn before any network
    x = chart._x
    chart.price.setXRange(x[100], x[150], padding=0)  # the user zooms in meanwhile
    view = tuple(chart.price.vb.viewRange()[0])
    QThreadPool.globalInstance().waitForDone(5000)
    App.processEvents()
    assert chart.df is fresh and "offline" in messages[-1]
    assert tuple(chart.price.vb.viewRange()[0]) == pytest.approx(view)  # the refresh kept the view
    chart._loaded_symbol = "OTHER"  # moved on to another symbol: a late refresh is dropped
    chart._loaded("X", cached, refresh=True)
    assert chart.df is fresh
