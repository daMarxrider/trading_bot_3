# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Persistent UI settings in the OS config folder."""

import json
import sys
from pathlib import Path

import pytest

from tradebot.config import Settings, os_config_dir


@pytest.mark.parametrize("platform, env, expected", [
    ("linux", {"XDG_CONFIG_HOME": "/xdg"}, Path("/xdg/tradebot")),
    ("linux", {}, Path("~/.config/tradebot").expanduser()),
    ("darwin", {}, Path("~/Library/Application Support/tradebot").expanduser()),
    ("win32", {"APPDATA": "/appdata"}, Path("/appdata/tradebot")),
])
def test_os_config_dir(monkeypatch, platform, env, expected):
    monkeypatch.setattr(sys, "platform", platform)
    for key in ("XDG_CONFIG_HOME", "APPDATA"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    assert os_config_dir() == expected


def test_round_trip_creates_the_folder(tmp_path):
    path = tmp_path / "new" / "settings.json"
    s = Settings(path)
    assert s.get("chart.autofit", True) is True  # missing file: defaults
    s.set("chart.autofit", False)
    assert json.loads(path.read_text()) == {"chart.autofit": False}
    assert Settings(path).get("chart.autofit", True) is False  # a new session reads it back
    assert not path.with_suffix(".tmp").exists()


@pytest.mark.parametrize("content", ["{not json", "[1, 2]"])
def test_corrupt_file_falls_back_to_defaults(tmp_path, content):
    path = tmp_path / "settings.json"
    path.write_text(content)
    s = Settings(path)
    assert s.get("chart.autofit", True) is True
    s.set("chart.autofit", False)  # and the next save repairs it
    assert Settings(path).get("chart.autofit") is False


def test_failed_save_does_not_raise(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("")
    s = Settings(blocker / "settings.json")  # its folder can't be created: a file is in the way
    s.set("chart.autofit", False)
    assert s.get("chart.autofit") is False  # still remembered for this session


# -- the theme saved to config.toml (#49) ---------------------------------------------------------
from tradebot.config import load_config, save_config_value  # noqa: E402

USER_CONFIG = """\
# my tradebot config
watchlist = ["AAPL", "BTC/USDT"]   # keep these
theme = "glass"  # set by hand once

[brokers.binance]
api_key = "abc"
theme = "not-top-level"
"""


@pytest.fixture
def config_file(tmp_path, monkeypatch):
    path = tmp_path / "conf" / "config.toml"
    monkeypatch.setenv("TRADEBOT_CONFIG", str(path))
    return path


def test_replaces_only_the_theme_line(config_file):
    config_file.parent.mkdir()
    config_file.write_text(USER_CONFIG)
    assert save_config_value("theme", "dark")
    assert config_file.read_text() == USER_CONFIG.replace('theme = "glass"  # set by hand once',
                                                         'theme = "dark"  # set by hand once')
    assert load_config()["theme"] == "dark" and load_config()["brokers"]["binance"]["theme"] == "not-top-level"


def test_adds_a_missing_key_before_the_first_table(config_file):
    config_file.parent.mkdir()
    config_file.write_text('watchlist = ["AAPL"]\n\n[brokers.binance]\napi_key = "abc"\n')
    assert save_config_value("theme", "light")
    assert config_file.read_text() == 'watchlist = ["AAPL"]\n\ntheme = "light"\n\n[brokers.binance]\napi_key = "abc"\n'
    assert load_config() == {"watchlist": ["AAPL"], "theme": "light", "brokers": {"binance": {"api_key": "abc"}}}


def test_creates_a_missing_file(config_file):
    assert save_config_value("theme", "dark")
    assert config_file.read_text() == 'theme = "dark"\n' and load_config() == {"theme": "dark"}


def test_never_breaks_a_config_it_cant_handle(config_file):
    config_file.parent.mkdir()
    broken = 'theme = "glass\nwatchlist = [\n'  # already invalid TOML
    config_file.write_text(broken)
    assert not save_config_value("theme", "dark")
    assert config_file.read_text() == broken  # untouched


def test_unwritable_location_is_reported_not_raised(tmp_path, monkeypatch):
    blocker = tmp_path / "file"
    blocker.write_text("")
    monkeypatch.setenv("TRADEBOT_CONFIG", str(blocker / "config.toml"))  # its folder can't be created
    assert not save_config_value("theme", "dark")


QApplication = None  # marks this as a GUI module for the conftest's GC handling


def test_choosing_a_theme_saves_it_but_starting_up_doesnt(config_file, monkeypatch):
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PyQt6")
    from PyQt6.QtWidgets import QApplication as App

    from tradebot.gui import app as gui
    from tradebot.gui.theme import ThemeManager

    app = App.instance() or App([])  # keep a reference: a collected QApplication crashes Qt
    monkeypatch.setattr(gui.ChartView, "load", lambda self: None)  # no network
    config_file.parent.mkdir()
    config_file.write_text(USER_CONFIG)
    themes = ThemeManager(user_dir=config_file.parent / "themes")
    monkeypatch.setattr(gui.SettingsDialog, "exec", lambda self: None)  # opened, not blocking
    window = gui.MainWindow(themes, "dark", Settings(config_file.parent / "settings.json"))
    assert config_file.read_text() == USER_CONFIG  # opening the app with the configured theme writes nothing
    assert themes.current.name == "Modern Dark"
    window.show_settings()  # the gear button
    box = window._settings_dialog.theme_box
    assert box.currentData() == "dark"  # opening the dialog with the current theme writes nothing either
    assert config_file.read_text() == USER_CONFIG
    box.setCurrentIndex(box.findData("light"))  # the user picks Light
    assert themes.current.name == "Light" and load_config()["theme"] == "light"
    window.show_settings()  # opened again: shows the theme picked last time
    assert window._settings_dialog.theme_box.currentData() == "light"
    assert "# set by hand once" in config_file.read_text() and "# keep these" in config_file.read_text()
    window.close()


def test_unknown_configured_theme_falls_back_to_glass(config_file, monkeypatch):
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PyQt6")
    from PyQt6.QtWidgets import QApplication as App

    from tradebot.gui import app as gui
    from tradebot.gui.theme import ThemeManager

    app = App.instance() or App([])  # keep a reference: a collected QApplication crashes Qt
    monkeypatch.setattr(gui.ChartView, "load", lambda self: None)
    monkeypatch.setattr(gui.SettingsDialog, "exec", lambda self: None)
    themes = ThemeManager(user_dir=config_file.parent / "themes")
    window = gui.MainWindow(themes, "deleted-user-theme", Settings(config_file.parent / "settings.json"))
    window.show_settings()
    assert themes.current.name == "Glass" and window._settings_dialog.theme_box.currentData() == "glass"
    assert not config_file.exists()  # and nothing written
    window.close()
