# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Third-party tool paths in config.toml and the settings dialog (#59)."""

import os
import stat
import tomllib

import pytest

from tradebot import tools
from tradebot.config import load_config, save_config_value

USER_CONFIG = """\
theme = "glass"  # mine

[brokers.binance]
api_key = "abc"

[tools]
sc = "/old/sc"  # set by hand
other = "x"

[brokers.kucoin]
api_key = "def"
"""


@pytest.fixture
def config_file(tmp_path, monkeypatch):
    path = tmp_path / "config.toml"
    monkeypatch.setenv("TRADEBOT_CONFIG", str(path))
    return path


@pytest.fixture
def executable(tmp_path):
    path = tmp_path / "bin" / "sc"
    path.parent.mkdir()
    path.write_text("#!/bin/sh\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


# -- [table] keys in config.toml ----------------------------------------------------------------------
def test_replaces_a_key_in_its_table_only(config_file):
    config_file.write_text(USER_CONFIG)
    assert save_config_value("sc", "/new/sc", table="tools")
    text = config_file.read_text()
    assert 'sc = "/new/sc"  # set by hand' in text and text.count("sc =") == 1
    assert text.replace('"/new/sc"', '"/old/sc"') == USER_CONFIG  # nothing else changed


def test_adds_a_key_at_the_end_of_its_table(config_file):
    config_file.write_text(USER_CONFIG)
    assert save_config_value("new", "1", table="tools")
    assert 'other = "x"\nnew = "1"\n\n[brokers.kucoin]' in config_file.read_text()
    assert load_config()["brokers"]["kucoin"] == {"api_key": "def"}


def test_adds_a_missing_table_at_the_end(config_file):
    config_file.write_text('theme = "glass"\n')
    assert save_config_value("sc", "/x/sc", table="tools")
    assert config_file.read_text() == 'theme = "glass"\n\n[tools]\nsc = "/x/sc"\n'
    config_file.unlink()
    assert save_config_value("sc", "/x/sc", table="tools")  # a missing file is created
    assert config_file.read_text() == '[tools]\nsc = "/x/sc"\n'


def test_none_removes_the_key(config_file):
    config_file.write_text(USER_CONFIG)
    assert save_config_value("sc", None, table="tools")
    assert "sc =" not in config_file.read_text() and load_config()["tools"] == {"other": "x"}
    assert save_config_value("missing", None, table="nothing")  # nothing to remove: fine, file unchanged
    assert "[nothing]" not in config_file.read_text()


def test_a_table_name_isnt_matched_inside_another(config_file):
    config_file.write_text('[toolsets]\nsc = "keep"\n')
    assert save_config_value("sc", "/x", table="tools")
    assert load_config() == {"toolsets": {"sc": "keep"}, "tools": {"sc": "/x"}}


# -- tools -------------------------------------------------------------------------------------------
def test_configured_path_wins_over_path_lookup(config_file, executable, monkeypatch):
    monkeypatch.setenv("PATH", str(executable.parent))
    assert tools.configured_path("sc") == ""
    assert tools.locate("sc") == str(executable)  # found on PATH
    assert tools.set_path("sc", "/nowhere/sc")
    assert tools.configured_path("sc") == "/nowhere/sc"
    assert tools.locate("sc") is None  # a configured path that doesn't exist isn't replaced by PATH
    assert tools.set_path("sc", "  ")  # empty: back to PATH
    assert "tools" not in load_config() or "sc" not in load_config()["tools"]
    assert tools.locate("sc") == str(executable)


def test_a_non_executable_file_is_not_found(tmp_path):
    plain = tmp_path / "sc.txt"
    plain.write_text("")
    assert tools.locate("sc", str(plain)) is None


def test_every_tool_has_a_download_page():
    for t in tools.TOOLS:
        assert t.url.startswith("https://") and t.command and t.key


# -- the settings dialog ------------------------------------------------------------------------------
QApplication = None  # marks this as a GUI module for the conftest's GC handling


@pytest.fixture
def app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PyQt6")
    from PyQt6.QtWidgets import QApplication as App

    return App.instance() or App([])


@pytest.fixture
def dialog(app, config_file, tmp_path, monkeypatch):
    from tradebot.gui.settings import SettingsDialog
    from tradebot.gui.theme import ThemeManager

    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    themes = ThemeManager(user_dir=tmp_path / "themes")
    themes.apply("dark")
    return SettingsDialog(themes, "dark")


def test_dialog_shows_missing_tool_with_download_link(dialog):
    status = dialog.status["sc"].text()
    assert "not found on PATH" in status and tools.tool("sc").url in status
    assert dialog.paths["sc"].text() == "" and "PATH" in dialog.paths["sc"].placeholderText()


def test_dialog_saves_an_edited_path(dialog, executable, config_file):
    edit = dialog.paths["sc"]
    edit.setText(str(executable))
    edit.editingFinished.emit()
    assert tomllib.loads(config_file.read_text())["tools"]["sc"] == str(executable)
    assert "✓" in dialog.status["sc"].text() and str(executable) in dialog.status["sc"].text()
    edit.setText("")
    edit.editingFinished.emit()
    assert "sc" not in load_config().get("tools", {})


def test_dialog_browse_picks_a_file(dialog, executable, monkeypatch):
    from PyQt6.QtWidgets import QFileDialog

    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (str(executable), ""))
    dialog._browse("sc")
    assert tools.configured_path("sc") == str(executable)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: ("", ""))  # cancelled: unchanged
    dialog._browse("sc")
    assert tools.configured_path("sc") == str(executable)


def test_dialog_reports_a_path_it_cannot_save(dialog, monkeypatch):
    messages = []
    dialog.message.connect(messages.append)
    monkeypatch.setattr(tools, "set_path", lambda key, path: False)
    dialog.paths["sc"].setText("/x/sc")
    dialog.paths["sc"].editingFinished.emit()
    assert messages and "not saved" in messages[0]


def test_gear_button_is_at_the_bottom_of_the_sidebar(app, config_file, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QComboBox

    from tradebot.config import Settings
    from tradebot.gui import app as gui
    from tradebot.gui.theme import ThemeManager

    monkeypatch.setattr(gui.ChartView, "load", lambda self: None)
    window = gui.MainWindow(ThemeManager(user_dir=tmp_path / "themes"), "dark", Settings(tmp_path / "s.json"))
    side = window.settings_btn.parentWidget()
    lay = side.layout()
    assert lay.itemAt(lay.count() - 1).widget() is window.settings_btn  # last: the bottom left
    assert "⚙" in window.settings_btn.text()
    assert not side.findChildren(QComboBox)  # the theme selector moved into the dialog
    window.close()
