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
