# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""The News sources page of the settings (#194)."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")
from PyQt6.QtWidgets import QApplication  # noqa: E402

from tradebot import news  # noqa: E402
from tradebot.gui.settings import SettingsDialog  # noqa: E402
from tradebot.gui.theme import ThemeManager  # noqa: E402


@pytest.fixture
def dialog(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADEBOT_CONFIG", str(tmp_path / "config.toml"))
    app = QApplication.instance() or QApplication([])  # keep a reference
    dlg = SettingsDialog(ThemeManager(user_dir=tmp_path / "themes"), "graphite", embedded=True)
    dlg._app = app
    news.LAST_ERRORS.clear()
    return dlg


def test_checkboxes_follow_and_save_the_choice(dialog):
    assert dialog.news_boxes["cnbc"].isChecked() and not dialog.news_boxes["bitcoinmagazine"].isChecked()
    dialog.news_boxes["cnbc"].setChecked(False)
    dialog.news_boxes["bitcoinmagazine"].setChecked(True)
    on = news.enabled_sources()
    assert "cnbc" not in on and "bitcoinmagazine" in on


def test_failed_source_gets_a_marker(dialog):
    assert dialog.news_errors["cnbc"].isHidden()
    news.LAST_ERRORS["cnbc"] = "HTTPError: 503"
    dialog._show_news_errors()
    assert not dialog.news_errors["cnbc"].isHidden() and "503" in dialog.news_errors["cnbc"].text()


def test_added_feed_is_listed_and_can_be_removed(dialog, monkeypatch):
    dialog._feed_checked("Wire", "https://x.example/f.xml", 3)
    sid = "custom:https://x.example/f.xml"
    assert dialog.news_boxes[sid].isChecked() and sid in news.enabled_sources()
    dialog.remove_feed(sid)
    assert sid not in dialog.news_boxes and news.custom_sources() == {}


def test_a_rejected_feed_is_not_stored(dialog):
    dialog._feed_rejected("not an RSS or Atom feed")
    assert "Not added" in dialog.feed_result.text() and news.custom_sources() == {}
