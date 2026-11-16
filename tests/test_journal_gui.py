# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Journal tab (#159)."""

import os
from datetime import datetime, timezone

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from tradebot.gui.journal import JournalView  # noqa: E402
from tradebot.gui.theme import ThemeManager  # noqa: E402
from tradebot.journal import Journal  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def view(app, tmp_path, monkeypatch):
    monkeypatch.setattr("tradebot.gui.journal.run_async", lambda *a, **k: None)  # no price download
    j = Journal(tmp_path / "g.sqlite")
    t = datetime(2025, 1, 1, tzinfo=timezone.utc)
    j.record("ETH/USDT", "buy", 1, 100, mode="paper", time=t)
    j.record("ETH/USDT", "sell", 0.5, 110, mode="paper", time=t.replace(day=2))
    j.close()
    themes = ThemeManager(user_dir=tmp_path / "themes")
    themes.apply("dark")
    v = JournalView(None, themes, journal_factory=lambda: Journal(tmp_path / "g.sqlite"))
    yield v
    v.close()
    v.deleteLater()


def test_shows_trades_and_saves_notes(view):
    view.refresh()
    assert len(view.model.df) == 2 and view.model.df.iloc[0]["side"] == "sell"  # newest first
    assert "realised <b>+5.00" in view.summary.text()
    view.table.selectRow(0)
    view.note.setText("trimmed")
    view.tags.setText("swing")
    view.save_annotation()
    assert view.model.df.iloc[0]["note"] == "trimmed" and view.model.df.iloc[0]["tags"] == "swing"
