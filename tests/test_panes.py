# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Resizable panes (#95): draggable dividers, remembered sizes, reset, scrolling panes, hidden cards."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")

from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtWidgets import QApplication, QLabel, QScrollArea, QWidget  # noqa: E402

from tradebot.config import Settings  # noqa: E402
from tradebot.gui.common import Panes  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def make(settings, key="test.panes"):
    panes = Panes(Qt.Orientation.Vertical, key, settings)
    top, tall, hidden = QWidget(), QLabel("line\n" * 200), QLabel("hidden")
    hidden.hide()
    panes.add(top, 3)
    panes.add(tall, 1, scroll=True)
    panes.add(hidden, 1, scroll=True)
    panes.resize(400, 800)
    return panes, top, tall, hidden


def test_defaults_follow_the_stretch_and_tall_panes_scroll(app, tmp_path):
    panes, top, tall, hidden = make(Settings(tmp_path / "s.json"))
    panes.show()
    QApplication.processEvents()
    sizes = panes.sizes()
    assert sizes[2] == 0 and sizes[0] > 2 * sizes[1] > 0  # 3 : 1, the hidden one takes nothing
    assert isinstance(panes.widget(1), QScrollArea) and panes.widget(1).widget() is tall  # 200 lines fit in a pane
    hidden.show()  # the card appears: its pane too
    QApplication.processEvents()
    assert panes.widget(2).isVisible() and panes.sizes()[2] > 0


def test_sizes_are_remembered_and_reset(app, tmp_path):
    settings = Settings(tmp_path / "s.json")
    panes = make(settings)[0]
    panes.show()
    QApplication.processEvents()
    panes.moveSplitter(150, 1)  # drag the first divider up
    panes._save()
    saved = settings.get("test.panes")
    assert saved[0] < 200
    again = make(Settings(tmp_path / "s.json"))[0]  # the app starts again
    again.show()
    QApplication.processEvents()
    restored = again.sizes()
    assert restored[0] / restored[1] == pytest.approx(saved[0] / saved[1], rel=0.02)  # scaled to the space there is
    again.handle(1).mouseDoubleClickEvent(None)  # double-click a divider
    assert again.sizes()[0] > 2 * again.sizes()[1] and Settings(tmp_path / "s.json").get("test.panes") is None


def test_explicitly_hidden_cards_stay_hidden(app, tmp_path):
    panes = Panes(Qt.Orientation.Vertical)
    card = QLabel("card")
    card.hide()
    pane = panes.add(card, scroll=True)
    panes.show()
    QApplication.processEvents()
    assert card.isHidden() and not pane.isVisible()
    shown = QLabel("never hidden")  # never shown either, but not hidden on purpose
    assert panes.add(shown, scroll=True).isVisibleTo(panes)
