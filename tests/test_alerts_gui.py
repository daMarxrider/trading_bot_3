# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Background alerts in the GUI (#156): chart context menu and the alerts dialog. No timers are started."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from tradebot import alerts  # noqa: E402
from tradebot.config import Settings  # noqa: E402
from tradebot.gui.alerts import AlertsDialog, AlertWatcher  # noqa: E402
from tradebot.gui.chart import ChartView  # noqa: E402
from tradebot.gui.theme import ThemeManager  # noqa: E402

from .test_chart_drawings import candles  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def chart(app, tmp_path, monkeypatch):
    monkeypatch.setenv("TRADEBOT_ALERTS", str(tmp_path / "alerts.json"))
    themes = ThemeManager(user_dir=tmp_path)
    themes.apply("dark")
    c = ChartView(None, themes, ["TEST"], Settings(tmp_path / "s.json"))
    c.resize(1000, 700)
    c.df = candles()
    c._loaded_symbol = "AAA"
    yield c
    c.close()
    c.deleteLater()


def test_chart_adds_price_alert_toward_the_level(chart):
    chart.add_price_alert(250)  # last close is 200: above
    chart.add_price_alert(150)  # below
    a, b = alerts.load()
    assert (a.symbols, a.op, a.value) == (["AAA"], ">=", 250)
    assert (b.op, b.value) == ("<=", 150)


def test_dialog_toggle_and_remove(app, tmp_path, monkeypatch):
    monkeypatch.setenv("TRADEBOT_ALERTS", str(tmp_path / "alerts.json"))
    alerts.add(alerts.Alert("price", ["X"], op=">=", value=1))
    d = AlertsDialog()
    try:
        assert d.table.rowCount() == 1
        d.table.selectRow(0)
        d.toggle_selected()
        assert alerts.load()[0].enabled is False and "off" in d.table.item(0, 3).text()
        d.remove_selected()
        assert alerts.load() == [] and d.table.rowCount() == 0
    finally:
        d.close()
        d.deleteLater()


def test_watcher_is_idle_until_started(app):
    w = AlertWatcher(poll_ms=10_000)
    try:
        assert not w.timer.isActive()
    finally:
        w.deleteLater()
