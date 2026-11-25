# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""The symbol info page and the event markers of the chart (#165). The data provider is replaced."""

import os
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

import pyqtgraph as pg  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from tradebot import fundamentals as fund  # noqa: E402
from tradebot.config import Settings  # noqa: E402
from tradebot.gui.chart import ChartView  # noqa: E402
from tradebot.gui.theme import ThemeManager  # noqa: E402

LAST = pd.Timestamp("2024-06-28", tz="UTC")


def candles(n=200):
    close = np.linspace(100, 200, n)
    idx = pd.date_range(end=LAST, periods=n, freq="D")
    return pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 1.0}, index=idx)


def snapshot(symbol="ACME"):
    day = lambda back: LAST.date() - timedelta(days=back)  # noqa: E731
    events = [fund.Event(day(30), "earnings", "Earnings"), fund.Event(day(60), "dividend", "Ex-dividend 0.24", 0.24),
              fund.Event(day(90), "split", "Split 4:1", 4.0), fund.Event(day(120), "earnings", "Earnings"),
              fund.Event(date.today() + timedelta(days=12), "earnings", "Earnings"),
              fund.Event(date.today() + timedelta(days=4), "dividend", "Ex-dividend")]
    f = fund.Fundamentals("Acme Corporation", "Industrials", "USD", 12.5, 11.0, 1.8, 0.12, 0.2, 0.07, -0.03, 0.02, 5e10)
    return fund.Snapshot(symbol, f, sorted(events), 1.0)


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(params=["graphite", "paper"])
def chart(app, tmp_path, monkeypatch, request):
    monkeypatch.setattr(fund, "load", lambda symbol, *a, **k: snapshot(symbol))
    monkeypatch.setattr("tradebot.gui.chart.run_async", lambda fn, ok, err=None: ok(fn()))
    themes = ThemeManager(user_dir=tmp_path)
    themes.apply(request.param)
    c = ChartView(None, themes, ["ACME"], Settings(tmp_path / "s.json"))
    c.resize(1440, 900)
    c.show()
    c.df = candles()
    c._loaded_symbol = "ACME"
    QApplication.processEvents()
    yield c
    c.close()
    c.deleteLater()


def marks(c):
    return [i for i in c.price.items if isinstance(i, pg.TextItem) and i.toPlainText() in ("E", "S", "D")]


def test_markers_are_off_until_switched_on(chart):
    chart.redraw(autorange=True)
    assert not marks(chart) and not chart.events_btn.isChecked()
    chart.events_btn.setChecked(True)
    assert sorted(m.toPlainText() for m in marks(chart)) == ["D", "E", "E", "S"]  # the future dates are off the chart


def test_info_page_shows_figures_and_events(chart):
    chart.sidebar.show_category("info")
    chart._loaded("ACME", candles())
    text = chart.info.facts.text()
    assert "12.5" in text and "Dividend yield" in text and "2.0%" in text and "▼" in text and "▲" in text
    assert chart.info.events.count() == 6
    assert "upcoming" in chart.info.events.item(0).text() and chart.info.events.item(0).text().startswith(("E", "D"))


def test_crypto_has_a_message_and_no_fetch(chart, monkeypatch):
    monkeypatch.setattr(fund, "load", lambda *a, **k: pytest.fail("no fetch for crypto"))
    chart.events_btn.setChecked(True)
    chart.sidebar.show_category("info")
    chart._loaded("BTC/USDT", candles())
    assert "stocks only" in chart.info.note.text() and not marks(chart)


def test_failed_fetch_leaves_the_chart_working(chart, monkeypatch):
    monkeypatch.setattr(fund, "load", lambda s, *a, **k: fund.Snapshot(s, error="offline"))
    chart.sidebar.show_category("info")
    chart._loaded("ACME", candles())
    assert "offline" in chart.info.note.text() and chart.info.note.isVisible()


def test_setting_is_remembered(chart, tmp_path):
    chart.events_btn.setChecked(True)
    assert Settings(tmp_path / "s.json").get("chart.events") is True


@pytest.mark.skipif(not os.environ.get("TRADEBOT_SHOTS"), reason="renders PNGs for a visual check")
def test_render_screenshots(chart):
    out = os.environ["TRADEBOT_SHOTS"]
    chart.events_btn.setChecked(True)
    chart.sidebar.show_category("info")
    chart._loaded("ACME", candles())
    for name, (w, h) in {"full": (1440, 900), "min": (900, 600)}.items():
        chart.resize(w, h)
        QApplication.processEvents()
        chart.grab().save(f"{out}/{chart.themes.current.name}_{name}.png")
