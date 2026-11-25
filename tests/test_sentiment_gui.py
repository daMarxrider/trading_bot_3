# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Sentiment scores in the news list and economic event marks on the chart (#167). The feeds are replaced."""

import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

import pyqtgraph as pg  # noqa: E402
from PyQt6.QtWidgets import QApplication, QLabel, QListWidget  # noqa: E402

from tradebot import economic  # noqa: E402
from tradebot.config import Settings  # noqa: E402
from tradebot.gui.chart import ChartView  # noqa: E402
from tradebot.gui.news import fill_news_list, summary_html  # noqa: E402
from tradebot.gui.theme import ThemeManager  # noqa: E402
from tradebot.news import NewsItem  # noqa: E402

NOW = datetime(2026, 1, 5, 12, 0, tzinfo=timezone.utc)
ITEMS = [NewsItem("AAPL", "Apple profits surge to a record", "p", "http://a", NOW),
         NewsItem("AAPL", "Apple shares plunge after probe", "p", "http://b", NOW),
         NewsItem("AAPL", "Apple holds its annual meeting", "p", "http://c", NOW)]


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(params=["graphite", "paper"])
def themes(app, tmp_path, request):
    t = ThemeManager(user_dir=tmp_path)
    t.apply(request.param)
    return t


def test_every_headline_shows_its_score_with_a_triangle(app, themes):
    widget = QListWidget()
    fill_news_list(widget, ITEMS, themes)
    texts = [widget.itemWidget(widget.item(i)).text() for i in range(3)]
    assert "▲" in texts[0] and "positive" in texts[0]
    assert "▼" in texts[1] and "negative" in texts[1]
    assert "–" in texts[2] and "neutral" in texts[2]
    assert all(isinstance(widget.itemWidget(widget.item(i)), QLabel) for i in range(3))


def test_summary_line_counts_the_split(app, themes):
    text = summary_html(ITEMS, themes.current)
    assert "1 positive · 1 neutral · 1 negative" in text and summary_html([], themes.current) == ""


def test_economic_events_are_marked_above_the_candles(app, themes, tmp_path, monkeypatch):
    monkeypatch.setattr("tradebot.gui.chart.run_async", lambda fn, ok, err=None: ok(fn()))
    cpi = economic.EconEvent(datetime(2024, 6, 12, 12, 30, tzinfo=timezone.utc), "CPI m/m", "USD", "high", "cpi")
    monkeypatch.setattr(economic, "fetch_feed", lambda: [cpi])
    c = ChartView(None, themes, ["ACME"], Settings(tmp_path / "s.json"))
    c.resize(1440, 900)
    c.show()
    close = np.linspace(100, 200, 200)
    idx = pd.date_range(end=pd.Timestamp("2024-06-28", tz="UTC"), periods=200, freq="D")
    c.df = pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 1.0}, index=idx)
    c._loaded_symbol = "ACME"
    c.redraw(autorange=True)
    letters = lambda: sorted(i.toPlainText() for i in c.price.items if isinstance(i, pg.TextItem) and i.toPlainText() in ("R", "C", "J", "!"))  # noqa: E731
    assert letters() == []
    c.events_btn.setChecked(True)
    assert "C" in letters() and "J" in letters()  # the CPI of the feed and the US jobs report of the rule
    c.close()
    c.deleteLater()
