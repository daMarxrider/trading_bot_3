# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""The chart's collapsible sidebar and its news for the charted symbol (#41)."""

import os
from datetime import datetime, timedelta, timezone

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

from PyQt6.QtCore import QThreadPool  # noqa: E402
from PyQt6.QtWidgets import QApplication, QLabel  # noqa: E402

import tradebot.news as news  # noqa: E402
from tradebot.config import Settings  # noqa: E402
from tradebot.gui.chart import ChartView  # noqa: E402
from tradebot.gui.news import NewsPanel, ago  # noqa: E402
from tradebot.gui.sidebar import ChartSidebar  # noqa: E402
from tradebot.gui.theme import ThemeManager  # noqa: E402

from .conftest import make_ohlc  # noqa: E402

NOW = datetime.now(timezone.utc)


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def themes(app, tmp_path):
    t = ThemeManager(user_dir=tmp_path / "themes")
    t.apply("dark")
    return t


@pytest.fixture
def fetched(monkeypatch):
    """Records the symbols news is fetched for; returns one headline per symbol."""
    calls = []

    def fetch(symbols, limit=50):
        calls.append(list(symbols))
        return [news.NewsItem(s, f"{s} headline", "Reuters", f"https://example.com/{s}", NOW - timedelta(hours=2))
                for s in symbols]

    monkeypatch.setattr(news, "fetch_news", fetch)
    return calls


def settle():
    QThreadPool.globalInstance().waitForDone(5000)
    QApplication.processEvents()


def make_chart(themes, settings=None):
    c = ChartView(None, themes, ["TEST"], settings)
    c.resize(1200, 700)
    c.show()
    QApplication.processEvents()
    return c


# -- the sidebar -------------------------------------------------------------------------------------------
def test_open_by_default_and_the_topmost_entry_collapses_it(themes, tmp_path):
    settings = Settings(tmp_path / "s.json")
    chart = make_chart(themes, settings)
    bar = chart.sidebar
    assert bar.is_open and bar.panel.isVisible() and bar.toggle.text() == "»"
    assert bar.strip.itemAt(0).widget() is bar.toggle  # the topmost entry
    wide = bar.width()
    bar.toggle.click()
    for _ in range(3):  # the layout settles over a couple of event rounds
        QApplication.processEvents()
    assert not bar.is_open and not bar.panel.isVisible() and bar.toggle.text() == "«"
    assert bar.width() < wide / 4 and chart.canvas.width() > 900  # the chart takes the room
    again = make_chart(themes, Settings(tmp_path / "s.json"))  # a new session
    assert not again.sidebar.is_open
    again.sidebar.toggle.click()
    assert again.sidebar.is_open and again.sidebar.toggle.text() == "»"


def test_sidebar_is_right_of_the_chart(themes):
    chart = make_chart(themes)
    assert chart.sidebar.geometry().left() >= chart.canvas.geometry().right()


def test_categories(themes, tmp_path):
    settings = Settings(tmp_path / "s.json")
    bar = ChartSidebar(settings)
    shown = []
    bar.shown.connect(shown.append)
    bar.add_category("news", "📰", "News", QLabel("n"))
    bar.add_category("other", "⭐", "Other", QLabel("o"))
    assert bar.current == "news" and bar.title.text() == "News"
    bar.set_open(False)
    bar.group.buttons()[1].click()  # a category button opens a collapsed sidebar on it
    assert bar.is_open and bar.current == "other" and bar.title.text() == "Other" and shown[-1] == "other"
    assert bar.strip.itemAt(2).widget() is bar.group.buttons()[1]  # below the toggle, in order
    restored = ChartSidebar(Settings(tmp_path / "s.json"))
    restored.add_category("news", "📰", "News", QLabel("n"))
    restored.add_category("other", "⭐", "Other", QLabel("o"))
    assert restored.current == "other"  # remembered


def test_strip_buttons_are_finger_sized_in_touch_mode(app):
    assert ChartSidebar(touch=True).toggle.width() == 44 and ChartSidebar().toggle.width() == 32


# -- news for the charted symbol ------------------------------------------------------------------------------
def test_news_follows_the_charted_symbol(themes, fetched):
    chart = make_chart(themes)
    chart._loaded("AAPL", make_ohlc(100))
    settle()
    assert fetched == [["AAPL"]] and chart.news.list.count() == 1
    assert "AAPL headline" in chart.news.list.itemWidget(chart.news.list.item(0)).text()
    chart._loaded("BTC/USDT", make_ohlc(100))
    settle()
    assert fetched[-1] == ["BTC/USDT"]
    chart._loaded("BTC/USDT", make_ohlc(120), refresh=True)  # newer candles of the same symbol: no refetch
    settle()
    assert len(fetched) == 2


def test_no_news_is_fetched_while_collapsed(themes, fetched):
    chart = make_chart(themes)
    chart.sidebar.toggle.click()  # collapse
    chart._loaded("AAPL", make_ohlc(100))
    settle()
    assert fetched == []
    chart.sidebar.toggle.click()  # expanding fetches it
    settle()
    assert fetched == [["AAPL"]]


def test_late_news_for_a_previous_symbol_is_dropped(themes):
    panel = NewsPanel(themes)
    panel.set_visible(True)
    panel.symbol = "NEW"
    panel._loaded("OLD", [news.NewsItem("OLD", "old", "x", "u", NOW)])
    assert panel.items == [] and panel.list.count() == 0


def test_empty_and_error_states(themes, monkeypatch):
    panel = NewsPanel(themes)
    panel.set_visible(True)
    panel.set_symbol("XYZ")
    settle()  # conftest: fetch_news returns nothing
    assert panel.body.currentWidget() is panel.message and "No news for XYZ" in panel.message.text()

    def down(symbols, limit=50):
        raise ConnectionError("offline")

    monkeypatch.setattr(news, "fetch_news", down)
    panel.set_symbol("ABC")
    settle()
    assert "unavailable" in panel.message.text() and "ABC" in panel.message.text()


def test_clicking_a_headline_opens_it(themes, fetched, monkeypatch):
    from PyQt6.QtGui import QDesktopServices

    opened = []
    monkeypatch.setattr(QDesktopServices, "openUrl", lambda url: opened.append(url.toString()))
    panel = NewsPanel(themes)
    panel.resize(340, 500)
    panel.set_visible(True)
    panel.set_symbol("AAPL", "Apple Inc.")
    settle()
    assert panel.header.text() == "Apple Inc. (AAPL)"
    panel.list.itemClicked.emit(panel.list.item(0))  # one click, not a double-click
    assert opened == ["https://example.com/AAPL"]


def test_refreshes_while_visible(themes, fetched):
    panel = NewsPanel(themes)
    panel.set_symbol("AAPL")
    panel.set_visible(True)
    assert panel.timer.isActive()
    settle()
    panel.refresh_btn.click()
    settle()
    assert fetched == [["AAPL"], ["AAPL"]]
    panel.set_visible(False)
    assert not panel.timer.isActive()


@pytest.mark.parametrize("key", ["glass", "dark", "light"])
def test_readable_in_every_theme(app, tmp_path, key, fetched):
    themes = ThemeManager(user_dir=tmp_path)
    t = themes.apply(key)
    panel = NewsPanel(themes)
    panel.set_visible(True)
    panel.set_symbol("AAPL")
    settle()
    html = panel.list.itemWidget(panel.list.item(0)).text()
    assert t.muted in html  # time and publisher in the theme's muted colour; the title in its text colour


def test_ago():
    now = datetime(2026, 10, 29, 12, 0, tzinfo=timezone.utc)
    cases = [(timedelta(seconds=20), "just now"), (timedelta(minutes=5), "5 min ago"), (timedelta(hours=3), "3 h ago"),
             (timedelta(days=2), "2 d ago"), (timedelta(days=10), "2026-10-19")]
    assert [ago(now - d, now) for d, _ in cases] == [text for _, text in cases]
