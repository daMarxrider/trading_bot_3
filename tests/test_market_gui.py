# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Crypto market data in the GUI (#166): an indicator pane that fetches its data in the background, and the order
book depth page of the chart sidebar. No network: the engine's market data is a fake."""

import os
import time

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")
pg = pytest.importorskip("pyqtgraph")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from tradebot.config import Settings  # noqa: E402
from tradebot.data.market import MarketDataError  # noqa: E402
from tradebot.gui.chart import ChartView, Curve  # noqa: E402
from tradebot.gui.depth import DepthPanel, imbalance_text  # noqa: E402
from tradebot.gui.theme import ThemeManager  # noqa: E402
from tradebot.indicators import INDICATORS  # noqa: E402

from .test_market_data import FakeExchange, frame, md  # noqa: E402


class Engine:
    def __init__(self, market):
        self.market = market


def wait(app, cond, seconds=5.0):
    end = time.time() + seconds
    while time.time() < end and not cond():
        app.processEvents()
        time.sleep(0.01)
    return cond()


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def chart(tmp_path, app):
    themes = ThemeManager(user_dir=tmp_path)
    themes.apply("dark")
    c = ChartView(Engine(md(spot=FakeExchange())), themes, ["TEST"], settings=Settings(tmp_path / "settings.json"))
    c.resize(1100, 700)
    c.show()
    c._loaded("BTC/USDT", frame(60))
    yield c
    c.close()
    c.deleteLater()
    app.processEvents()


def test_market_indicator_pane_fetches_its_data_in_the_background(chart, app):
    chart.indicators = [INDICATORS.get("fear_greed")()]
    chart.redraw()
    assert wait(app, lambda: "fear_greed" in chart.df.columns)
    assert chart.df["fear_greed"].notna().any()
    assert len(chart.panes) == 1 and any(isinstance(i, Curve) for i in chart.panes[0].items)


def test_market_indicator_says_why_it_has_no_data(tmp_path, app):
    themes = ThemeManager(user_dir=tmp_path)
    themes.apply("dark")
    c = ChartView(Engine(md()), themes, ["TEST"], settings=Settings(tmp_path / "settings.json"))
    c.show()
    c._loaded("AAPL", frame(60))
    c.indicators = [INDICATORS.get("funding")()]
    c.redraw()
    assert wait(app, lambda: c._market_errors)
    assert "not a crypto pair" in next(iter(c._market_errors.values()))
    assert any("no data" in label for rows in c._rows.values() for label, _ in rows)
    c.close()
    c.deleteLater()
    app.processEvents()


def test_imbalance_pairs_colour_with_a_triangle():
    assert imbalance_text(0.3, "±2 %")[0].startswith("▲") and imbalance_text(0.3, "±2 %")[1] == "up"
    assert imbalance_text(-0.3, "±2 %")[0].startswith("▼") and imbalance_text(-0.3, "±2 %")[1] == "down"
    assert imbalance_text(0.01, "±2 %")[1] == "flat"


@pytest.fixture
def panel(tmp_path, app):
    themes = ThemeManager(user_dir=tmp_path)
    themes.apply("dark")
    p = DepthPanel(Engine(md(spot=FakeExchange())), themes)
    p.resize(340, 520)
    p.show()
    yield p
    p.timer.stop()
    p.close()
    p.deleteLater()
    app.processEvents()


def test_depth_panel_loads_only_while_visible_and_draws_both_sides(panel, app):
    panel.set_symbol("BTC/USDT")
    assert panel.book is None  # not visible yet: no request
    panel.set_visible(True)
    assert wait(app, lambda: panel.book is not None)
    assert panel.body.currentWidget() is panel._page
    assert len([i for i in panel.plot.getPlotItem().items if isinstance(i, pg.PlotDataItem)]) == 2
    assert "Mid" in panel.stats.text() and "spread" in panel.stats.text()
    panel.set_visible(False)
    assert not panel.timer.isActive()


def test_depth_panel_for_stocks_and_failures(panel, app):
    panel.set_visible(True)
    panel.set_symbol("AAPL")
    assert "crypto" in panel.message.text() and panel.book is None
    panel.engine.market.spot = FakeExchange(fail=True)
    panel.set_symbol("ETH/USDT")
    assert wait(app, lambda: "unavailable" in panel.message.text())
    assert isinstance(MarketDataError("x"), RuntimeError)


def test_depth_range_changes_the_chart(panel, app):
    panel.set_symbol("BTC/USDT")
    panel.set_visible(True)
    assert wait(app, lambda: panel.book is not None)
    panel.range.setCurrentText("±1 %")
    app.processEvents()
    low, high = panel.plot.getPlotItem().viewRange()[0]
    assert np.isclose((high - low) / 2, 100 * 0.01 * 1.02, rtol=0.05)
