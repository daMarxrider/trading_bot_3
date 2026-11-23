# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Higher-timeframe indicators on the chart and in the parameter dialog (#161)."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from tradebot.config import Settings  # noqa: E402
from tradebot.gui.chart import ChartView  # noqa: E402
from tradebot.gui.params import ParamDialog  # noqa: E402
from tradebot.gui.theme import ThemeManager  # noqa: E402
from tradebot.indicators import INDICATORS  # noqa: E402

from .conftest import make_ohlc  # noqa: E402


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def chart(app, tmp_path):
    themes = ThemeManager(user_dir=tmp_path)
    themes.apply("dark")
    c = ChartView(None, themes, ["TEST"], settings=Settings(tmp_path / "settings.json"))
    c.resize(900, 600)
    c.show()
    c._loaded("BTC/USDT", make_ohlc(24 * 20, seed=5, freq="h"))
    yield c
    c.close()
    c.deleteLater()
    app.processEvents()


def test_param_dialog_offers_a_timeframe(app):
    sma = INDICATORS.get("sma")
    dlg = ParamDialog("sma", sma.params, None, None, timeframe=True, tf="1d")
    try:
        assert dlg.tf_box.currentData() == "1d" and dlg.values()["tf"] == "1d"
        dlg.tf_box.setCurrentIndex(0)  # the chart's own candles
        assert "tf" not in dlg.values()
    finally:
        dlg.close()
        dlg.deleteLater()
    plain = ParamDialog("sma", sma.params)
    try:
        assert plain.tf_box is None and "tf" not in plain.values()
    finally:
        plain.close()
        plain.deleteLater()


def test_chart_draws_a_higher_timeframe_indicator(chart, app):
    chart.indicators = [INDICATORS.get("sma")(tf="1d", period=3), INDICATORS.get("rsi")(tf="1mo"),
                        INDICATORS.get("ema")(tf="15m", period=5)]
    chart.redraw()
    app.processEvents()
    labels = [label for rows in chart._rows.values() for label, _ in rows]
    assert any(label.startswith("sma 3 · 1d") for label in labels)
    assert any("1mo" in label for label in labels)  # a monthly RSI on 20 days: drawn, still warming up
    # shorter than the chart's candles: drawn on the chart's own candles, with a note
    assert any(label.startswith("ema 5 · 15m") and "shorter than the candles" in label for label in labels)
