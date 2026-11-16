# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Chart types in the chart: Heikin-Ashi, Renko, Compare and Ratio (#163)."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")
pg = pytest.importorskip("pyqtgraph")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from tradebot.config import Settings  # noqa: E402
from tradebot.gui.chart import CandlestickItem, ChartView, Curve  # noqa: E402
from tradebot.gui.theme import ThemeManager  # noqa: E402
from tradebot.indicators import INDICATORS  # noqa: E402

from .conftest import make_ohlc  # noqa: E402


class Engine:
    def __init__(self):
        self.calls = []

    def peek(self, *a, **k):
        return None

    def get_ohlc(self, symbol, tf, **k):
        self.calls.append(symbol)
        return make_ohlc(300, seed=len(symbol))


@pytest.fixture
def chart(tmp_path):
    app = QApplication.instance() or QApplication([])
    themes = ThemeManager(user_dir=tmp_path)
    themes.apply("dark")
    engine = Engine()
    c = ChartView(engine, themes, ["TEST"], settings=Settings(tmp_path / "settings.json"))
    c.resize(900, 600)
    c.show()
    c._loaded("BTC/USDT", make_ohlc(300, seed=5))
    yield c, app, engine
    c.close()
    c.deleteLater()
    app.processEvents()


def _candles(c):
    return [i for i in c.price.items if isinstance(i, CandlestickItem)]


def _wait(app, condition):
    for _ in range(100):
        app.processEvents()
        if condition():
            return
        pg.QtCore.QThread.msleep(20)


def test_heikin_ashi_and_renko_draw_candles_but_indicators_use_real_data(chart):
    c, app, _ = chart
    c.indicators = [INDICATORS.get("sma")(period=10)]
    c.type_box.setCurrentText("Heikin-Ashi")
    assert len(_candles(c)) == 1 and len(c._x) == 300
    c.type_box.setCurrentText("Renko")
    assert len(_candles(c)) == 1
    assert any(isinstance(i, Curve) for i in c.price.items)  # the sma is still drawn on the real candles
    c.type_box.setCurrentText("Standard")
    assert len(_candles(c)) == 1


def test_compare_draws_one_line_per_symbol_and_no_candles(chart):
    c, app, engine = chart
    c.symbol.setText("BTC/USDT")
    c.versus.setText("ETH/USDT, SOL/USDT")
    c.type_box.setCurrentText("Compare")
    _wait(app, lambda: len(c.compare) == 2)
    assert sorted(c.compare) == ["ETH/USDT", "SOL/USDT"]
    assert not _candles(c)
    curves = [i for i in c.price.items if isinstance(i, Curve)]
    assert len(curves) == 3
    assert all(abs(cv.yData[0]) < 1e-9 for cv in curves)  # all start at 0 %


def test_ratio_loads_the_pair_symbol(chart):
    c, app, engine = chart
    c.symbol.setText("KO")
    c.versus.setText("PEP")
    c.type_box.setCurrentText("Ratio")
    _wait(app, lambda: "KO:PEP" in engine.calls)
    assert "KO:PEP" in engine.calls
