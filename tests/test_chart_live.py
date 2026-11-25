# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""The chart's Live button (#158): the forming candle follows a (fake) stream; no network."""

import os
import time

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from tradebot.config import Settings  # noqa: E402
from tradebot.data.stream import MarketStream  # noqa: E402
from tradebot.gui.chart import ChartView  # noqa: E402
from tradebot.gui.live import merge_candle  # noqa: E402
from tradebot.gui.theme import ThemeManager  # noqa: E402

DAY = pd.Timedelta(days=1)


def candles(n=50):
    close = np.linspace(100, 150, n)
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame({"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1.0}, index=idx)


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def chart(app, tmp_path):
    themes = ThemeManager(user_dir=tmp_path)
    themes.apply("dark")
    c = ChartView(None, themes, ["TEST"], Settings(tmp_path / "s.toml"))
    c.resize(1000, 700)
    c.df = candles()
    c._loaded_symbol = "BTC/USDT"
    c.tf.blockSignals(True)
    c.tf.setCurrentText("1d")
    c.tf.blockSignals(False)
    c.show()
    c.redraw(autorange=True)
    yield c
    c.close()
    c.deleteLater()
    QApplication.processEvents()


def pump(cond, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        QApplication.processEvents()
        if cond():
            return True
        time.sleep(0.01)
    return False


def test_merge_candle():
    df = candles(3)
    last = df.index[-1]
    out = merge_candle(df, (last, 0, 200, 50, 160, 9), DAY)
    assert out.loc[last, ["open", "high", "low", "close", "volume"]].tolist() == [df["open"].iloc[-1], 200, 50, 160, 9]
    assert df["close"].iloc[-1] != 160  # the original is untouched
    out = merge_candle(df, (last + DAY, 160, 161, 159, 160.5, 0), DAY)
    assert len(out) == 4 and out["close"].iloc[-1] == 160.5 and out["volume"].iloc[-1] == 0
    out = merge_candle(df, (last + pd.Timedelta(hours=5), 1, 300, 149.5, 170, 0), DAY)  # an offset grid: same candle
    assert len(out) == 3 and out["high"].iloc[-1] == 300 and out["low"].iloc[-1] == df["low"].iloc[-1]
    assert merge_candle(df, (df.index[0], 1, 1, 1, 1, 1), DAY) is None
    assert merge_candle(None, (last, 1, 1, 1, 1, 1), DAY) is None


def test_live_button_streams_the_last_candle(chart):
    last = chart.df.index[-1]
    frames = [pd.DataFrame({"open": [150.0], "high": [158.0], "low": [149.0], "close": [157.0]}, index=[last]),
              pd.DataFrame({"open": [157.0], "high": [157.0], "low": [157.0], "close": [157.0]}, index=[last + DAY])]
    opened = []

    def fake_open(symbol, tf, engine=None):
        opened.append((symbol, tf))
        return MarketStream(symbol, tf, None, lambda: frames[0] if len(opened) and not seen else frames[-1],
                            poll_interval=0.02)

    seen = []
    chart.live._open = fake_open
    chart.live._redraw.setInterval(10)
    chart.live.button.setChecked(True)
    assert opened == [("BTC/USDT", "1d")]
    assert pump(lambda: chart.df["close"].iloc[-1] == 157.0)
    assert chart.df["high"].iloc[-1] == 158.0 and len(chart.df) == 50
    assert pump(lambda: chart.live.current == "polling")
    assert "polling" in chart.live.button.toolTip()
    seen.append(1)
    assert pump(lambda: len(chart.df) == 51)  # the next candle opened
    chart.live.follow()  # the same symbol loaded again: the stream goes on
    assert len(opened) == 1
    chart._loaded_symbol = "ETH/USDT"
    chart.live.follow()
    assert opened[-1] == ("ETH/USDT", "1d")
    chart.live.button.setChecked(False)
    assert chart.live.stream is None and chart.live.current == "stopped"


def test_live_refuses_a_ratio_and_an_unknown_source(chart):
    msgs = []
    chart.status.connect(msgs.append)
    chart._loaded_symbol = "KO:PEP"
    chart.live.button.setChecked(True)
    assert not chart.live.button.isChecked() and "single symbol" in msgs[-1]

    def no_source(symbol, tf, engine=None):
        raise ValueError("no data provider")

    chart._loaded_symbol = "AAPL"
    chart.live._open = no_source
    chart.live.button.setChecked(True)
    assert not chart.live.button.isChecked() and "no data provider" in msgs[-1]
