# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""The market regime indicator shades the chart background (#168)."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")
pg = pytest.importorskip("pyqtgraph")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from tradebot.config import Settings  # noqa: E402
from tradebot.gui.chart import ChartView  # noqa: E402
from tradebot.gui.theme import ThemeManager  # noqa: E402
from tradebot.indicators.regime import Regime  # noqa: E402

from .conftest import make_ohlc  # noqa: E402


@pytest.fixture
def chart(tmp_path):
    app = QApplication.instance() or QApplication([])
    themes = ThemeManager(user_dir=tmp_path)
    themes.apply("dark")
    c = ChartView(None, themes, ["TEST"], settings=Settings(tmp_path / "settings.json"))
    c.resize(900, 600)
    c.show()
    c._loaded("BTC/USDT", make_ohlc(300, seed=5))
    yield c, app
    c.close()
    c.deleteLater()
    app.processEvents()


def test_regime_draws_background_bands(chart):
    c, app = chart
    c.indicators = [Regime(period=20, threshold=3.0)]
    c.redraw()
    app.processEvents()
    bands = [i for i in c.price.items if isinstance(i, pg.LinearRegionItem)]
    assert bands and all(b.zValue() < 0 for b in bands)
    c.indicators = []
    c.redraw()
    assert not [i for i in c.price.items if isinstance(i, pg.LinearRegionItem)]
