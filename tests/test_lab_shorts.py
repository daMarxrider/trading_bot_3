# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Shorts and leverage in the backtest lab (#149)."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from tradebot.algorithms.direction import Directed  # noqa: E402
from tradebot.algorithms.sizing import Sized  # noqa: E402
from tradebot.backtest import backtest  # noqa: E402
from tradebot.gui.theme import ThemeManager  # noqa: E402
from tradebot.gui.views import BacktestView  # noqa: E402

from .conftest import make_ohlc  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def lab(app, tmp_path):
    themes = ThemeManager(user_dir=tmp_path)
    themes.apply("dark")
    view = BacktestView(None, themes, ["TEST"])
    view.wf_folds.setValue(0)
    view.show()
    view.algo.setCurrentText("sma_cross")
    QApplication.processEvents()
    yield view
    view.close()  # a shown view freed by Python's refcount segfaults in sip (#143)
    view.deleteLater()
    QApplication.processEvents()


def test_long_without_leverage_is_the_default(lab):
    assert lab.direction.currentData() == "long"
    assert not isinstance(lab._algorithm(), Directed)
    assert lab._margin() is None


def test_direction_leverage_and_costs(lab):
    lab.direction.setCurrentIndex(lab.direction.findData("long_short"))
    lab.sizing.setCurrentIndex(lab.sizing.findData("fixed"))
    lab.leverage.setValue(3)
    lab.borrow_fee.setValue(2.0)
    lab.funding.setValue(0.01)
    QApplication.processEvents()
    algo = lab._algorithm()
    assert isinstance(algo, Sized) and isinstance(algo.inner, Directed) and algo.inner.direction == "long_short"
    m = lab._margin()
    assert (m.leverage, m.borrow, m.funding) == (3.0, pytest.approx(0.02), pytest.approx(0.0001))
    res = backtest(make_ohlc(300), algo, margin=m)
    assert (res.positions < 0).any() and "liquidations" in res.metrics


def test_earnings_rule_field_and_mask(lab, monkeypatch):
    from tradebot import fundamentals
    from tradebot.gui import views

    assert lab.earnings_days.value() == 0  # off by default
    df = make_ohlc(30)
    assert views._blackout(df, "AAPL", 0) is None and views._blackout(df, None, 5) is None
    monkeypatch.setattr(fundamentals, "earnings_blackout", lambda d, s, n: (s, n))
    assert views._blackout(df, "AAPL", 5) == ("AAPL", 5)
    monkeypatch.setattr(fundamentals, "earnings_blackout", lambda d, s, n: 1 / 0)
    assert views._blackout(df, "AAPL", 5) is None
