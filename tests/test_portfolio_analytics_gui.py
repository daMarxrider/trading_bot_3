# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Portfolio page analytics (#169), offline."""

import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from tradebot.core.models import AssetClass  # noqa: E402
from tradebot.gui.theme import ThemeManager  # noqa: E402
from tradebot.gui.views import PortfolioView  # noqa: E402
from tradebot.journal import Journal  # noqa: E402
from tradebot.portfolio import DividendInfo, Holding, PortfolioSummary  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def view(app, tmp_path, monkeypatch):
    monkeypatch.setattr("tradebot.gui.views.run_async", lambda work, done, err=None: done(work()))
    j = Journal(tmp_path / "p.sqlite")
    j.record("AAA", "buy", 10, 100, mode="paper", time=datetime(2025, 1, 2, tzinfo=timezone.utc))
    j.close()
    idx = pd.date_range("2025-01-01", periods=60, freq="D", tz="UTC")
    rng = np.random.default_rng(0)

    class Engine:
        def get_ohlc(self, symbol, tf, start=None, **kw):
            n = 100 + rng.normal(0, 1, len(idx)).cumsum()
            return pd.DataFrame({"close": n}, index=idx)

    themes = ThemeManager(user_dir=tmp_path / "themes")
    themes.apply("dark")
    v = PortfolioView(Engine(), themes, journal_factory=lambda: Journal(tmp_path / "p.sqlite"))
    v._pf = PortfolioSummary([
        Holding("AAA", 10, 120.0, 100.0, DividendInfo(), AssetClass.STOCK),
        Holding("BBB", 5, 100.0, None, DividendInfo(), AssetClass.STOCK)])
    yield v
    v.close()
    v.deleteLater()


def test_analyze_fills_all_tabs(view):
    view.targets.setText("AAA=50, BBB=50")
    view.analyze()
    assert "Time-weighted" in view.perf_summary.text()
    assert len(view.corr_model.df) == 2
    assert len(view.risk_model.df) == 2
    assert set(view.reb_model.df["symbol"]) == {"AAA", "BBB"}


def test_bad_target_is_reported(view):
    msgs = []
    view.status.connect(msgs.append)
    view.targets.setText("AAA")
    view.analyze()
    assert any("target allocation" in m for m in msgs)
    assert view.reb_model.df.empty
