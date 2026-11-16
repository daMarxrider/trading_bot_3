# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Market regime detection (#168): the indicator, combinator filter, evolve primitive, report table."""

import numpy as np
import pandas as pd

from tradebot.algorithms import build_algorithm
from tradebot.backtest import backtest
from tradebot.backtest import report as rep
from tradebot.core.series import OHLC
from tradebot.evolve.primitives import causal_outputs, indicator_outputs
from tradebot.indicators import INDICATORS

from .conftest import make_ohlc


def _trend_frame():
    close = np.r_[np.linspace(100, 200, 100), np.linspace(200, 100, 100), np.full(60, 100.0)]
    idx = pd.date_range("2020-01-01", periods=len(close), freq="D", tz="UTC")
    return pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99, "close": close,
                         "volume": 1.0}, index=idx)


def test_regime_classifies_trend_and_is_exclusive():
    df = _trend_frame()
    out = INDICATORS.get("regime").compute(OHLC.from_frame(df), period=20, threshold=5.0, vol_period=10)
    assert out["bull"][50] == 1 and out["bear"][50] == 0
    assert out["bear"][150] == 1 and out["bull"][150] == 0
    assert out["sideways"][-1] == 1
    assert np.isnan(out["bull"][:20]).all()
    total = out["bull"] + out["bear"] + out["sideways"]
    assert np.all(total[~np.isnan(total)] == 1)
    assert np.all((out["high_vol"] + out["low_vol"])[~np.isnan(out["high_vol"])] == 1)


def test_regime_is_causal_and_offered_to_evolve():
    assert set(causal_outputs(INDICATORS.get("regime"))) == set(INDICATORS.get("regime").outputs)
    assert {"trend", "volatility"} <= {o for c, o, _ in indicator_outputs(["regime"])}


def test_regime_filter_in_combo_only_trades_in_regime():
    df = _trend_frame()
    plain = build_algorithm("combo", entry="close > sma(5)", exit="close < sma(5)")
    filtered = build_algorithm("combo", entry="close > sma(5) & regime(20, 5, 10).bull", exit="close < sma(5)")
    p, f = plain.run(df), filtered.run(df)
    assert (f != 0).sum() < (p != 0).sum()
    bull = INDICATORS.get("regime").compute(OHLC.from_frame(df), period=20, threshold=5.0, vol_period=10)["bull"]
    entered = np.flatnonzero((f.to_numpy() != 0) & (np.r_[0, f.to_numpy()[:-1]] == 0))
    assert len(entered) and all(bull[i] == 1 for i in entered)


def test_regime_table_splits_bars_and_returns():
    df = make_ohlc(400, seed=3)
    res = backtest(df, build_algorithm("combo", entry="close > sma(10)", exit="close < sma(10)"))
    t = rep.regime_table(res, df, period=20, threshold=3.0)
    assert list(t.index) == ["bull", "bear", "sideways", "high volatility", "low volatility"]
    trend = t.loc[["bull", "bear", "sideways"]]
    assert trend["bars"].sum() <= len(df) - 20
    assert abs(trend["share"].sum() - trend["bars"].sum() / len(df)) < 1e-9
    assert t["exposure"].between(0, 1).all()
    assert rep.fmt_regime(t).loc["bull", "share"].endswith("%")
    html = rep.html_report(res, "x", t)
    assert "By market regime" in html and "sideways" in html
