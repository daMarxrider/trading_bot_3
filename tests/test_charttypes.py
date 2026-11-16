# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Heikin-Ashi, Renko and percent-change transforms (#163)."""

import numpy as np
import pandas as pd
import pytest

from tradebot.analysis.charttypes import atr_brick, heikin_ashi, percent_change, renko

from .conftest import make_ohlc


def _closes(values):
    idx = pd.date_range("2024-01-01", periods=len(values), freq="D", tz="UTC")
    v = np.asarray(values, float)
    return pd.DataFrame({"open": v, "high": v, "low": v, "close": v}, index=idx)


def test_heikin_ashi_formula():
    df = make_ohlc(50, seed=1)
    ha = heikin_ashi(df)
    assert ha.index.equals(df.index)
    np.testing.assert_allclose(ha["close"], df[["open", "high", "low", "close"]].mean(axis=1))
    assert ha["open"].iloc[0] == pytest.approx((df["open"].iloc[0] + df["close"].iloc[0]) / 2)
    assert ha["open"].iloc[5] == pytest.approx((ha["open"].iloc[4] + ha["close"].iloc[4]) / 2)
    assert (ha["high"] >= ha[["open", "close"]].max(axis=1)).all()
    assert (ha["low"] <= ha[["open", "close"]].min(axis=1)).all()


def test_heikin_ashi_leaves_input_alone():
    df = make_ohlc(20)
    before = df.copy()
    heikin_ashi(df)
    pd.testing.assert_frame_equal(df, before)


def test_renko_bricks_and_reversal_needs_two():
    r = renko(_closes([100, 101, 103.5, 104, 102, 100.9, 99.5]), brick=1)
    assert r["up"].tolist()[:4] == [True] * 4  # 100 -> 104 in four bricks
    assert r["open"].iloc[0] == 100 and r["close"].iloc[3] == 104
    down = r[~r["up"]]
    assert down["open"].iloc[0] == 103 and down["close"].iloc[0] == 102  # a reversal starts at the last brick's open


def test_renko_index_strictly_increasing_and_inside_candle_span():
    df = _closes([100, 105, 100, 106])
    r = renko(df, brick=1)
    assert r.index.is_monotonic_increasing and r.index.is_unique
    assert r.index[0] >= df.index[1] and r.index[-1] < df.index[-1] + pd.Timedelta(days=1)


def test_renko_default_brick_is_atr_and_validates():
    df = make_ohlc(100, seed=3)
    assert atr_brick(df) > 0
    assert not renko(df).empty
    with pytest.raises(ValueError):
        renko(_closes([5, 5, 5]))  # no range: ATR 0
    assert renko(df.iloc[0:0]).empty


def test_percent_change_starts_at_zero_on_common_index():
    a, b = make_ohlc(30, seed=1)["close"], make_ohlc(30, seed=2)["close"].iloc[5:]
    out = percent_change({"a": a, "b": b})
    assert out.index[0] == b.index[0]
    assert out.iloc[0].tolist() == [0.0, 0.0]
    assert out["a"].iloc[-1] == pytest.approx((a.iloc[-1] / a.loc[out.index[0]] - 1) * 100)
