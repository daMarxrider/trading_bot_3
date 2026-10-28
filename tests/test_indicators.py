# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Indicator accuracy against straightforward pandas reference implementations."""

import numpy as np
import pandas as pd
import pytest

from tradebot.core import ops
from tradebot.core.series import OHLC
from tradebot.indicators import INDICATORS
from tradebot.indicators.builtin import ATR, EMA, MACD, RSI, SMA, Bollinger, Donchian, Ichimoku, Stochastic


def close_eq(a, b):
    np.testing.assert_allclose(np.asarray(a, float), np.asarray(b, float), rtol=1e-9, atol=1e-9, equal_nan=True)


def test_sma_ema(ohlc):
    c = ohlc["close"]
    close_eq(SMA(period=20)(ohlc)["sma_value"], c.rolling(20).mean())
    close_eq(EMA(period=20)(ohlc)["ema_value"], c.ewm(span=20, adjust=False).mean())


def test_rsi_matches_wilder(ohlc):
    c = ohlc["close"]
    d = c.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    ref = 100 - 100 / (1 + gain / loss)
    ref.iloc[:14] = np.nan
    close_eq(RSI(period=14)(ohlc)["rsi_value"], ref)


def test_rsi_bounds_and_all_gains():
    df = pd.DataFrame({k: np.arange(1.0, 51.0) for k in ("open", "high", "low", "close")})
    rsi = RSI(period=14)(df)["rsi_value"]
    assert rsi.iloc[:14].isna().all()
    assert (rsi.iloc[14:] == 100).all()


def test_macd(ohlc):
    c = ohlc["close"]
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    sig = macd.ewm(span=9, adjust=False).mean()
    out = MACD()(ohlc)
    close_eq(out["macd_macd"], macd)
    close_eq(out["macd_signal"], sig)
    close_eq(out["macd_hist"], macd - sig)


def test_bollinger(ohlc):
    c = ohlc["close"]
    mid, sd = c.rolling(20).mean(), c.rolling(20).std(ddof=0)
    out = Bollinger(period=20, width=2)(ohlc)
    close_eq(out["bbands_upper"], mid + 2 * sd)
    close_eq(out["bbands_lower"], mid - 2 * sd)


def test_atr(ohlc):
    prev = ohlc["close"].shift()
    tr = pd.concat([ohlc.high - ohlc.low, (ohlc.high - prev).abs(), (ohlc.low - prev).abs()], axis=1).max(axis=1)
    close_eq(ATR(period=14)(ohlc)["atr_value"], tr.ewm(alpha=1 / 14, adjust=False).mean())


def test_stoch_donchian_ichimoku(ohlc):
    hh, ll = ohlc.high.rolling(14).max(), ohlc.low.rolling(14).min()
    k = 100 * (ohlc.close - ll) / (hh - ll)
    out = Stochastic()(ohlc)
    close_eq(out["stoch_k"], k)
    close_eq(out["stoch_d"], k.rolling(3).mean())
    close_eq(Donchian(period=20)(ohlc)["donchian_upper"], ohlc.high.rolling(20).max())
    ich = Ichimoku()(ohlc)
    conv = (ohlc.high.rolling(9).max() + ohlc.low.rolling(9).min()) / 2
    base = (ohlc.high.rolling(26).max() + ohlc.low.rolling(26).min()) / 2
    close_eq(ich["ichimoku_span_a"], (conv + base) / 2)


def test_registry_contains_builtins():
    for name in ("sma", "ema", "rsi", "macd", "bbands", "atr", "stoch", "donchian", "ichimoku"):
        assert name in INDICATORS


def test_unknown_param_rejected():
    with pytest.raises(ValueError):
        SMA(length=3)


@pytest.mark.skipif(ops.torch is None, reason="torch not installed")
def test_torch_backend_matches_numpy(ohlc):
    o_np, o_t = OHLC.from_frame(ohlc), OHLC.from_frame(ohlc, device="cpu")
    for cls in INDICATORS:
        p = {k: v.default for k, v in cls.params.items()}
        p = {k: int(v) if cls.params[k].integer else v for k, v in p.items()}
        a, b = cls.compute(o_np, **p), cls.compute(o_t, **p)
        for key in a:
            close_eq(b[key].numpy(), a[key])


def test_latch():
    enter = np.array([0, 1, 0, 0, 1, 0, 0], bool)
    exit_ = np.array([1, 0, 0, 1, 0, 0, 1], bool)
    np.testing.assert_array_equal(ops.latch(enter, exit_), [0, 1, 1, 0, 1, 1, 0])
    both = np.array([1, 1], bool)
    np.testing.assert_array_equal(ops.latch(both, both), [0, 0])


def test_batched_float_params_broadcast(ohlc):
    o = OHLC.from_frame(ohlc)
    rsi = RSI.compute(o, period=14)["value"]
    lowers = np.array([[20.0], [30.0], [40.0]])
    pos = ops.latch(rsi < lowers, rsi > 70)
    assert pos.shape == (3, len(ohlc))
    # lower entry threshold => never more time in the market than a higher one
    assert pos[0].sum() <= pos[2].sum()
