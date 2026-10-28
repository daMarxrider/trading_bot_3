# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Built-in technical indicators (all OHLC-only, backend-agnostic)."""

from __future__ import annotations

from tradebot.core import ops
from tradebot.core.registry import IntParam, Param
from tradebot.indicators.base import BaseIndicator


class SMA(BaseIndicator):
    name = "sma"
    description = "Simple moving average of the close"
    params = {"period": IntParam(20, 2, 400)}

    @classmethod
    def compute(cls, o, period):
        return {"value": ops.sma(o.close, period)}


class EMA(BaseIndicator):
    name = "ema"
    description = "Exponential moving average of the close"
    params = {"period": IntParam(20, 2, 400)}

    @classmethod
    def compute(cls, o, period):
        return {"value": ops.ema(o.close, period)}


class RSI(BaseIndicator):
    name = "rsi"
    description = "Relative Strength Index (Wilder)"
    params = {"period": IntParam(14, 2, 100)}
    overlay = False
    levels = (30.0, 70.0)

    @classmethod
    def compute(cls, o, period):
        delta = o.close - ops.shift(o.close, 1)
        gain = ops.wilder(ops.where(delta > 0, delta, 0.0), period)
        loss = ops.wilder(ops.where(delta < 0, -delta, 0.0), period)
        rsi = 100 - 100 / (1 + gain / loss)
        rsi = ops.where(loss == 0, 100.0, rsi)
        # Wilder's average is undefined until `period` deltas have been seen
        rsi = ops.where(ops.isnan(ops.shift(o.close, period)), ops.NAN, rsi)
        return {"value": rsi}


class MACD(BaseIndicator):
    name = "macd"
    description = "Moving Average Convergence/Divergence"
    params = {"fast": IntParam(12, 2, 100), "slow": IntParam(26, 3, 200), "signal": IntParam(9, 2, 100)}
    outputs = ("macd", "signal", "hist")
    overlay = False
    levels = (0.0,)

    @classmethod
    def compute(cls, o, fast, slow, signal):
        macd = ops.ema(o.close, fast) - ops.ema(o.close, slow)
        sig = ops.ema(macd, signal)
        return {"macd": macd, "signal": sig, "hist": macd - sig}


class Bollinger(BaseIndicator):
    name = "bbands"
    description = "Bollinger Bands"
    params = {"period": IntParam(20, 2, 200), "width": Param(2.0, 0.5, 4.0)}
    outputs = ("upper", "middle", "lower")

    @classmethod
    def compute(cls, o, period, width):
        mid = ops.sma(o.close, period)
        dev = ops.rolling_std(o.close, period) * width
        return {"upper": mid + dev, "middle": mid, "lower": mid - dev}


class ATR(BaseIndicator):
    name = "atr"
    description = "Average True Range (Wilder)"
    params = {"period": IntParam(14, 2, 100)}
    overlay = False

    @classmethod
    def compute(cls, o, period):
        prev = ops.shift(o.close, 1)
        tr = ops.maximum(o.high - o.low, ops.maximum(ops.abs_(o.high - prev), ops.abs_(o.low - prev)))
        tr = ops.where(ops.isnan(prev), o.high - o.low, tr)
        return {"value": ops.wilder(tr, period)}


class Stochastic(BaseIndicator):
    name = "stoch"
    description = "Stochastic oscillator %K/%D"
    params = {"period": IntParam(14, 2, 100), "smooth": IntParam(3, 1, 20)}
    outputs = ("k", "d")
    overlay = False
    levels = (20.0, 80.0)

    @classmethod
    def compute(cls, o, period, smooth):
        hh, ll = ops.rolling_max(o.high, period), ops.rolling_min(o.low, period)
        k = 100 * (o.close - ll) / (hh - ll)
        return {"k": k, "d": ops.sma(k, smooth)}


class Donchian(BaseIndicator):
    name = "donchian"
    description = "Donchian channel (highest high / lowest low)"
    params = {"period": IntParam(20, 2, 200)}
    outputs = ("upper", "lower")

    @classmethod
    def compute(cls, o, period):
        return {"upper": ops.rolling_max(o.high, period), "lower": ops.rolling_min(o.low, period)}


class Ichimoku(BaseIndicator):
    """Ichimoku cloud. Spans are *not* displaced forward so no future value leaks into a bar."""

    name = "ichimoku"
    description = "Ichimoku Kinko Hyo (conversion, base, span A/B)"
    params = {"conversion": IntParam(9, 2, 60), "base": IntParam(26, 5, 120), "span_b": IntParam(52, 10, 240)}
    outputs = ("conversion", "base", "span_a", "span_b")

    @classmethod
    def compute(cls, o, conversion, base, span_b):
        def mid(n):
            return (ops.rolling_max(o.high, n) + ops.rolling_min(o.low, n)) / 2

        conv, base_line = mid(conversion), mid(base)
        return {"conversion": conv, "base": base_line, "span_a": (conv + base_line) / 2, "span_b": mid(span_b)}
