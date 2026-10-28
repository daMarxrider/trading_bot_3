# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Built-in technical indicators (all OHLC-only, backend-agnostic)."""

from __future__ import annotations

import numpy as np

from tradebot.core import ops
from tradebot.core.registry import IntParam, Param
from tradebot.indicators.base import BaseIndicator, Style


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
        # NaN (first bar) must stay NaN so the averages seed on the first real delta
        gain = ops.wilder(ops.where(delta < 0, 0.0, delta), period)
        loss = ops.wilder(ops.where(delta > 0, 0.0, -delta), period)
        with np.errstate(divide="ignore", invalid="ignore"):
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
        with np.errstate(divide="ignore", invalid="ignore"):
            k = 100 * (o.close - ll) / (hh - ll)
        return {"k": k, "d": ops.sma(k, smooth)}


# Custom Indicator made by daMarxrider
class StochasticShortWaves(BaseIndicator):
    """Stochastic Short Waves (SSW): the SMA of a fast (5) and a slow (14) RSI, with three signal levels.

    Custom Indicator made by daMarxrider, translated from the Pine Script study "Stochastic Short Waves"::

        period = input(22) · first_drop = input(66) · ath = input(83) · lower = input(35)
        plot(sma(rsi(close,5),period), color=yellow) · plot(sma(rsi(close,14),period), color=green)
        plot(lower, color=green) · plot(first_drop, color=orange) · plot(ath, color=red)

    Pine's ``rsi`` uses Wilder's smoothing like :class:`RSI`. The levels depend on parameters, so they're
    constant outputs rather than fixed ``levels``, which also shows their values in the chart's readout.
    """

    name = "ssw"
    description = "Stochastic Short Waves: smoothed fast and slow RSI with signal levels"
    # same order as the Pine inputs, so the chart reads "ssw 22 66 83 35"
    params = {
        "period": IntParam(22, 1, 200, "SMA length applied to both RSIs"),
        "first_drop": IntParam(66, 1, 100, "first-drop level (orange)"),
        "ath": IntParam(83, 1, 100, "all-time-high level (red)"),
        "lower": IntParam(35, 1, 100, "lower level (green)"),
    }
    outputs = ("fast", "slow", "lower", "first_drop", "ath")
    overlay = False
    # Pine's colours (TradingView palette): yellow, green, green, orange, red
    styles = {
        "fast": Style(color="#ffeb3b"),
        "slow": Style(color="#4caf50"),
        "lower": Style(color="#4caf50", width=1),
        "first_drop": Style(color="#ff9800", width=1),
        "ath": Style(color="#ff5252", width=1),
    }

    @classmethod
    def compute(cls, o, period, first_drop, ath, lower):
        def level(v):  # a constant line on the same backend and length as the data
            return o.close * 0.0 + v

        return {
            "fast": ops.sma(RSI.compute(o, period=5)["value"], period),  # Pine: sma(rsi(close,5), period)
            "slow": ops.sma(RSI.compute(o, period=14)["value"], period),  # Pine: sma(rsi(close,14), period)
            "lower": level(lower),
            "first_drop": level(first_drop),
            "ath": level(ath),
        }


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
