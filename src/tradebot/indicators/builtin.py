# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Built-in technical indicators (backend-agnostic; the volume ones need volume data, #116)."""

from __future__ import annotations

import numpy as np

from tradebot.analysis import halving
from tradebot.core import ops
from tradebot.core.registry import IntParam, Param
from tradebot.indicators.base import BaseIndicator, Style


def _cross(a, b, up: bool):
    """Alert array: ``a`` crosses above (``up``) or below ``b`` on this bar; NaN until both have two values."""
    b = a * 0.0 + b  # a number becomes a constant line on a's backend
    pa, pb = ops.shift(a, 1), ops.shift(b, 1)
    return ops.flag(ops.cross_above(a, b) if up else ops.cross_below(a, b), a, b, pa, pb)


class SMA(BaseIndicator):
    name = "sma"
    description = "Simple moving average of the close"
    params = {"period": IntParam(20, 2, 400)}
    outputs = ("value", "cross_up", "cross_down")
    alerts = {"cross_up": "Close crosses above the average", "cross_down": "Close crosses below the average"}

    @classmethod
    def compute(cls, o, period):
        ma = ops.sma(o.close, period)
        return {"value": ma, "cross_up": _cross(o.close, ma, True), "cross_down": _cross(o.close, ma, False)}


class EMA(BaseIndicator):
    name = "ema"
    description = "Exponential moving average of the close"
    params = {"period": IntParam(20, 2, 400)}
    outputs = ("value", "cross_up", "cross_down")
    alerts = {"cross_up": "Close crosses above the average", "cross_down": "Close crosses below the average"}

    @classmethod
    def compute(cls, o, period):
        ma = ops.ema(o.close, period)
        return {"value": ma, "cross_up": _cross(o.close, ma, True), "cross_down": _cross(o.close, ma, False)}


class RSI(BaseIndicator):
    name = "rsi"
    description = "Relative Strength Index (Wilder)"
    params = {"period": IntParam(14, 2, 100)}
    overlay = False
    levels = (30.0, 70.0)
    outputs = ("value", "cross_up", "cross_down")
    alerts = {"cross_up": "RSI crosses above 70", "cross_down": "RSI crosses below 30"}

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
        return {"value": rsi, "cross_up": _cross(rsi, 70.0, True), "cross_down": _cross(rsi, 30.0, False)}


class MACD(BaseIndicator):
    name = "macd"
    description = "Moving Average Convergence/Divergence"
    params = {"fast": IntParam(12, 2, 100), "slow": IntParam(26, 3, 200), "signal": IntParam(9, 2, 100)}
    outputs = ("macd", "signal", "hist", "cross_up", "cross_down")
    overlay = False
    levels = (0.0,)
    alerts = {"cross_up": "MACD crosses above the signal line", "cross_down": "MACD crosses below the signal line"}

    @classmethod
    def compute(cls, o, fast, slow, signal):
        macd = ops.ema(o.close, fast) - ops.ema(o.close, slow)
        sig = ops.ema(macd, signal)
        return {"macd": macd, "signal": sig, "hist": macd - sig,
                "cross_up": _cross(macd, sig, True), "cross_down": _cross(macd, sig, False)}


class Bollinger(BaseIndicator):
    name = "bbands"
    description = "Bollinger Bands"
    params = {"period": IntParam(20, 2, 200), "width": Param(2.0, 0.5, 4.0)}
    outputs = ("upper", "middle", "lower", "above_upper", "below_lower")
    alerts = {"above_upper": "Close is above the upper band", "below_lower": "Close is below the lower band"}

    @classmethod
    def compute(cls, o, period, width):
        mid = ops.sma(o.close, period)
        dev = ops.rolling_std(o.close, period) * width
        upper, lower = mid + dev, mid - dev
        return {"upper": upper, "middle": mid, "lower": lower,
                "above_upper": ops.flag(o.close > upper, upper), "below_lower": ops.flag(o.close < lower, lower)}


class MACross(BaseIndicator):
    name = "ma_cross"
    description = "Fast and slow moving average (SMA or EMA) with cross alerts"
    params = {"fast": IntParam(9, 2, 200), "slow": IntParam(21, 3, 400),
              "ema": IntParam(0, 0, 1, "0: simple averages, 1: exponential")}
    outputs = ("fast", "slow", "cross_up", "cross_down")
    alerts = {"cross_up": "Fast MA crosses above slow MA", "cross_down": "Fast MA crosses below slow MA"}

    @classmethod
    def compute(cls, o, fast, slow, ema):
        avg = ops.ema if ema else ops.sma
        f, s = avg(o.close, fast), avg(o.close, slow)
        return {"fast": f, "slow": s, "cross_up": _cross(f, s, True), "cross_down": _cross(f, s, False)}


class ATR(BaseIndicator):
    name = "atr"
    description = "Average True Range (Wilder)"
    params = {"period": IntParam(14, 2, 100)}
    overlay = False
    units = {"value": "dist"}

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
    outputs = ("k", "d", "cross_up", "cross_down")
    overlay = False
    levels = (20.0, 80.0)
    alerts = {"cross_up": "%K crosses above %D", "cross_down": "%K crosses below %D"}

    @classmethod
    def compute(cls, o, period, smooth):
        hh, ll = ops.rolling_max(o.high, period), ops.rolling_min(o.low, period)
        with np.errstate(divide="ignore", invalid="ignore"):
            k = 100 * (o.close - ll) / (hh - ll)
        d = ops.sma(k, smooth)
        return {"k": k, "d": d, "cross_up": _cross(k, d, True), "cross_down": _cross(k, d, False)}


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
    outputs = ("upper", "lower", "breakout_up", "breakout_down")
    alerts = {"breakout_up": "Close breaks above the previous upper band",
              "breakout_down": "Close breaks below the previous lower band"}

    @classmethod
    def compute(cls, o, period):
        upper, lower = ops.rolling_max(o.high, period), ops.rolling_min(o.low, period)
        prev_up, prev_low = ops.shift(upper, 1), ops.shift(lower, 1)  # the band before this bar's own high/low
        return {"upper": upper, "lower": lower,
                "breakout_up": ops.flag(o.close > prev_up, prev_up),
                "breakout_down": ops.flag(o.close < prev_low, prev_low)}


class Ichimoku(BaseIndicator):
    """Ichimoku cloud. Spans are *not* displaced forward so no future value leaks into a bar."""

    name = "ichimoku"
    description = "Ichimoku Kinko Hyo (conversion, base, span A/B)"
    params = {
        "conversion": IntParam(9, 2, 60, "Candles for the fast midpoint line (highest high plus lowest low, halved)."),
        "base": IntParam(26, 5, 120, "Candles for the slower midpoint line. Span A, one edge of the cloud, is the "
                                     "average of the two."),
        "span_b": IntParam(52, 10, 240, "Candles for span B, the cloud's slowest edge."),
    }
    outputs = ("conversion", "base", "span_a", "span_b", "above_cloud", "below_cloud")
    alerts = {"above_cloud": "Close is above the cloud", "below_cloud": "Close is below the cloud"}

    @classmethod
    def compute(cls, o, conversion, base, span_b):
        def mid(n):
            return (ops.rolling_max(o.high, n) + ops.rolling_min(o.low, n)) / 2

        conv, base_line = mid(conversion), mid(base)
        span_a, b = (conv + base_line) / 2, mid(span_b)
        return {"conversion": conv, "base": base_line, "span_a": span_a, "span_b": b,
                "above_cloud": ops.flag(o.close > ops.maximum(span_a, b), span_a, b),
                "below_cloud": ops.flag(o.close < ops.minimum(span_a, b), span_a, b)}


class Halving(BaseIndicator):
    """Bitcoin halving cycle (#99): a calendar clock, so it needs timestamped candles (NaN without)."""

    name = "halving"
    description = "Days since the last Bitcoin halving, and until the predicted next one"
    outputs = ("days", "days_to_next", "phase")
    overlay = False
    #: the halving, the typical top (~530 days after) and bottom (~850) of past cycles, and the 4-year rule
    levels = (0.0, 530.0, 850.0, round(halving.NOMINAL_DAYS, 2))
    hidden = ("phase",)  # 0..1, its own scale; for strategies

    @classmethod
    def compute(cls, o):
        if o.time is None:
            nan = ops.full(tuple(o.close.shape), ops.NAN, o.close)
            return {k: nan for k in cls.outputs}
        return halving.cycle(o.time)


# -- volume (#116) ------------------------------------------------------------------------------------
# Volume is venue-dependent (see OHLC): these give NaN on symbols without volume data.
class Volume(BaseIndicator):
    name = "volume"
    description = "Volume bars (up/down by candle) with their moving average"
    params = {"period": IntParam(20, 2, 200, "candles of the average"),
              "spike": Param(2.0, 1.1, 10.0, "a spike is volume above this multiple of the average")}
    outputs = ("volume", "sma", "spike")
    overlay = False
    uses_volume = True
    units = {"volume": "volume", "sma": "volume"}
    styles = {"volume": Style(bars=True), "sma": Style(color="accent", width=1.2)}
    alerts = {"spike": "Volume spike: above N× its average"}

    @classmethod
    def compute(cls, o, period, spike):
        ma = ops.sma(o.volume, period)
        return {"volume": o.volume, "sma": ma, "spike": ops.flag(o.volume > spike * ma, ma, o.volume)}


class OBV(BaseIndicator):
    name = "obv"
    description = "On-balance volume: volume added on up closes, subtracted on down closes"
    params = {"period": IntParam(20, 2, 200, "candles of its moving average")}
    outputs = ("value", "sma", "cross_up", "cross_down")
    overlay = False
    uses_volume = True
    units = {"value": "volume", "sma": "volume"}
    styles = {"sma": Style(color="muted", width=1.2)}
    alerts = {"cross_up": "OBV crosses above its average", "cross_down": "OBV crosses below its average"}

    @classmethod
    def compute(cls, o, period):
        delta = o.close - ops.shift(o.close, 1)
        signed = ops.where(delta > 0, o.volume, ops.where(delta < 0, -o.volume, o.volume * 0.0))
        obv = ops.cumsum(signed)  # starts at 0 on the first bar (no previous close); NaN without volume
        ma = ops.sma(obv, period)
        return {"value": obv, "sma": ma, "cross_up": _cross(obv, ma, True), "cross_down": _cross(obv, ma, False)}


#: VWAP anchors: a new average starts with each of these periods (UTC)
ANCHORS = ("day", "week", "month", "year")


def _period_starts(time, anchor: int, like):
    """True on the first candle of each UTC day/week/month/year (on ``like``'s backend); NaN times never start one."""
    t = time.detach().cpu().numpy() if ops.is_torch(time) else np.asarray(time, dtype=float)
    ok = np.isfinite(t)
    stamp = np.where(ok, t, 0).astype("int64").astype("datetime64[s]")
    unit = ANCHORS[anchor]
    if unit == "week":  # weeks start on Monday; 1970-01-01 was a Thursday
        key = (stamp.astype("datetime64[D]").astype("int64") + 3) // 7
    else:
        key = stamp.astype({"day": "datetime64[D]", "month": "datetime64[M]", "year": "datetime64[Y]"}[unit])
        key = key.astype("int64")
    prev = np.concatenate([np.full(key.shape[:-1] + (1,), -1), key[..., :-1]], axis=-1)
    start = (key != prev) & ok
    start[..., 0] = True
    return ops.asarray(start, like) > 0


class VWAP(BaseIndicator):
    name = "vwap"
    description = "Volume-weighted average price, restarting every day, week, month or year (UTC)"
    params = {"anchor": IntParam(0, 0, 3, "a new average starts each 0: day, 1: week, 2: month, 3: year")}
    outputs = ("value", "cross_up", "cross_down")
    uses_volume = True
    alerts = {"cross_up": "Close crosses above the VWAP", "cross_down": "Close crosses below the VWAP"}

    @classmethod
    def compute(cls, o, anchor):
        v = o.volume
        if o.time is None:  # no timestamps: one average from the first candle
            start = ops.asarray(np.zeros(o.close.shape[-1]), o.close) > 0
        else:
            start = _period_starts(o.time, int(anchor), o.close)
        weighted = ops.cumsum_since(ops.nan_to_num(o.typical * v), start)
        total = ops.cumsum_since(ops.nan_to_num(v), start)
        with np.errstate(divide="ignore", invalid="ignore"):
            vwap = ops.where(total > 0, weighted / total, ops.NAN)
        return {"value": vwap, "cross_up": _cross(o.close, vwap, True), "cross_down": _cross(o.close, vwap, False)}


class MFI(BaseIndicator):
    name = "mfi"
    description = "Money Flow Index: a volume-weighted RSI of the typical price"
    params = {"period": IntParam(14, 2, 100)}
    outputs = ("value", "cross_up", "cross_down")
    overlay = False
    uses_volume = True
    levels = (20.0, 80.0)
    alerts = {"cross_up": "MFI crosses above 80", "cross_down": "MFI crosses below 20"}

    @classmethod
    def compute(cls, o, period):
        tp = o.typical
        prev = ops.shift(tp, 1)
        flow = tp * o.volume
        first = ops.isnan(prev)  # the first candle has no direction: NaN, so the average waits for `period` flows
        up = ops.where(first, ops.NAN, ops.where(tp > prev, flow, 0.0))
        down = ops.where(first, ops.NAN, ops.where(tp < prev, flow, 0.0))
        pos, neg = ops.sma(up, period), ops.sma(down, period)  # the ratio of the averages is that of the sums
        with np.errstate(divide="ignore", invalid="ignore"):
            mfi = 100 - 100 / (1 + pos / neg)
        mfi = ops.where((neg == 0) & ~ops.isnan(pos), 100.0, mfi)
        return {"value": mfi, "cross_up": _cross(mfi, 80.0, True), "cross_down": _cross(mfi, 20.0, False)}


class VolumeRatio(BaseIndicator):
    name = "volume_ratio"
    description = "Volume relative to its moving average (2 = twice the usual volume)"
    params = {"period": IntParam(20, 2, 200, "candles of the average"),
              "spike": Param(2.0, 1.1, 10.0, "a spike is a ratio above this")}
    outputs = ("value", "spike")
    overlay = False
    uses_volume = True
    levels = (1.0,)
    units = {"value": "ratio"}
    alerts = {"spike": "Volume above N× its average"}

    @classmethod
    def compute(cls, o, period, spike):
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = o.volume / ops.sma(o.volume, period)
        return {"value": ratio, "spike": ops.flag(ratio > spike, ratio)}
