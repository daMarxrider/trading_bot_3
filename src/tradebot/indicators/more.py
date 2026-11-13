# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""More built-in indicators (#162): trend (ADX, Supertrend, Parabolic SAR), channels (Keltner), oscillators
(CCI, Williams %R, ROC, momentum), calendar pivots, a rolling volume profile and an anchored VWAP.

The recursive ones (Supertrend, SAR) and the windowed ones (CCI, volume profile, pivots) run in NumPy whatever the
backend of the candles and hand the result back on it, so CPU and GPU agree. Everything uses only the current and
earlier bars. The volume ones give NaN on symbols without volume data (#116).
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from tradebot.core import ops
from tradebot.core.registry import IntParam, Param
from tradebot.indicators.base import BaseIndicator, Style
from tradebot.indicators.builtin import ATR, _cross, _period_starts


def _np(x) -> np.ndarray:
    return x.detach().cpu().numpy() if ops.is_torch(x) else np.asarray(x, dtype=float)


def _rows(x) -> tuple[np.ndarray, tuple]:
    """``x`` as a NumPy ``(rows, time)`` array, and the original shape to restore."""
    a = _np(x)
    return a.reshape(-1, a.shape[-1]), a.shape


def _back(a: np.ndarray, shape: tuple, like):
    return ops.asarray(a.reshape(shape), like)


# -- trend ------------------------------------------------------------------------------------------
class ADX(BaseIndicator):
    name = "adx"
    description = "Average Directional Index (Wilder): trend strength with +DI / -DI"
    params = {"period": IntParam(14, 2, 100, "candles of the smoothing; the ADX needs about twice that to settle")}
    outputs = ("adx", "plus_di", "minus_di", "cross_up", "cross_down")
    overlay = False
    levels = (25.0,)
    styles = {"adx": Style(color="text", width=2), "plus_di": Style(color="up", width=1.2),
              "minus_di": Style(color="down", width=1.2)}
    alerts = {"cross_up": "+DI crosses above -DI (uptrend taking over)",
              "cross_down": "+DI crosses below -DI (downtrend taking over)"}

    @classmethod
    def compute(cls, o, period):
        prev_h, prev_l, prev_c = ops.shift(o.high, 1), ops.shift(o.low, 1), ops.shift(o.close, 1)
        up, down = o.high - prev_h, prev_l - o.low  # NaN on the first bar
        plus_dm = ops.where(ops.isnan(up), ops.NAN, ops.where((up > down) & (up > 0), up, 0.0))
        minus_dm = ops.where(ops.isnan(up), ops.NAN, ops.where((down > up) & (down > 0), down, 0.0))
        tr = ops.maximum(o.high - o.low, ops.maximum(ops.abs_(o.high - prev_c), ops.abs_(o.low - prev_c)))
        tr = ops.where(ops.isnan(prev_c), o.high - o.low, tr)
        atr = ops.wilder(tr, period)
        with np.errstate(divide="ignore", invalid="ignore"):
            pdi = 100 * ops.wilder(plus_dm, period) / atr
            mdi = 100 * ops.wilder(minus_dm, period) / atr
            dx = 100 * ops.abs_(pdi - mdi) / (pdi + mdi)
        dx = ops.where((pdi + mdi) == 0, 0.0, dx)
        adx = ops.wilder(dx, period)
        # the DIs are defined after `period` bars of movement, the ADX after another `period`
        pdi = ops.where(ops.isnan(ops.shift(o.close, period)), ops.NAN, pdi)
        mdi = ops.where(ops.isnan(ops.shift(o.close, period)), ops.NAN, mdi)
        adx = ops.where(ops.isnan(ops.shift(o.close, 2 * period - 1)), ops.NAN, adx)
        return {"adx": adx, "plus_di": pdi, "minus_di": mdi,
                "cross_up": _cross(pdi, mdi, True), "cross_down": _cross(pdi, mdi, False)}


def _supertrend(h, l, c, atr, period: int, mult: float):
    """Supertrend on ``(rows, time)`` arrays: the active band and its direction (+1 up, -1 down)."""
    B, T = c.shape
    hl2 = (h + l) / 2
    ub, lb = hl2 + mult * atr, hl2 - mult * atr
    line = np.full((B, T), np.nan)
    direction = np.full((B, T), np.nan)
    fub = flb = d = None
    for t in range(max(period - 1, 0), T):
        if fub is None:
            fub, flb, d = ub[:, t].copy(), lb[:, t].copy(), np.ones(B)
        else:
            pc = c[:, t - 1]
            fub = np.where((ub[:, t] < fub) | (pc > fub), ub[:, t], fub)
            flb = np.where((lb[:, t] > flb) | (pc < flb), lb[:, t], flb)
            d = np.where(d < 0, np.where(c[:, t] > fub, 1.0, -1.0), np.where(c[:, t] < flb, -1.0, 1.0))
        direction[:, t] = d
        line[:, t] = np.where(d > 0, flb, fub)
    return line, direction


def _split(line, direction, like, shape):
    """A stop line as two outputs, below the price (uptrend) and above it (downtrend), with a gap at each flip."""
    up = _back(np.where(direction > 0, line, np.nan), shape, like)
    down = _back(np.where(direction < 0, line, np.nan), shape, like)
    return up, down, _back(direction, shape, like)


def _flips(direction):
    prev = ops.shift(direction, 1)
    return (ops.flag((direction > 0) & (prev < 0), direction, prev),
            ops.flag((direction < 0) & (prev > 0), direction, prev))


class Supertrend(BaseIndicator):
    name = "supertrend"
    description = "Supertrend: an ATR band that flips sides when the close breaks through it"
    params = {"period": IntParam(10, 2, 100, "candles of the ATR"),
              "mult": Param(3.0, 0.5, 10.0, "band distance from the candle's midpoint, in ATRs")}
    outputs = ("up", "down", "direction", "flip_up", "flip_down")
    hidden = ("direction",)  # +1 uptrend, -1 downtrend; for strategies
    units = {"direction": "direction"}
    styles = {"up": Style(color="up", width=1.6), "down": Style(color="down", width=1.6)}
    alerts = {"flip_up": "Supertrend turns up (close breaks above the band)",
              "flip_down": "Supertrend turns down (close breaks below the band)"}

    @classmethod
    def compute(cls, o, period, mult):
        atr, shape = _rows(ATR.compute(o, period)["value"])
        line, d = _supertrend(_rows(o.high)[0], _rows(o.low)[0], _rows(o.close)[0], atr, period, mult)
        up, down, direction = _split(line, d, o.close, shape)
        flip_up, flip_down = _flips(direction)
        return {"up": up, "down": down, "direction": direction, "flip_up": flip_up, "flip_down": flip_down}


def _sar(h, l, c, step: float, max_af: float):
    """Wilder's Parabolic SAR on ``(rows, time)`` arrays: the stop and the trend (+1 up, -1 down)."""
    B, T = c.shape
    sar = np.full((B, T), np.nan)
    trend = np.full((B, T), np.nan)
    if T < 2:
        return sar, trend
    up = c[:, 1] >= c[:, 0]
    s = np.where(up, l[:, 0], h[:, 0])
    ep = np.where(up, h[:, 1], l[:, 1])
    af = np.full(B, step)
    sar[:, 1], trend[:, 1] = s, np.where(up, 1.0, -1.0)
    for t in range(2, T):
        s = s + af * (ep - s)
        s = np.where(up, np.minimum(s, np.minimum(l[:, t - 1], l[:, t - 2])),
                     np.maximum(s, np.maximum(h[:, t - 1], h[:, t - 2])))
        flip = np.where(up, l[:, t] < s, h[:, t] > s)
        new_ep = np.where(up, np.maximum(ep, h[:, t]), np.minimum(ep, l[:, t]))
        af = np.where(flip, step, np.where(new_ep != ep, np.minimum(af + step, max_af), af))
        s = np.where(flip, ep, s)  # a reversal starts at the old extreme point
        ep = np.where(flip, np.where(up, l[:, t], h[:, t]), new_ep)
        up = np.where(flip, ~up, up)
        sar[:, t], trend[:, t] = s, np.where(up, 1.0, -1.0)
    return sar, trend


class ParabolicSAR(BaseIndicator):
    name = "psar"
    description = "Parabolic SAR (Wilder): a trailing stop that accelerates towards the price"
    params = {"step": Param(0.02, 0.005, 0.1, "acceleration added each time the trend makes a new extreme"),
              "max_af": Param(0.2, 0.05, 0.5, "the acceleration never grows beyond this")}
    outputs = ("up", "down", "trend", "flip_up", "flip_down")
    hidden = ("trend",)  # +1 uptrend, -1 downtrend; for strategies
    units = {"trend": "direction"}
    styles = {"up": Style(color="up", width=1.6, dash="dot"), "down": Style(color="down", width=1.6, dash="dot")}
    alerts = {"flip_up": "SAR flips below the price (uptrend starts)",
              "flip_down": "SAR flips above the price (downtrend starts)"}

    @classmethod
    def compute(cls, o, step, max_af):
        c, shape = _rows(o.close)
        sar, trend = _sar(_rows(o.high)[0], _rows(o.low)[0], c, step, max_af)
        up, down, direction = _split(sar, trend, o.close, shape)
        flip_up, flip_down = _flips(direction)
        return {"up": up, "down": down, "trend": direction, "flip_up": flip_up, "flip_down": flip_down}


# -- channels and oscillators -----------------------------------------------------------------------
class Keltner(BaseIndicator):
    name = "keltner"
    description = "Keltner channel: an EMA with bands a multiple of the ATR away"
    params = {"period": IntParam(20, 2, 200, "candles of the EMA of the close"),
              "mult": Param(2.0, 0.5, 6.0, "band distance in ATRs"),
              "atr_period": IntParam(10, 2, 100, "candles of the ATR")}
    outputs = ("upper", "middle", "lower", "above_upper", "below_lower")
    alerts = {"above_upper": "Close is above the upper channel", "below_lower": "Close is below the lower channel"}
    fills = (("upper", "lower", Style(color="rgba(255, 209, 102, 0.08)")),)

    @classmethod
    def compute(cls, o, period, mult, atr_period):
        mid = ops.ema(o.close, period)
        band = ATR.compute(o, atr_period)["value"] * mult
        warm = ops.shift(o.close, max(period, atr_period) - 1)  # NaN until both averages have a full window
        mid = ops.where(ops.isnan(warm), ops.NAN, mid)
        upper, lower = mid + band, mid - band
        return {"upper": upper, "middle": mid, "lower": lower,
                "above_upper": ops.flag(o.close > upper, upper), "below_lower": ops.flag(o.close < lower, lower)}


class CCI(BaseIndicator):
    name = "cci"
    description = "Commodity Channel Index: how far the typical price is from its average, in mean deviations"
    params = {"period": IntParam(20, 2, 200)}
    outputs = ("value", "cross_up", "cross_down")
    overlay = False
    levels = (-100.0, 100.0)
    units = {"value": "cci"}
    alerts = {"cross_up": "CCI crosses above +100", "cross_down": "CCI crosses below -100"}

    @classmethod
    def compute(cls, o, period):
        tp, shape = _rows(o.typical)
        out = np.full(tp.shape, np.nan)
        if tp.shape[-1] >= period:
            w = sliding_window_view(tp, period, axis=-1)
            mean = w.mean(-1)
            dev = np.abs(w - mean[..., None]).mean(-1)  # mean absolute deviation
            with np.errstate(divide="ignore", invalid="ignore"):
                cci = (tp[:, period - 1:] - mean) / (0.015 * dev)
            out[:, period - 1:] = np.where(dev == 0, 0.0, cci)
        cci = _back(out, shape, o.close)
        return {"value": cci, "cross_up": _cross(cci, 100.0, True), "cross_down": _cross(cci, -100.0, False)}


class WilliamsR(BaseIndicator):
    name = "williams_r"
    description = "Williams %R: where the close sits in the recent range, from -100 (low) to 0 (high)"
    params = {"period": IntParam(14, 2, 100)}
    outputs = ("value", "cross_up", "cross_down")
    overlay = False
    levels = (-80.0, -20.0)
    units = {"value": "willr"}
    alerts = {"cross_up": "%R crosses above -20 (overbought)", "cross_down": "%R crosses below -80 (oversold)"}

    @classmethod
    def compute(cls, o, period):
        hh, ll = ops.rolling_max(o.high, period), ops.rolling_min(o.low, period)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = -100 * (hh - o.close) / (hh - ll)
        return {"value": r, "cross_up": _cross(r, -20.0, True), "cross_down": _cross(r, -80.0, False)}


class ROC(BaseIndicator):
    name = "roc"
    description = "Rate of change: the close's percentage change over N candles"
    params = {"period": IntParam(12, 1, 200, "candles to look back")}
    outputs = ("value", "cross_up", "cross_down")
    overlay = False
    levels = (0.0,)
    units = {"value": "percent"}
    alerts = {"cross_up": "ROC crosses above zero", "cross_down": "ROC crosses below zero"}

    @classmethod
    def compute(cls, o, period):
        with np.errstate(divide="ignore", invalid="ignore"):
            roc = (o.close / ops.shift(o.close, period) - 1) * 100
        return {"value": roc, "cross_up": _cross(roc, 0.0, True), "cross_down": _cross(roc, 0.0, False)}


class Momentum(BaseIndicator):
    name = "momentum"
    description = "Momentum: the close minus the close N candles ago"
    params = {"period": IntParam(10, 1, 200, "candles to look back")}
    outputs = ("value", "cross_up", "cross_down")
    overlay = False
    levels = (0.0,)
    units = {"value": "dist"}
    alerts = {"cross_up": "Momentum crosses above zero", "cross_down": "Momentum crosses below zero"}

    @classmethod
    def compute(cls, o, period):
        m = o.close - ops.shift(o.close, period)
        return {"value": m, "cross_up": _cross(m, 0.0, True), "cross_down": _cross(m, 0.0, False)}


# -- levels -----------------------------------------------------------------------------------------
def _previous_period(h, l, c, start):
    """For one series: each bar's previous period (high, low, close) as three arrays; NaN in the first period."""
    T = len(h)
    out = np.full((3, T), np.nan)
    first = np.flatnonzero(start)
    if len(first) < 2:
        return out
    base = first[0]
    ends = np.append(first[1:], T)
    stats = np.stack([np.maximum.reduceat(h[base:], first - base), np.minimum.reduceat(l[base:], first - base),
                      c[ends - 1]])  # each period's high, low and last close
    out[:, base:] = np.repeat(np.concatenate([np.full((3, 1), np.nan), stats[:, :-1]], axis=1), ends - first, axis=1)
    return out


class Pivots(BaseIndicator):
    """Classic pivot points from the previous day/week/month/year's high, low and close (UTC).

    A calendar clock like the VWAP: it needs timestamped candles (NaN without) and is NaN during the first period.
    """

    name = "pivots"
    description = "Classic pivot points (P, R1, R2, S1, S2) from the previous day, week, month or year (UTC)"
    params = {"anchor": IntParam(0, 0, 3, "pivots of the previous 0: day, 1: week, 2: month, 3: year")}
    outputs = ("p", "r1", "r2", "s1", "s2", "above_r1", "below_s1")
    styles = {"p": Style(color="accent", width=1.2, dash="dash"),
              "r1": Style(color="down", width=1, dash="dot"), "r2": Style(color="down", width=1, dash="dot"),
              "s1": Style(color="up", width=1, dash="dot"), "s2": Style(color="up", width=1, dash="dot")}
    alerts = {"above_r1": "Close is above the first resistance (R1)", "below_s1": "Close is below the first support (S1)"}

    @classmethod
    def compute(cls, o, anchor):
        h, shape = _rows(o.high)
        l, c = _rows(o.low)[0], _rows(o.close)[0]
        prev = np.full((3, *h.shape), np.nan)
        if o.time is not None:
            start = np.broadcast_to(_np(_period_starts(o.time, int(anchor), o.close)).astype(bool).reshape(-1, h.shape[-1]),
                                    h.shape)
            for i in range(h.shape[0]):
                prev[:, i] = _previous_period(h[i], l[i], c[i], start[i])
        ph, pl, pc = prev
        p = (ph + pl + pc) / 3
        levels = {"p": p, "r1": 2 * p - pl, "r2": p + (ph - pl), "s1": 2 * p - ph, "s2": p - (ph - pl)}
        out = {k: _back(v, shape, o.close) for k, v in levels.items()}
        out["above_r1"] = ops.flag(o.close > out["r1"], out["r1"])
        out["below_s1"] = ops.flag(o.close < out["s1"], out["s1"])
        return out


# -- volume (#116) ----------------------------------------------------------------------------------
def _volume_profile(tp, v, h, l, n: int, bins: int, area: float):
    """Rolling volume profile on ``(rows, time)`` arrays: POC, value area high and low over each ``n``-candle window.

    A candle's volume goes into the bin of its typical price; the bins split the window's low-to-high range. The
    value area grows from the POC one bin at a time, taking the busier neighbour, until it holds ``area`` of the volume.
    """
    B, T = tp.shape
    poc, vah, val = (np.full((B, T), np.nan) for _ in range(3))
    if T < n:
        return poc, vah, val
    win = [sliding_window_view(x, n, axis=-1).reshape(-1, n) for x in (tp, v, h, l)]
    R = win[0].shape[0]
    res = np.full((3, R), np.nan)
    for a in range(0, R, 4096):
        wtp, wv, wh, wl = (w[a:a + 4096] for w in win)
        r = len(wtp)
        ok = ~np.isnan(wv).any(1) & ~np.isnan(wtp).any(1) & (wv.sum(1) > 0)
        lo, hi = wl.min(1), wh.max(1)
        width = np.where(hi > lo, (hi - lo) / bins, 1.0)
        idx = np.clip(np.nan_to_num((wtp - lo[:, None]) / width[:, None]).astype(int), 0, bins - 1)
        hist = np.bincount((np.arange(r)[:, None] * bins + idx).ravel(), weights=np.nan_to_num(wv).ravel(),
                           minlength=r * bins).reshape(r, bins)
        k = hist.argmax(1)
        lo_i, hi_i, acc = k.copy(), k.copy(), hist[np.arange(r), k]
        target = area * hist.sum(1)
        pad = np.full((r, 1), -1.0)
        padded = np.concatenate([pad, hist, pad], axis=1)  # padded[:, j + 1] is bin j; -1 marks the edges
        for _ in range(bins):
            go = acc < target
            above, below = padded[np.arange(r), hi_i + 2], padded[np.arange(r), lo_i]
            go &= (above >= 0) | (below >= 0)
            take_up = above >= below
            hi_i = np.where(go & take_up, hi_i + 1, hi_i)
            lo_i = np.where(go & ~take_up, lo_i - 1, lo_i)
            acc = acc + np.where(go, np.where(take_up, above, below), 0.0)
        res[0, a:a + r] = np.where(ok, lo + (k + 0.5) * width, np.nan)
        res[1, a:a + r] = np.where(ok, lo + (hi_i + 1) * width, np.nan)
        res[2, a:a + r] = np.where(ok, lo + lo_i * width, np.nan)
    for out, src in zip((poc, vah, val), res):
        out[:, n - 1:] = src.reshape(B, T - n + 1)
    return poc, vah, val


class VolumeProfile(BaseIndicator):
    """Volume profile over a fixed rolling window (the last ``period`` candles), not the chart's visible range."""

    name = "volume_profile"
    description = "Rolling volume profile: point of control and value area of the last N candles"
    params = {"period": IntParam(100, 10, 1000, "candles of the window"),
              "bins": IntParam(24, 5, 200, "price bins the window's range is split into"),
              "value_area": Param(0.7, 0.5, 0.95, "share of the volume the value area holds")}
    outputs = ("poc", "vah", "val", "above_value", "below_value")
    uses_volume = True
    styles = {"poc": Style(color="accent", width=1.6), "vah": Style(color="muted", width=1, dash="dash"),
              "val": Style(color="muted", width=1, dash="dash")}
    fills = (("vah", "val", Style(color="rgba(128, 128, 128, 0.08)")),)
    alerts = {"above_value": "Close is above the value area", "below_value": "Close is below the value area"}

    @classmethod
    def compute(cls, o, period, bins, value_area):
        tp, shape = _rows(o.typical)
        rows = [_rows(x)[0] for x in (o.volume, o.high, o.low)]
        poc, vah, val = (_back(a, shape, o.close) for a in _volume_profile(tp, *rows, period, bins, value_area))
        return {"poc": poc, "vah": vah, "val": val,
                "above_value": ops.flag(o.close > vah, vah), "below_value": ops.flag(o.close < val, val)}


class AnchoredVWAP(BaseIndicator):
    """VWAP from a fixed start date. The chart has no anchor picker, so the anchor is a parameter."""

    name = "avwap"
    description = "Anchored VWAP: volume-weighted average price since a chosen date (UTC)"
    params = {"anchor": IntParam(0, 0, 21001231, "start date as YYYYMMDD (UTC); 0: from the first candle")}
    outputs = ("value", "cross_up", "cross_down")
    uses_volume = True
    alerts = {"cross_up": "Close crosses above the anchored VWAP", "cross_down": "Close crosses below the anchored VWAP"}

    @classmethod
    def compute(cls, o, anchor):
        active = ~ops.isnan(o.close)
        if int(anchor) <= 0:
            pass
        elif o.time is None:
            active = active & False  # a dated anchor needs timestamps
        else:
            y, m, d = anchor // 10000, anchor // 100 % 100, anchor % 100
            try:
                day = np.datetime64(f"{y:04d}-{m:02d}-{d:02d}", "s")
            except ValueError:  # e.g. 20240230: take the month's first day
                day = np.datetime64(f"{y:04d}-{min(max(m, 1), 12):02d}-01", "s")
            active = active & (o.time >= float(day.astype("int64")))
        first = active & ~(ops.shift(active * 1.0, 1, fill=0.0) > 0)
        mask = active * 1.0
        weighted = ops.cumsum_since(ops.nan_to_num(o.typical * o.volume) * mask, first)
        total = ops.cumsum_since(ops.nan_to_num(o.volume) * mask, first)
        with np.errstate(divide="ignore", invalid="ignore"):
            vwap = ops.where(active & (total > 0), weighted / total, ops.NAN)
        vwap = ops.where(ops.isnan(o.volume), ops.NAN, vwap)
        return {"value": vwap, "cross_up": _cross(o.close, vwap, True), "cross_down": _cross(o.close, vwap, False)}
