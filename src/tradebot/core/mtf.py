# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Multi-timeframe values (#161): an indicator or strategy computed on higher-timeframe candles and lined up
with the candles it runs on, without look-ahead.

The candles are grouped into the higher timeframe's calendar buckets (UTC: ``4h`` from midnight, ``1d`` days,
``1w`` weeks from Monday, ``1mo`` calendar months) and aggregated: first open, highest high, lowest low, last
close, summed volume. A higher-timeframe candle is used from the first candle whose close (its open time plus
the usual spacing of the candles) reaches the end of the bucket, and never before the last candle of the bucket:
on a 1h chart the daily SMA changes on the 23:00 candle, which closes at midnight, and holds until the next one.
Before the first closed bucket the value is NaN (still warming up).

Where the candles end before the bucket does (a stock's last hourly candle closes at 16:00, its day at midnight)
the bucket is used from the next candle on: one candle later than strictly possible, never early.

::

    h = Higher(o, "1d")
    daily_sma = h.map(lambda d: ops.sma(d.close, 200))     # on o's candles
    compute(SMA, o, "1d", period=200)                        # {output: array}, the same for an indicator
"""

from __future__ import annotations

import numpy as np

from tradebot.core import ops
from tradebot.core.series import OHLC, unix_seconds
from tradebot.core.timeframes import TIMEFRAMES, to_timedelta

_DAY = 86_400.0


def check(tf: str) -> str:
    """``tf`` if it is a known timeframe, else ``ValueError``."""
    if tf not in TIMEFRAMES:
        raise ValueError(f"unknown timeframe tf={tf!r}; use one of {', '.join(TIMEFRAMES)}")
    return tf


def seconds(tf: str) -> float:
    """The shortest length of a ``tf`` candle in seconds (a month: 28 days)."""
    return 28 * _DAY if tf == "1mo" else to_timedelta(tf).total_seconds()


def buckets(time: np.ndarray, tf: str) -> tuple[np.ndarray, np.ndarray]:
    """``(start, end)`` in Unix seconds of the ``tf`` bucket each time (Unix seconds) falls into."""
    if tf == "1mo":
        month = time.astype(np.int64).astype("datetime64[s]").astype("datetime64[M]")
        sec = lambda m: m.astype("datetime64[s]").astype(np.int64).astype(np.float64)  # noqa: E731
        return sec(month), sec(month + 1)
    step = to_timedelta(tf).total_seconds()
    origin = 4 * _DAY if tf == "1w" else 0.0  # 1970-01-05 was a Monday
    start = np.floor((time - origin) / step) * step + origin
    return start, start + step


def spacing(time: np.ndarray) -> float:
    """The usual distance between candles in seconds (the median of the gaps; 0 for a single candle)."""
    gaps = np.diff(time)
    gaps = gaps[gaps > 0]
    return float(np.median(gaps)) if gaps.size else 0.0


def _np(x):
    return x.detach().cpu().numpy() if ops.is_torch(x) else np.asarray(x)


def _resample(fields: list, tf: str) -> tuple[OHLC, np.ndarray]:
    """The higher candles of one series, and per candle the index of the last higher candle closed at its close."""
    open_, high, low, close, time, volume = fields
    if time is None or not np.isfinite(time).all():
        raise ValueError(f"tf={tf} needs candles with timestamps")
    dt = spacing(time)
    if dt > seconds(tf) * (1 + 1e-9):
        raise ValueError(f"tf={tf} is shorter than the candles; use a longer timeframe")
    start, end = buckets(time, tf)
    first = np.flatnonzero(np.r_[True, start[1:] != start[:-1]])
    last = np.r_[first[1:] - 1, len(time) - 1]
    red = lambda f, x: f.reduceat(x, first, axis=-1)  # noqa: E731
    higher = OHLC(open_[..., first], red(np.maximum, high), red(np.minimum, low), close[..., last], start[first],
                  red(np.add, volume))
    at = np.searchsorted(end[first], time + dt, side="right") - 1
    bucket = np.cumsum(np.r_[True, start[1:] != start[:-1]]) - 1
    is_last = np.zeros(len(time), bool)
    is_last[last] = True
    return higher, np.minimum(at, bucket - (~is_last).astype(int))  # never a bucket that later candles still belong to


class Higher:
    """The ``tf`` candles of ``o`` (any leading dimensions, NumPy or torch) and how they line up with ``o``."""

    def __init__(self, o: OHLC, tf: str):
        self.tf = check(tf)
        self.like = o.close
        fields = [None if f is None else _np(f).astype(np.float64) for f in o.fields()]
        time = fields[4]
        if time is None:
            raise ValueError(f"tf={tf} needs candles with timestamps")
        self.lead, self.length = time.shape[:-1], time.shape[-1]
        self.rows = []  # (leading index, higher OHLC, lineup)
        for ix in np.ndindex(*self.lead):
            sub = [None if f is None else f[ix] if f.ndim == len(self.lead) + 1 else f for f in fields]
            self.rows.append((ix, *_resample(sub, tf)))

    def map(self, fn):
        """``fn(higher OHLC)`` -> an array or ``{name: array}`` on the higher candles; the same on ``o``'s
        candles (float, NaN before the first closed higher candle), in ``o``'s backend."""
        outs = [(ix, fn(h), at) for ix, h, at in self.rows]
        if isinstance(outs[0][1], dict):
            return {k: self._join([(ix, r[k], at) for ix, r, at in outs]) for k in outs[0][1]}
        return self._join(outs)

    def _join(self, parts):
        mapped = []
        for ix, x, at in parts:
            x = np.asarray(_np(x), dtype=np.float64)
            if x.ndim == 0:  # a constant
                x = np.full(at.max(initial=-1) + 2, float(x))
            pad = np.concatenate([np.full(x.shape[:-1] + (1,), np.nan), x], axis=-1)
            mapped.append((ix, pad[..., at + 1]))
        if not self.lead:
            out = mapped[0][1]
        else:
            shape = np.broadcast_shapes(self.lead + (self.length,), *(m.shape for _, m in mapped))
            out = np.full(shape, np.nan)
            for ix, m in mapped:
                sel = [slice(None)] * len(shape)
                for d, (i, size) in enumerate(zip(ix, self.lead)):
                    if size > 1:
                        sel[len(shape) - 1 - len(self.lead) + d] = i
                out[tuple(sel)] = m
        return ops.asarray(out, self.like) if ops.is_torch(self.like) else out


def numpy_params(params: dict) -> dict:
    """Parameters with torch tensors as NumPy arrays (the higher candles are NumPy)."""
    return {k: _np(v) if ops.is_torch(v) else v for k, v in params.items()}


def compute(cls, o: OHLC, tf: str, **params) -> dict:
    """``cls.compute`` (an indicator class) on the ``tf`` candles of ``o``, lined up with ``o``."""
    params = numpy_params(params)
    return Higher(o, tf).map(lambda h: cls.compute(h, **params))


def closed(df, tf: str, now: float):
    """A ``tf`` candle frame without its last candle while that one is still open at ``now`` (Unix seconds)."""
    if df.empty:
        return df
    t = unix_seconds(df.index[-1:])
    if t is None:
        return df
    _, end = buckets(t, check(tf))
    return df.iloc[:-1] if end[0] > now else df
