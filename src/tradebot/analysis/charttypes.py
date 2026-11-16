# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Chart types that are display transforms of candles (#163): Heikin-Ashi, Renko and percent-change comparison.

None of these feed strategies, indicators or backtests: those keep using the real candles. The functions take and
return candle frames (``open``/``high``/``low``/``close`` columns on a time index) like the rest of the code."""

from __future__ import annotations

import numpy as np
import pandas as pd

TYPES = ("candles", "line", "heikin-ashi", "renko")


def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """Heikin-Ashi candles: the average of open/high/low/close as the close, and an open that is the midpoint of the
    previous Heikin-Ashi body, so trends show as runs of same-coloured candles. Same index; volume is kept."""
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    close = (o + h + l + c) / 4
    open_ = np.empty_like(close)
    if len(close):
        open_[0] = (o[0] + c[0]) / 2
        for i in range(1, len(close)):
            open_[i] = (open_[i - 1] + close[i - 1]) / 2
    out = df.copy()
    out["open"], out["close"] = open_, close
    out["high"], out["low"] = np.maximum.reduce([h, open_, close]), np.minimum.reduce([l, open_, close])
    return out


def atr_brick(df: pd.DataFrame, period: int = 14) -> float:
    """A brick size for :func:`renko`: the mean true range of the last ``period`` candles."""
    h, l, c = (df[k].to_numpy(float) for k in ("high", "low", "close"))
    prev = np.r_[c[0], c[:-1]]
    tr = np.maximum.reduce([h - l, np.abs(h - prev), np.abs(l - prev)])
    return float(np.nanmean(tr[-period:])) if len(tr) else 0.0


def renko(df: pd.DataFrame, brick: float | None = None, atr_period: int = 14) -> pd.DataFrame:
    """Renko bricks from the closes: a new brick of height ``brick`` (default: the ATR over ``atr_period``) forms when
    the close moves a full brick beyond the last one; a reversal needs a full brick beyond the brick's *open*,
    so it takes two bricks of movement.

    Time is not an axis of Renko. To stay aligned with the real candles each brick is stamped with the time of the
    candle that completed it; several bricks from one candle share that candle's span evenly. The index is
    strictly increasing. Columns: open, high, low, close, ``up`` (bool)."""
    cols = ["open", "high", "low", "close", "up"]
    if df.empty:
        return pd.DataFrame(columns=cols, index=pd.DatetimeIndex([], tz="UTC"))
    brick = float(brick) if brick else atr_brick(df, atr_period)
    if not brick > 0:
        raise ValueError("Renko brick size must be above 0")
    closes = df["close"].to_numpy(float)
    times = df.index.as_unit("ns").asi8  # nanoseconds, whatever resolution the index has
    step = int(np.median(np.diff(times))) if len(times) > 1 else 86_400 * 10**9
    last_open = last_close = closes[0]
    rows, stamps = [], []
    for i, c in enumerate(closes):
        if not np.isfinite(c):
            continue
        made = []
        while c >= max(last_open, last_close) + brick:
            top = max(last_open, last_close)
            last_open, last_close = top, top + brick
            made.append((last_open, last_close))
        while c <= min(last_open, last_close) - brick:
            bottom = min(last_open, last_close)
            last_open, last_close = bottom, bottom - brick
            made.append((last_open, last_close))
        span = (times[i + 1] - times[i]) if i + 1 < len(times) else step
        for k, (b_open, b_close) in enumerate(made):
            rows.append((b_open, max(b_open, b_close), min(b_open, b_close), b_close, b_close > b_open))
            stamps.append(times[i] + span * k // len(made))
    index = pd.DatetimeIndex(pd.to_datetime(stamps, unit="ns", utc=True)) if stamps else pd.DatetimeIndex([], tz="UTC")
    out = pd.DataFrame(rows, columns=cols, index=index)
    out.index.name = df.index.name
    return out


def percent_change(closes: dict[str, pd.Series]) -> pd.DataFrame:
    """Several symbols' closes as percent change from the first timestamp they all have (compare mode): every column
    starts at 0, so symbols of any price level are comparable. Only timestamps all of them have are kept."""
    frame = pd.concat({name: s for name, s in closes.items()}, axis=1, join="inner").dropna()
    if frame.empty:
        return frame
    return (frame / frame.iloc[0] - 1) * 100
