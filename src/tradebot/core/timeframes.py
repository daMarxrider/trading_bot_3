# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Canonical timeframe names and OHLC resampling."""

from __future__ import annotations

import pandas as pd

TIMEFRAMES: dict[str, pd.Timedelta] = {
    "1m": pd.Timedelta(minutes=1),
    "5m": pd.Timedelta(minutes=5),
    "15m": pd.Timedelta(minutes=15),
    "30m": pd.Timedelta(minutes=30),
    "1h": pd.Timedelta(hours=1),
    "4h": pd.Timedelta(hours=4),
    "1d": pd.Timedelta(days=1),
    "1w": pd.Timedelta(weeks=1),
}


def to_timedelta(timeframe: str) -> pd.Timedelta:
    try:
        return TIMEFRAMES[timeframe]
    except KeyError:
        raise ValueError(f"unknown timeframe {timeframe!r}; use one of {', '.join(TIMEFRAMES)}") from None


def resample(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Aggregate a finer OHLC frame into ``timeframe`` candles."""
    rule = to_timedelta(timeframe)
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in df.columns:
        agg["volume"] = "sum"
    return df.resample(rule, label="left", closed="left").agg(agg).dropna(subset=["close"])
