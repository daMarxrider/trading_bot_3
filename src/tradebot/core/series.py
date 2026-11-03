# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""The OHLC container handed to indicators and algorithms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from tradebot.core import ops


@dataclass(frozen=True)
class OHLC:
    """Open/high/low/close arrays (NumPy or torch), time on the last axis.

    Only OHLC is exposed on purpose: every chart, indicator and backtest in the
    platform works from these four series so that results are identical no
    matter which venue or data provider delivered the candles. ``time`` (the
    candle's open time in Unix seconds, same shape and backend) is there for
    calendar-based indicators like the halving cycle (#99); it is ``None`` when
    the candles have no timestamps.
    """

    open: object
    high: object
    low: object
    close: object
    time: object = None

    @classmethod
    def from_frame(cls, df: pd.DataFrame, device: str | None = None) -> "OHLC":
        """Build from a normalized frame; ``device`` (e.g. ``"cuda"``) yields torch tensors."""
        cols = [df[c].to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")]
        cols.append(unix_seconds(df.index))
        if device is not None:
            cols = [None if c is None else ops.torch.tensor(c, dtype=ops.torch.float64, device=device) for c in cols]
        return cls(*cols)

    def fields(self) -> tuple:
        """``(open, high, low, close, time)``, e.g. to slice or convert all of them: ``OHLC(*(f(x) ...))``."""
        return self.open, self.high, self.low, self.close, self.time

    def __len__(self) -> int:
        return self.close.shape[-1]

    @property
    def typical(self):
        return (self.high + self.low + self.close) / 3


def unix_seconds(index) -> np.ndarray | None:
    """Unix seconds (float64) of a datetime index (naive = UTC); ``None`` for any other index."""
    if not isinstance(index, pd.DatetimeIndex):
        return None
    utc = index.tz_localize("UTC") if index.tz is None else index.tz_convert("UTC")
    return ((utc - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta(seconds=1)).to_numpy(dtype=np.float64)
