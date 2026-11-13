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
    """Open/high/low/close arrays (NumPy or torch), time on the last axis, plus ``time`` and ``volume``.

    **Prices are venue-independent**: every chart, indicator and backtest works from open, high, low and close,
    so results that use only those are the same no matter which venue or data provider delivered the candles.

    ``time`` (the candle's open time in Unix seconds, same shape and backend) is there for calendar-based
    indicators like the halving cycle (#99); it is ``None`` when the candles have no timestamps.

    ``volume`` (#116) is the traded amount per candle. **It is venue-dependent**: one exchange's BTC volume isn't
    another's, and a provider may report a different number for the same stock, so results that use volume can
    differ between providers. Without volume data (no column, or all 0 as for many indices) it is an all-NaN array
    of the same shape and backend as ``close``, so indicators that use it give NaN (no signal) rather than failing.
    """

    open: object
    high: object
    low: object
    close: object
    time: object = None
    volume: object = None

    def __post_init__(self):
        if self.volume is None:  # always an array: NaN of close's shape and backend
            object.__setattr__(self, "volume", self.close * ops.NAN)

    @classmethod
    def from_frame(cls, df: pd.DataFrame, device: str | None = None) -> "OHLC":
        """Build from a normalized frame; ``device`` (e.g. ``"cuda"``) yields torch tensors."""
        cols = [df[c].to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")]
        cols += [unix_seconds(df.index), volume_of(df)]
        if device is not None:
            cols = [None if c is None else ops.torch.tensor(c, dtype=ops.torch.float64, device=device) for c in cols]
        return cls(*cols)

    def fields(self) -> tuple:
        """``(open, high, low, close, time, volume)``, e.g. to slice or convert all of them: ``OHLC(*(f(x) ...))``."""
        return self.open, self.high, self.low, self.close, self.time, self.volume

    @property
    def has_volume(self) -> bool:
        """Whether any candle has a volume."""
        return bool((~ops.isnan(self.volume)).any())

    def __len__(self) -> int:
        return self.close.shape[-1]

    @property
    def typical(self):
        return (self.high + self.low + self.close) / 3


def volume_of(df: pd.DataFrame) -> np.ndarray:
    """The ``volume`` column as float64; all NaN when there is none or it's all 0 (no volume data, as for indices)."""
    if "volume" not in df.columns:
        return np.full(len(df), np.nan)
    v = df["volume"].to_numpy(dtype=np.float64)
    return v if np.nansum(np.abs(v)) > 0 else np.full(len(df), np.nan)


def unix_seconds(index) -> np.ndarray | None:
    """Unix seconds (float64) of a datetime index (naive = UTC); ``None`` for any other index."""
    if not isinstance(index, pd.DatetimeIndex):
        return None
    utc = index.tz_localize("UTC") if index.tz is None else index.tz_convert("UTC")
    return ((utc - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta(seconds=1)).to_numpy(dtype=np.float64)
