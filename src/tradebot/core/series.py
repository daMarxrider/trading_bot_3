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
    matter which venue or data provider delivered the candles.
    """

    open: object
    high: object
    low: object
    close: object

    @classmethod
    def from_frame(cls, df: pd.DataFrame, device: str | None = None) -> "OHLC":
        """Build from a normalized frame; ``device`` (e.g. ``"cuda"``) yields torch tensors."""
        cols = [df[c].to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")]
        if device is not None:
            cols = [ops.torch.tensor(c, dtype=ops.torch.float64, device=device) for c in cols]
        return cls(*cols)

    def __len__(self) -> int:
        return self.close.shape[-1]

    @property
    def typical(self):
        return (self.high + self.low + self.close) / 3
