# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Numbers for a pair of assets that move together (#117): z-score of the gap, half-life, hedge ratio."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PairStats:
    bars: int
    correlation: float  #: of the legs' returns
    beta: float  #: hedge ratio: log a ≈ α + β · log b
    zscore: float  #: where the log spread is now, in standard deviations from its mean
    half_life: float | None  #: bars for a gap to close halfway; None when the spread doesn't revert
    spread_mean: float
    spread_std: float


def half_life(x: np.ndarray) -> float | None:
    """Mean-reversion half-life in bars from an AR(1) fit ``x[t] = c + φ·x[t-1]``: ``-ln 2 / ln φ``.
    None unless ``0 < φ < 1``, i.e. the series wanders off instead of coming back."""
    if len(x) < 10:
        return None
    phi = np.polyfit(x[:-1], x[1:], 1)[0]
    return float(-np.log(2) / np.log(phi)) if 0 < phi < 1 else None


def pair_stats(a: pd.Series, b: pd.Series) -> PairStats:
    """Statistics of two close-price series, on the timestamps both have."""
    df = pd.concat([a, b], axis=1, join="inner").dropna()
    df = df[(df > 0).all(axis=1)]
    if len(df) < 10:
        raise ValueError("need at least 10 common candles")
    la, lb = np.log(df.iloc[:, 0].to_numpy()), np.log(df.iloc[:, 1].to_numpy())
    beta = float(np.polyfit(lb, la, 1)[0])
    spread = la - beta * lb
    std = float(spread.std())
    return PairStats(
        bars=len(df),
        correlation=float(np.corrcoef(np.diff(la), np.diff(lb))[0, 1]),
        beta=beta,
        zscore=float((spread[-1] - spread.mean()) / std) if std > 0 else 0.0,
        half_life=half_life(spread),
        spread_mean=float(spread.mean()),
        spread_std=std,
    )
