# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Performance metrics: Sharpe, max drawdown, win rate, profit factor, CAGR, ..."""

from __future__ import annotations

import numpy as np
import pandas as pd

SECONDS_PER_YEAR = 365.25 * 24 * 3600


def infer_periods_per_year(index: pd.DatetimeIndex) -> float:
    """Bars per year as observed in the data.

    Measured rather than assumed, so daily stock data (≈252 bars/year, no
    weekends) and daily crypto data (365) are both annualized correctly.
    """
    if len(index) < 2:
        return 252.0
    span = (index[-1] - index[0]).total_seconds()
    return (len(index) - 1) / span * SECONDS_PER_YEAR if span > 0 else 252.0


def extract_trades(positions: pd.Series, returns: pd.Series) -> pd.DataFrame:
    """One row per contiguous holding period with the same direction.

    A trade's return compounds the strategy returns from the entry bar (entry
    fee) through the exit bar (last price move + exit fee). A trade still open
    at the end is closed on the last bar.
    """
    pos = positions.to_numpy()
    ret = returns.to_numpy()
    rows = []
    start, side = None, 0.0
    for t in range(len(pos) + 1):
        cur = np.sign(pos[t]) if t < len(pos) else 0.0
        if cur != side:
            if side != 0:
                exit_t = min(t, len(pos) - 1)
                r = np.prod(1 + ret[start : exit_t + 1]) - 1
                rows.append((positions.index[start], positions.index[exit_t], "long" if side > 0 else "short", r))
            start, side = t, cur
    return pd.DataFrame(rows, columns=["entry", "exit", "side", "return"])


def max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    return float(np.max(1 - equity / peak)) if len(equity) else 0.0


def compute_metrics(returns: pd.Series, positions: pd.Series, trades: pd.DataFrame, ppy: float) -> dict:
    r = returns.to_numpy()
    equity = np.cumprod(1 + r)
    total = float(equity[-1] - 1)
    years = (len(r) - 1) / ppy if len(r) > 1 else 0.0
    cagr = float(equity[-1] ** (1 / years) - 1) if years > 0 and equity[-1] > 0 else float("nan")
    std = r.std()
    downside = r[r < 0].std() if (r < 0).sum() > 1 else 0.0
    wins = trades["return"][trades["return"] > 0]
    losses = trades["return"][trades["return"] <= 0]
    gross_loss = -losses.sum()
    return {
        "total_return": total,
        "cagr": cagr,
        "sharpe": float(r.mean() / std * np.sqrt(ppy)) if std > 0 else 0.0,
        "sortino": float(r.mean() / downside * np.sqrt(ppy)) if downside > 0 else 0.0,
        "max_drawdown": max_drawdown(equity),
        "trades": int(len(trades)),
        "win_rate": float(len(wins) / len(trades)) if len(trades) else 0.0,
        "profit_factor": float(wins.sum() / gross_loss) if gross_loss > 0 else float("inf") if len(wins) else 0.0,
        "exposure": float((positions != 0).mean()),
    }


_PERCENT = {"total_return", "cagr", "max_drawdown", "win_rate", "exposure"}


def fmt(key: str, value) -> str:
    if isinstance(value, int):
        return str(value)
    if key in _PERCENT:
        return f"{value * 100:.2f}%"
    return f"{value:.2f}"
