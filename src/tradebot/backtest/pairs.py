# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Two-legged pairs backtest (#117): long one asset and short the other at the same time.

Same execution model as :mod:`tradebot.backtest.engine`: a signal from candles up to and including bar ``t``
is filled at the close of ``t`` and earns bar ``t+1``. Every number the signal uses comes from candles
before the bar (:func:`~tradebot.analysis.pairs.rolling_pair`), so there is no look-ahead.

Per trade the capital is split by the hedge ratio at entry and the two weights stay fixed until the exit:
``w_a = s / (1 + |β|)``, ``w_b = -s · β / (1 + |β|)`` with ``s = +1`` (long the spread: long A, short B)
or ``-1``. Costs: ``fee`` on every change of either leg's weight, and ``borrow`` (per year) on the short leg's
weight for every bar it is held.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradebot.analysis.pairs import kalman_pair, rolling_pair
from tradebot.backtest import metrics as m
from tradebot.backtest.engine import BacktestResult


def next_side(s: float, z: float, beta: float, entry: float, exit: float, stop: float) -> float:
    """The position (+1 long the gap: long A / short B, -1 short it, 0 flat) after seeing z-score ``z`` and
    hedge ratio ``beta`` while holding ``s``. The one rule shared by the backtest and the live bot."""
    if np.isnan(z) or np.isnan(beta):
        return 0.0
    if s == 0.0:
        # a negative β is no pair: the legs move against each other
        return (-1.0 if z > 0 else 1.0) if entry <= abs(z) < stop and beta > 0 else 0.0
    return 0.0 if abs(z) <= exit or abs(z) >= stop else s


def pair_backtest(a: pd.DataFrame, b: pd.DataFrame, window: int = 60, entry: float = 2.0, exit: float = 0.5,
                  stop: float = 4.0, fee: float = 0.001, borrow: float = 0.02, initial_capital: float = 10_000.0,
                  periods_per_year: float | None = None, method: str = "rolling") -> BacktestResult:
    """Trade the gap of two OHLC frames: open when its z-score passes ``±entry``, close inside ``±exit``,
    give up (stop) beyond ``±stop``. ``borrow`` is the yearly cost of the short leg (stocks; ~0 for crypto).
    ``method`` picks the hedge ratio: ``rolling`` (OLS over ``window`` candles) or ``kalman`` (a filter that
    follows the pair smoothly, no window)."""
    if method not in ("rolling", "kalman"):
        raise ValueError("method must be rolling or kalman")
    if not 0 <= exit < entry < stop:
        raise ValueError("need 0 <= exit < entry < stop")
    idx = a.index.intersection(b.index)
    ca, cb = a["close"].loc[idx], b["close"].loc[idx]
    if len(idx) < window + 10:
        raise ValueError(f"need at least {window + 10} common candles")
    roll = kalman_pair(ca, cb) if method == "kalman" else rolling_pair(ca, cb, window)
    z, beta = roll.zscore.to_numpy(), roll.beta.to_numpy()
    n = len(idx)
    side, wa, wb = np.zeros(n), np.zeros(n), np.zeros(n)
    s = 0.0
    held_beta = 1.0
    for t in range(n):
        new = next_side(s, z[t], beta[t], entry, exit, stop)
        if s == 0.0 and new != 0.0:
            held_beta = beta[t]
        s = new
        side[t] = s
        wa[t] = s / (1 + held_beta)
        wb[t] = -s * held_beta / (1 + held_beta)
    ra = ca.pct_change().fillna(0.0).to_numpy()
    rb = cb.pct_change().fillna(0.0).to_numpy()
    prev_a, prev_b = np.r_[0.0, wa[:-1]], np.r_[0.0, wb[:-1]]
    ppy = periods_per_year or m.infer_periods_per_year(idx)
    short = np.maximum(-prev_a, 0) + np.maximum(-prev_b, 0)
    ret = prev_a * ra + prev_b * rb - fee * (np.abs(wa - prev_a) + np.abs(wb - prev_b)) - borrow / ppy * short
    returns = pd.Series(ret, index=idx, name="returns")
    positions = pd.Series(side, index=idx, name="position")
    trades = m.extract_trades(positions, returns)
    return BacktestResult(
        algorithm="pairs", params={"method": method, "window": window, "entry": entry, "exit": exit, "stop": stop},
        positions=positions, returns=returns, equity=(initial_capital * (1 + returns).cumprod()).rename("equity"),
        trades=trades, metrics=m.compute_metrics(returns, positions, trades, ppy))
