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

from tradebot.analysis.pairs import rolling_pair
from tradebot.backtest import metrics as m
from tradebot.backtest.engine import BacktestResult


def pair_backtest(a: pd.DataFrame, b: pd.DataFrame, window: int = 60, entry: float = 2.0, exit: float = 0.5,
                  stop: float = 4.0, fee: float = 0.001, borrow: float = 0.02, initial_capital: float = 10_000.0,
                  periods_per_year: float | None = None) -> BacktestResult:
    """Trade the gap of two OHLC frames: open when its z-score passes ``±entry``, close inside ``±exit``,
    give up (stop) beyond ``±stop``. ``borrow`` is the yearly cost of the short leg (stocks; ~0 for crypto)."""
    if not 0 <= exit < entry < stop:
        raise ValueError("need 0 <= exit < entry < stop")
    idx = a.index.intersection(b.index)
    ca, cb = a["close"].loc[idx], b["close"].loc[idx]
    if len(idx) < window + 10:
        raise ValueError(f"need at least {window + 10} common candles")
    roll = rolling_pair(ca, cb, window)
    z, beta = roll.zscore.to_numpy(), roll.beta.to_numpy()
    n = len(idx)
    side, wa, wb = np.zeros(n), np.zeros(n), np.zeros(n)
    s = 0.0
    held_beta = 1.0
    for t in range(n):
        if np.isnan(z[t]) or np.isnan(beta[t]):
            s = 0.0
        elif s == 0.0:
            if entry <= abs(z[t]) < stop and beta[t] > 0:  # a negative β is no pair: the legs move against each other
                s, held_beta = (-1.0 if z[t] > 0 else 1.0), beta[t]
        elif abs(z[t]) <= exit or abs(z[t]) >= stop:
            s = 0.0
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
        algorithm="pairs", params={"window": window, "entry": entry, "exit": exit, "stop": stop},
        positions=positions, returns=returns, equity=(initial_capital * (1 + returns).cumprod()).rename("equity"),
        trades=trades, metrics=m.compute_metrics(returns, positions, trades, ppy))
