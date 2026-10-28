# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Vectorized OHLC backtester.

Execution model (shared with the live bot): the algorithm's position for bar
``t`` is computed from candles up to and including ``t`` and filled at the
close of ``t``; it earns the close-to-close return of bar ``t+1``. A
proportional ``fee`` is charged on every change of position, so a full
round trip costs ``2 * fee``. There is no look-ahead by construction.

:func:`simulate` and :func:`batch_fitness` work on NumPy or torch arrays and
accept a population axis, which the optimizer uses to score thousands of
parameter sets in one pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from tradebot.algorithms.base import BaseAlgorithm
from tradebot.backtest import metrics as m
from tradebot.core import ops


def simulate(close, positions, fee: float = 0.001):
    """Per-bar strategy returns, shape ``broadcast(positions, close)``."""
    positions = ops.nan_to_num(positions)
    asset_ret = ops.nan_to_num(close / ops.shift(close, 1) - 1)
    held = ops.shift(positions, 1, fill=0.0)
    turnover = ops.abs_(positions - held)
    return held * asset_ret - fee * turnover


def batch_fitness(returns, periods_per_year: float, objective: str = "sharpe"):
    """Score each row of ``returns`` (``[..., T]``); higher is better."""
    equity = ops.cumprod(1 + returns)
    if objective == "return":
        return equity[..., -1] - 1
    drawdown = 1 - equity / ops.cummax(equity)
    max_dd = drawdown.max(-1) if not ops.is_torch(drawdown) else drawdown.amax(-1)
    if objective == "calmar":
        years = returns.shape[-1] / periods_per_year
        cagr = equity[..., -1] ** (1 / years) - 1
        return cagr / (max_dd + 1e-9)
    mean = returns.mean(-1)
    std = returns.std(-1) if not ops.is_torch(returns) else returns.std(-1, unbiased=False)
    sharpe = mean / (std + 1e-12) * periods_per_year**0.5
    if objective == "sharpe":
        return sharpe
    raise ValueError(f"unknown objective {objective!r} (sharpe, return, calmar)")


@dataclass
class BacktestResult:
    algorithm: str
    params: dict
    positions: pd.Series
    returns: pd.Series
    equity: pd.Series
    trades: pd.DataFrame
    metrics: dict = field(default_factory=dict)

    def summary(self) -> str:
        return " | ".join(f"{k}: {m.fmt(k, v)}" for k, v in self.metrics.items())


def backtest(
    df: pd.DataFrame,
    algorithm: BaseAlgorithm,
    fee: float = 0.001,
    initial_capital: float = 10_000.0,
    periods_per_year: float | None = None,
) -> BacktestResult:
    """Run ``algorithm`` over the OHLC frame ``df`` and compute performance metrics."""
    if len(df) < 2:
        raise ValueError("need at least two candles to backtest")
    positions = algorithm.run(df)
    close = df["close"].to_numpy(dtype=float)
    returns = pd.Series(simulate(close, positions.to_numpy(), fee), index=df.index, name="returns")
    equity = initial_capital * (1 + returns).cumprod()
    ppy = periods_per_year or m.infer_periods_per_year(df.index)
    trades = m.extract_trades(positions, returns)
    return BacktestResult(
        algorithm=algorithm.name,
        params=dict(algorithm.values),
        positions=positions,
        returns=returns,
        equity=equity.rename("equity"),
        trades=trades,
        metrics=m.compute_metrics(returns, positions, trades, ppy),
    )
