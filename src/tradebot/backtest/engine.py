# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Vectorized OHLC backtester.

Execution model (shared with the live bot): the algorithm's position for bar
``t`` is computed from candles up to and including ``t`` and filled at the
close of ``t``; it earns the close-to-close return of bar ``t+1``. A
proportional ``fee`` is charged on every change of position, so a full
round trip costs ``2 * fee``. There is no look-ahead by construction.

Fills inside a candle (#150): a strategy that fills stops, take-profits or limit entries against the
candle's high and low (a combo with ``fills="intrabar"``) also returns :class:`Fills`, the correction of
each candle's return for those fills away from the close and the extra turnover of round trips inside one
candle. Without them (every other strategy) everything is filled at the close as above.

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


@dataclass
class Fills:
    """Fills away from the close, per candle (arrays shaped like the positions; NumPy or torch).

    ``adjust`` is added to the candle's strategy return (the return of the fills' prices minus what the close
    model would have earned); ``turnover`` is the turnover beyond ``|position - previous position|`` (2 for a
    buy and a sell inside one candle), charged the fee. Both are 0 on candles filled at the close.
    """

    adjust: object
    turnover: object

    def map(self, f) -> "Fills":
        """The same fills with ``f`` applied to both arrays (slicing, broadcasting, backend conversion)."""
        return Fills(f(self.adjust), f(self.turnover))

    def frame(self, index) -> pd.DataFrame:
        a = self.adjust.detach().cpu().numpy() if ops.is_torch(self.adjust) else np.asarray(self.adjust)
        t = self.turnover.detach().cpu().numpy() if ops.is_torch(self.turnover) else np.asarray(self.turnover)
        return pd.DataFrame({"adjust": np.broadcast_to(a, (len(index),)), "turnover": np.broadcast_to(t, (len(index),))},
                            index=index)

    @classmethod
    def of(cls, frame: pd.DataFrame | None) -> "Fills | None":
        return None if frame is None else cls(frame["adjust"].to_numpy(), frame["turnover"].to_numpy())


def simulate(close, positions, fee: float = 0.001, fills: Fills | None = None):
    """Per-bar strategy returns, shape ``broadcast(positions, close)``; ``fills``: intrabar fills (see
    :class:`Fills`), None = everything at the close."""
    positions = ops.nan_to_num(positions)
    asset_ret = ops.nan_to_num(close / ops.shift(close, 1) - 1)
    held = ops.shift(positions, 1, fill=0.0)
    turnover = ops.abs_(positions - held)
    out = held * asset_ret - fee * turnover
    if fills is not None:
        out = out + ops.nan_to_num(fills.adjust) - fee * ops.nan_to_num(fills.turnover)
    return out


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
    #: intrabar fills (columns ``adjust``, ``turnover``; see :class:`Fills`), None when all fills are at the close
    fills: pd.DataFrame | None = None

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
    positions, fills = algorithm.run_fills(df)
    return from_positions(df, positions, algorithm.name, dict(algorithm.values), fee, initial_capital,
                          periods_per_year, fills=fills)


def from_positions(
    df: pd.DataFrame,
    positions: pd.Series,
    name: str,
    params: dict | None = None,
    fee: float = 0.001,
    initial_capital: float = 10_000.0,
    periods_per_year: float | None = None,
    fills: pd.DataFrame | None = None,
) -> BacktestResult:
    """Backtest a given series of target positions (one per candle of ``df``); ``fills``: intrabar fills as in
    :attr:`BacktestResult.fills`."""
    close = df["close"].to_numpy(dtype=float)
    returns = pd.Series(simulate(close, positions.to_numpy(), fee, Fills.of(fills)), index=df.index, name="returns")
    equity = initial_capital * (1 + returns).cumprod()
    ppy = periods_per_year or m.infer_periods_per_year(df.index)
    trips = None
    if fills is not None:  # a buy and a sell inside one candle that starts and ends flat
        flat = (positions == 0) & (positions.shift(1, fill_value=0.0) == 0)
        trips = (flat & (fills["turnover"] > 1.5)).to_numpy()
    trades = m.extract_trades(positions, returns, trips)
    return BacktestResult(
        algorithm=name,
        params=params or {},
        positions=positions,
        returns=returns,
        equity=equity.rename("equity"),
        trades=trades,
        metrics=m.compute_metrics(returns, positions, trades, ppy),
        fills=fills,
    )
