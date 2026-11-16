# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Vectorized OHLC backtester.

Execution model (shared with the live bot): the algorithm's position for bar
``t`` is computed from candles up to and including ``t`` and filled at the
close of ``t``; it earns the close-to-close return of bar ``t+1``. A
proportional ``fee`` is charged on every change of position, so a full
round trip costs ``2 * fee``. ``slippage`` (also a fraction per side, 0 by default) models the
bid/ask spread and the worse fill price, and is charged the same way on top of the fee. There is no
look-ahead by construction.

Fills inside a candle (#150): a strategy that fills stops, take-profits or limit entries against the
candle's high and low (a combo with ``fills="intrabar"``) also returns :class:`Fills`, the correction of
each candle's return for those fills away from the close and the extra turnover of round trips inside one
candle. Without them (every other strategy) everything is filled at the close as above.

Shorts and leverage (#149): a short position or a :class:`~tradebot.backtest.margin.Margin` with leverage or
funding switches to the margin model of :mod:`tradebot.backtest.margin` (fixed units per position, liquidation,
borrow fees and funding); a long-only strategy without them is simulated exactly as before.

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
from tradebot.backtest.margin import Margin, needs_margin
from tradebot.backtest.margin import simulate as simulate_margin
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


def bps(value: float) -> float:
    """Basis points as a fraction (``5`` -> ``0.0005``)."""
    return value / 10_000.0


def simulate(close, positions, fee: float = 0.001, slippage: float = 0.0, fills: Fills | None = None,
             margin: Margin | None = None, low=None, high=None, funding=None, periods_per_year: float = 365.0):
    """Per-bar strategy returns, shape ``broadcast(positions, close)``.

    ``fee`` and ``slippage`` are fractions per side; both are charged on the turnover, so only their sum
    matters (``slippage`` may also be an array broadcastable against ``positions``, e.g. a per-bar cost).
    ``fills``: intrabar fills (see :class:`Fills`), None = everything at the close; their extra turnover pays
    the fee and the slippage like any other fill.
    ``margin``, ``low``, ``high``, ``funding``, ``periods_per_year``: shorts and leverage, see
    :func:`tradebot.backtest.margin.simulate` (used whenever a position is short, ``margin`` has leverage or
    funding, or historical ``funding`` is given).
    """
    positions = ops.nan_to_num(positions)
    if funding is not None or needs_margin(positions, margin):
        if fills is not None:
            raise ValueError("intrabar fills cannot be combined with shorts or leverage yet; use fills=close")
        return simulate_margin(close, positions, fee, slippage, margin, low, high, funding, periods_per_year)
    asset_ret = ops.nan_to_num(close / ops.shift(close, 1) - 1)
    held = ops.shift(positions, 1, fill=0.0)
    turnover = ops.abs_(positions - held)
    cost = fee + slippage
    out = held * asset_ret - cost * turnover
    if fills is not None:
        out = out + ops.nan_to_num(fills.adjust) - cost * ops.nan_to_num(fills.turnover)
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
    #: candles in which a position was liquidated (#149), None when the margin model was not needed
    liquidations: pd.Series | None = None

    def summary(self) -> str:
        return " | ".join(f"{k}: {m.fmt(k, v)}" for k, v in self.metrics.items())


def backtest(
    df: pd.DataFrame,
    algorithm: BaseAlgorithm,
    fee: float = 0.001,
    initial_capital: float = 10_000.0,
    periods_per_year: float | None = None,
    slippage: float = 0.0,
    margin: Margin | None = None,
    funding=None,
) -> BacktestResult:
    """Run ``algorithm`` over the OHLC frame ``df`` and compute performance metrics. ``margin``: leverage,
    maintenance margin, borrow fee and funding (#149); ``funding``: historical funding per candle (aligned with
    ``df``, see :func:`tradebot.data.funding.per_candle`)."""
    if len(df) < 2:
        raise ValueError("need at least two candles to backtest")
    positions, fills = algorithm.run_fills(df)
    return from_positions(df, positions, algorithm.name, dict(algorithm.values), fee, initial_capital,
                          periods_per_year, slippage, fills=fills, margin=margin, funding=funding)


def from_positions(
    df: pd.DataFrame,
    positions: pd.Series,
    name: str,
    params: dict | None = None,
    fee: float = 0.001,
    initial_capital: float = 10_000.0,
    periods_per_year: float | None = None,
    slippage: float = 0.0,
    fills: pd.DataFrame | None = None,
    margin: Margin | None = None,
    funding=None,
) -> BacktestResult:
    """Backtest a given series of target positions (one per candle of ``df``); ``fills``: intrabar fills as in
    :attr:`BacktestResult.fills`; ``margin``/``funding`` as in :func:`backtest`."""
    close = df["close"].to_numpy(dtype=float)
    ppy = periods_per_year or m.infer_periods_per_year(df.index)
    pos = np.nan_to_num(positions.to_numpy(dtype=float))
    liquidations = None
    if funding is not None or needs_margin(pos, margin):
        if fills is not None:
            raise ValueError("intrabar fills cannot be combined with shorts or leverage yet; use fills=close")
        col = lambda c: df[c].to_numpy(dtype=float) if c in df else None  # noqa: E731
        fund = None if funding is None else np.nan_to_num(np.asarray(funding, dtype=float))
        ret, liq = simulate_margin(close, pos, fee, slippage, margin, col("low"), col("high"), fund, ppy, detail=True)
        returns = pd.Series(ret, index=df.index, name="returns")
        liquidations = pd.Series(liq, index=df.index, name="liquidations")
    else:
        returns = pd.Series(simulate(close, pos, fee, slippage, Fills.of(fills)), index=df.index, name="returns")
    equity = initial_capital * (1 + returns).cumprod()
    trips = None
    if fills is not None:  # a buy and a sell inside one candle that starts and ends flat
        flat = (positions == 0) & (positions.shift(1, fill_value=0.0) == 0)
        trips = (flat & (fills["turnover"] > 1.5)).to_numpy()
    trades = m.extract_trades(positions, returns, trips)
    metrics = m.compute_metrics(returns, positions, trades, ppy)
    if liquidations is not None:
        metrics["liquidations"] = int(liquidations.sum())
    return BacktestResult(
        algorithm=name,
        params=params or {},
        positions=positions,
        returns=returns,
        equity=equity.rename("equity"),
        trades=trades,
        metrics=metrics,
        fills=fills,
        liquidations=liquidations,
    )
