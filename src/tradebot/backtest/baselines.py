# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Baselines a strategy has to beat (#19): buy & hold, and investing a fixed amount every month.

Both run on the same candles with the same fee and the backtester's execution model: an order fills at
the close of its bar, and every purchase pays ``fee``.

Monthly investing puts the money in gradually, so its time-weighted metrics (total return, CAGR, Sharpe,
max drawdown, all computed from the returns of the money already invested) look much like buy & hold.
The fair comparison is the money-weighted return, XIRR: the yearly rate that grows the dated payments
into the final value. Every result here therefore also gets an XIRR, and "strategy - baseline" is the
difference of the two XIRRs. For a single payment at the start (the strategy, buy & hold) XIRR is the
CAGR measured in calendar time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from tradebot.backtest import metrics as m
from tradebot.backtest.engine import BacktestResult, from_positions

BUY_HOLD, MONTHLY = "buy & hold", "monthly investing"


def xirr(dates, amounts) -> float:
    """Yearly rate at which the payments (negative) grow into the payouts (positive); NaN if there is none.

    The future value ``sum(a_i * (1 + r) ** (T - t_i))`` falls as ``r`` rises when all payments come
    before the payout, so bisection finds the single root.
    """
    t = np.array([(pd.Timestamp(d) - pd.Timestamp(dates[0])).total_seconds() for d in dates]) / m.SECONDS_PER_YEAR
    a = np.asarray(amounts, dtype=float)
    if t[-1] <= 0 or not (a < 0).any() or not (a > 0).any():
        return float("nan")
    left = t[-1] - t

    def future(x):  # x = log(1 + r)
        with np.errstate(over="ignore"):
            return float(np.sum(a * np.exp(x * left)))

    lo, hi = -10.0, 10.0  # r from -99.995% to about +2,200,000% a year
    if future(lo) < 0 or future(hi) > 0:
        return float("nan")
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if future(mid) > 0 else (lo, mid)
    return float(np.expm1((lo + hi) / 2))


def lump_sum_xirr(result: BacktestResult, initial_capital: float) -> float:
    """XIRR of investing ``initial_capital`` on the first candle and ending with the result's final equity."""
    index = result.equity.index
    return xirr([index[0], index[-1]], [-initial_capital, float(result.equity.iloc[-1])])


def buy_and_hold(df: pd.DataFrame, fee: float = 0.001, initial_capital: float = 10_000.0,
                 periods_per_year: float | None = None) -> BacktestResult:
    """Everything invested at the close of the first candle and held to the end."""
    res = from_positions(df, pd.Series(1.0, index=df.index, name="position"), BUY_HOLD, fee=fee,
                         initial_capital=initial_capital, periods_per_year=periods_per_year)
    res.metrics["xirr"] = lump_sum_xirr(res, initial_capital)
    return res


def month_starts(index: pd.DatetimeIndex) -> np.ndarray:
    """True on the first candle of each calendar month (the first candle of the data always counts)."""
    month = index.year * 12 + index.month
    return np.r_[True, month[1:] != month[:-1]] if len(index) else np.zeros(0, bool)


def monthly_investing(df: pd.DataFrame, fee: float = 0.001, initial_capital: float = 10_000.0,
                      amount: float | None = None, periods_per_year: float | None = None) -> BacktestResult:
    """Invest ``amount`` at the close of the first candle of each month (default: capital / months).

    ``returns`` are time-weighted: each bar's price move on the units already held, times the fee paid on
    that bar's purchase. ``equity`` is the account: holdings plus the capital not invested yet, which waits
    as cash (it goes negative if the payments add up to more than the capital). ``metrics["xirr"]`` is the
    money-weighted return of the payments.
    """
    close = df["close"].to_numpy(dtype=float)
    buy = month_starts(df.index)
    amount = float(amount or initial_capital / buy.sum())
    paid = np.where(buy, amount, 0.0)
    units = np.cumsum(paid * (1 - fee) / close)
    held = np.r_[0.0, units[:-1]]
    before = held * close  # last bar's units at this bar's close
    previous = np.r_[0.0, units[:-1] * close[:-1]]
    move = np.divide(before, previous, out=np.ones_like(close), where=previous > 0)
    after_fee = np.divide(before + paid * (1 - fee), before + paid, out=np.ones_like(close), where=before + paid > 0)
    returns = pd.Series(move * after_fee - 1, index=df.index, name="returns")
    value = units * close
    positions = pd.Series((units > 0).astype(float), index=df.index, name="position")
    ppy = periods_per_year or m.infer_periods_per_year(df.index)
    trades = m.extract_trades(positions, returns)
    metrics = m.compute_metrics(returns, positions, trades, ppy)
    metrics["xirr"] = xirr([*df.index[buy], df.index[-1]], [*-paid[buy], value[-1]])
    equity = pd.Series(initial_capital - np.cumsum(paid) + value, index=df.index, name="equity")
    return BacktestResult(MONTHLY, {"amount": amount}, positions, returns, equity, trades, metrics)


@dataclass
class Baselines:
    """The baselines on one stretch of data, and how the strategy did against them."""

    results: dict[str, BacktestResult]  #: name -> baseline (with an "xirr" metric)
    strategy_xirr: float

    @property
    def excess(self) -> dict[str, float]:
        """Strategy XIRR minus baseline XIRR, per baseline: positive = the strategy did better."""
        return {name: self.strategy_xirr - r.metrics["xirr"] for name, r in self.results.items()}

    @property
    def not_beaten(self) -> list[str]:
        """The baselines the strategy did not beat (NaN counts as not beaten)."""
        return [name for name, gap in self.excess.items() if not gap > 0]

    def columns(self, name: str, strategy: BacktestResult) -> dict[str, dict]:
        """Results-table columns: the strategy with its XIRR and its lead per baseline, then each baseline."""
        lead = {f"vs {k}": v for k, v in self.excess.items()}
        return {name: {**strategy.metrics, "xirr": self.strategy_xirr, **lead},
                **{k: r.metrics for k, r in self.results.items()}}


def baselines(df: pd.DataFrame, strategy: BacktestResult, fee: float = 0.001, initial_capital: float = 10_000.0,
              amount: float | None = None, periods_per_year: float | None = None) -> Baselines:
    """Buy & hold and monthly investing on the strategy's candles ``df``, with its fee and capital."""
    kw = dict(fee=fee, initial_capital=initial_capital, periods_per_year=periods_per_year)
    return Baselines({BUY_HOLD: buy_and_hold(df, **kw), MONTHLY: monthly_investing(df, amount=amount, **kw)},
                     lump_sum_xirr(strategy, initial_capital))
