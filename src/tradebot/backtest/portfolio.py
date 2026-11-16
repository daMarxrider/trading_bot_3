# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Portfolio backtest (#148): one shared account trading several symbols with the same strategy.

The strategy runs on each symbol's own candles and gives a target position per candle (as in
:mod:`tradebot.backtest.engine`). The account then decides what to hold:

* **slots**: at most ``max_positions`` positions at a time (default: one per symbol). Positions already held
  keep their slot; free slots go to new signals, the strongest ``|signal|`` first, ties in the order the
  symbols were given. Each slot is worth ``1 / max_positions`` of the account, so empty slots stay in cash.
* **allocation** of the held slots' capital: ``equal`` (each the same), ``signal`` (in proportion to the
  signal's strength, e.g. a sized strategy's fraction) or ``volatility`` (inverse volatility: calm symbols get
  more, so each adds about the same risk; volatility of the last ``vol_window`` candles). The signal's own
  size always scales the position (a strategy that is half in gets half its share).
* **rebalancing**: between rebalance dates the positions drift with their prices; a new entry gets its target
  weight (as far as the cash allows) and an exit is sold. On each rebalance date (``candle``, ``week``,
  ``month``, ``quarter``; ``none`` = never) every held position is set back to its target weight.

Execution as in the single-symbol backtest: decisions from candles up to and including ``t`` are filled at the
close of ``t`` and earn ``t+1``; ``fee`` + ``slippage`` are charged on every weight change. No leverage: the
gross exposure never exceeds the account. The symbols are aligned on the union of their candle times inside
the span all of them cover (a stock's last close is carried over a crypto weekend). Intrabar fills (#150) are
not modelled here: every fill is at the close.

The benchmark is buy & hold of the same basket: equal weights bought on the first candle, never rebalanced.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from tradebot.algorithms.base import BaseAlgorithm
from tradebot.backtest import metrics as m
from tradebot.backtest.baselines import lump_sum_xirr
from tradebot.backtest.engine import BacktestResult, simulate

ALLOCATIONS = {
    "equal": "the same share for every held position",
    "signal": "in proportion to the signal's strength",
    "volatility": "inverse volatility: calm symbols get more",
}
REBALANCE = {
    "none": "never: positions drift with their prices",
    "candle": "every candle",
    "week": "on the first candle of each week",
    "month": "on the first candle of each month",
    "quarter": "on the first candle of each quarter",
}
BASKET = "buy & hold basket"
EPS = 1e-9


def rebalance_days(index: pd.DatetimeIndex, rule: str) -> np.ndarray:
    """True on the candles where everything is set back to its target weight (never the first candle)."""
    if rule not in REBALANCE:
        raise ValueError(f"unknown rebalance rule {rule!r} ({', '.join(REBALANCE)})")
    n = len(index)
    if rule == "none" or n == 0:
        return np.zeros(n, bool)
    if rule == "candle":
        key = np.arange(n)
    elif rule == "week":
        iso = index.isocalendar()
        key = (iso["year"] * 53 + iso["week"]).to_numpy()
    elif rule == "month":
        key = np.asarray(index.year * 12 + index.month)
    else:
        key = np.asarray(index.year * 4 + (index.month - 1) // 3)
    return np.r_[False, key[1:] != key[:-1]]


def align(frames: dict[str, pd.DataFrame], signals: dict[str, pd.Series]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Closes and signals of all symbols on one time axis: the union of their candles inside the span every
    symbol covers, the last value carried forward."""
    start = max(df.index[0] for df in frames.values())
    end = min(df.index[-1] for df in frames.values())
    if start >= end:
        raise ValueError("the symbols' histories do not overlap")
    index = frames[next(iter(frames))].index[:0]
    for df in frames.values():
        index = index.union(df.index[(df.index >= start) & (df.index <= end)])
    close = pd.DataFrame({s: df["close"].astype(float).reindex(index, method="ffill") for s, df in frames.items()})
    sig = pd.DataFrame({s: signals[s].astype(float).reindex(index, method="ffill").fillna(0.0) for s in frames})
    return close, sig


@dataclass
class PortfolioResult:
    """The account (``result``: returns, equity, trades with a ``symbol`` column, metrics) and its weights."""

    result: BacktestResult
    weights: pd.DataFrame  #: share of the account per symbol after each candle's trades (negative = short)
    contribution: pd.Series  #: money each symbol made or lost, costs included (sums to the account's P&L)
    benchmark: BacktestResult  #: buy & hold of the same basket
    settings: dict = field(default_factory=dict)

    @property
    def excess(self) -> float:
        """Portfolio XIRR minus the basket's: positive = the portfolio did better."""
        return self.result.metrics["xirr"] - self.benchmark.metrics["xirr"]

    def columns(self, name: str = "portfolio") -> dict[str, dict]:
        """Results-table columns: the portfolio with its lead over the basket, then the basket."""
        return {name: {**self.result.metrics, "vs buy & hold": self.excess}, BASKET: self.benchmark.metrics}

    def per_symbol(self) -> pd.DataFrame:
        """Per symbol: P&L contribution, share of the time held, average weight while held, trades."""
        w = self.weights
        held = w.abs() > EPS
        trades = self.result.trades
        return pd.DataFrame({
            "contribution": self.contribution,
            "time held": held.mean(),
            "avg weight": w.abs().where(held).mean().fillna(0.0),
            "trades": trades["symbol"].value_counts().reindex(w.columns, fill_value=0) if len(trades)
            else pd.Series(0, index=w.columns),
        })


def run_portfolio(close: pd.DataFrame, signals: pd.DataFrame, max_positions: int | None = None,
                  allocation: str = "equal", rebalance: str = "none", fee: float = 0.001,
                  slippage: float | dict = 0.0, initial_capital: float = 10_000.0, vol_window: int = 20,
                  periods_per_year: float | None = None, name: str = "portfolio",
                  params: dict | None = None) -> tuple[BacktestResult, pd.DataFrame, pd.Series]:
    """Simulate the shared account on aligned ``close`` and ``signals`` (one column per symbol).

    Returns ``(result, weights, contribution)``; see :class:`PortfolioResult`.
    """
    if allocation not in ALLOCATIONS:
        raise ValueError(f"unknown allocation {allocation!r} ({', '.join(ALLOCATIONS)})")
    symbols = list(close.columns)
    n_sym, T = len(symbols), len(close)
    if T < 2:
        raise ValueError("need at least two candles to backtest")
    slots = n_sym if not max_positions else int(max_positions)
    if slots < 1:
        raise ValueError("max_positions must be at least 1")
    cost = np.array([(slippage.get(s, 0.0) if isinstance(slippage, dict) else slippage) + fee for s in symbols])
    px = close.to_numpy(dtype=float)
    sig = np.nan_to_num(signals[symbols].to_numpy(dtype=float))
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.nan_to_num(px / np.vstack([px[:1], px[:-1]]) - 1)
    vol = pd.DataFrame(r).rolling(vol_window, min_periods=max(2, vol_window // 2)).std().to_numpy()
    rebal = rebalance_days(close.index, rebalance)
    order = np.arange(n_sym)

    weights = np.zeros((T, n_sym))
    ret = np.zeros(T)
    contrib = np.zeros((T, n_sym))
    w = np.zeros(n_sym)  # after the previous candle's trades
    prev_sig = np.zeros(n_sym)
    equity = initial_capital
    for t in range(T):
        gross = float(w @ r[t])
        drift = w * (1 + r[t]) / (1 + gross) if 1 + gross > EPS else np.zeros(n_sym)
        s = sig[t]
        want = np.abs(s) > EPS
        keep = want & (np.abs(drift) > EPS) & (np.sign(drift) == np.sign(s))
        new = want & ~keep
        free = slots - int(keep.sum())
        if new.sum() > free:  # strongest signals first, ties in the given order
            ranked = sorted(np.flatnonzero(new), key=lambda i: (-abs(s[i]), order[i]))
            new = np.zeros(n_sym, bool)
            new[ranked[:max(free, 0)]] = True
        held = keep | new
        target = np.zeros(n_sym)
        if held.any():
            share = held.sum() / slots  # the held slots' part of the account
            if allocation == "signal":  # the strength sets the share (and so already scales the position)
                a = np.abs(s[held])
                target[held] = np.sign(s[held]) * share * a / a.sum()
            else:
                if allocation == "equal":
                    a = np.ones(n_sym)
                else:
                    v = vol[t]
                    ok = np.isfinite(v) & (v > EPS)
                    fill = np.mean(v[held & ok]) if (held & ok).any() else 1.0  # warm-up: the others' average
                    a = 1.0 / np.where(ok, v, fill)
                target[held] = s[held] * share * a[held] / a[held].sum()
        reset = held & (new | rebal[t] | (np.abs(s - prev_sig) > EPS))
        nw = np.where(held & ~reset, drift, 0.0)
        nw[reset] = target[reset]
        fixed = float(np.abs(nw[held & ~reset]).sum())
        moved = float(np.abs(nw[reset]).sum())
        if fixed + moved > 1 + EPS:  # no leverage: what is bought this candle gets the cash that is left
            room = max(1.0 - fixed, 0.0)
            nw[reset] *= room / moved if moved > 0 else 0.0
            if fixed > 1:
                nw /= float(np.abs(nw).sum())
        paid = cost * np.abs(nw - drift)
        contrib[t] = equity * (w * r[t] - paid)
        ret[t] = gross - paid.sum()
        equity *= 1 + ret[t]
        w = nw
        weights[t] = w
        prev_sig = np.where(held, s, 0.0)

    index = close.index
    returns = pd.Series(ret, index=index, name="returns")
    weights_df = pd.DataFrame(weights, index=index, columns=symbols)
    exposure = pd.Series(np.abs(weights).sum(1), index=index, name="position")
    trades = []
    for j, sym in enumerate(symbols):  # per symbol: the price move while held, its own fees
        side = pd.Series(np.sign(weights[:, j]), index=index)
        unit = pd.Series(simulate(px[:, j], side.to_numpy(), float(cost[j]), 0.0), index=index)
        t_ = m.extract_trades(side, unit)
        t_.insert(0, "symbol", sym)
        trades.append(t_)
    trades = pd.concat(trades, ignore_index=True).sort_values(["entry", "symbol"], ignore_index=True)
    ppy = periods_per_year or m.infer_periods_per_year(index)
    metrics = m.compute_metrics(returns, exposure, trades, ppy)
    metrics["avg_positions"] = float((np.abs(weights) > EPS).sum(1).mean())
    res = BacktestResult(algorithm=name, params=params or {}, positions=exposure, returns=returns,
                         equity=(initial_capital * (1 + returns).cumprod()).rename("equity"), trades=trades,
                         metrics=metrics)
    res.metrics["xirr"] = lump_sum_xirr(res, initial_capital)
    return res, weights_df, pd.Series(contrib.sum(0), index=symbols, name="contribution")


def portfolio_backtest(frames: dict[str, pd.DataFrame], algorithm: BaseAlgorithm, max_positions: int | None = None,
                       allocation: str = "equal", rebalance: str = "none", fee: float = 0.001,
                       slippage: float | dict = 0.0, initial_capital: float = 10_000.0, vol_window: int = 20,
                       periods_per_year: float | None = None) -> PortfolioResult:
    """Trade ``algorithm`` on every symbol of ``frames`` (symbol -> OHLC) from one account, next to buy & hold
    of the same basket. ``slippage``: a fraction per side, or one per symbol."""
    if len(frames) < 1:
        raise ValueError("need at least one symbol")
    signals = {s: algorithm.run(df) for s, df in frames.items()}  # each on its own full history (warm-up)
    close, sig = align(frames, signals)
    settings = {"max_positions": max_positions or len(frames), "allocation": allocation, "rebalance": rebalance,
                "vol_window": vol_window}
    kw = dict(fee=fee, slippage=slippage, initial_capital=initial_capital, periods_per_year=periods_per_year)
    res, weights, contrib = run_portfolio(close, sig, max_positions, allocation, rebalance, vol_window=vol_window,
                                          name=algorithm.name, params=dict(algorithm.values), **kw)
    bench, _, _ = run_portfolio(close, pd.DataFrame(1.0, index=close.index, columns=close.columns), name=BASKET, **kw)
    return PortfolioResult(res, weights, contrib, bench, settings)
