# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Shorts, leverage, liquidation, borrow fees and funding (#149).

The plain engine (:func:`tradebot.backtest.engine.simulate`) treats a position as a fraction of the current
equity, rebalanced every candle. That is fine for a long position without leverage, but it hides the risk of a
short or a leveraged position: a real exchange holds a fixed number of units, so losses raise the leverage and
the position can be liquidated. With :class:`Margin` (and always when a strategy goes short) the backtester
models that instead:

* A target position ``p`` (``-1..1``) puts ``|p|`` of the equity up as margin for a position of
  ``leverage * |p|`` of the equity (isolated margin: the rest of the account is not at risk).
* While the target stays the same the units are fixed at the entry price ``P0``. The margin's value per unit of
  margin is ``g = 1 + s * L * (P / P0 - 1)`` (``s`` = +1 long, -1 short, ``L`` = leverage).
* Liquidation: the position is liquidated in the first candle whose worst price (the low for a long, the high
  for a short) brings the margin down to the maintenance margin, ``g <= maintenance * L * P / P0``. The whole
  margin is lost (what is left goes to the exchange's liquidation fee and insurance fund) and the position stays
  flat until the strategy's target changes. A gap through the liquidation price loses the same, not more.
  Without leverage a long can never be liquidated; a short without leverage is liquidated when the price
  roughly doubles.
* Costs: the fee and the slippage are charged on the traded notional (``L * |change of p|``); a short pays
  ``borrow`` per year on its notional (stock borrow fee); every position pays ``funding`` (perpetual futures:
  a positive rate means longs pay shorts) per candle on its notional, either a constant rate per 8 hours or
  the historical rate per candle (see :mod:`tradebot.data.funding`).

Simplifications (documented, all on the cautious side or second order): the fee on a change of the target
ignores how far the notional drifted since the entry, and the borrow fee and funding are taken from the equity,
not from the margin, so they do not move the liquidation price.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tradebot.core import ops

#: funding periods per year of a perpetual future that settles every 8 hours
FUNDING_PER_YEAR = 3 * 365.25


@dataclass(frozen=True)
class Margin:
    """Margin settings of a backtest.

    ``leverage``: notional per unit of margin (1 = none, the maximum is 125 as on the big exchanges);
    ``maintenance``: maintenance margin rate (0.005 = 0.5 % of the notional, Binance's lowest tier);
    ``borrow``: yearly borrow fee of a short (0.03 = 3 % a year, stocks; ~0 for perpetuals);
    ``funding``: funding rate per 8 hours (0.0001 = 0.01 %, perpetuals), used when no historical rates are given.
    """

    leverage: float = 1.0
    maintenance: float = 0.005
    borrow: float = 0.0
    funding: float = 0.0

    def __post_init__(self):
        if not 1.0 <= self.leverage <= 125.0:
            raise ValueError("leverage must be between 1 and 125")
        if not 0.0 <= self.maintenance < 1.0 / self.leverage:
            raise ValueError(f"the maintenance margin must be between 0 and 1/leverage ({1 / self.leverage:.4f})")
        if self.borrow < 0:
            raise ValueError("the borrow fee cannot be negative")

    @property
    def plain(self) -> bool:
        """True when it changes nothing for a long-only strategy (no leverage, no funding)."""
        return self.leverage == 1.0 and self.funding == 0.0


def needs_margin(positions, margin: Margin | None) -> bool:
    """Whether :func:`simulate` must be used: leverage or funding set, or any short position."""
    if margin is not None and not margin.plain:
        return True
    short = positions < 0
    return bool(short.any()) if ops.is_torch(short) else bool(np.any(short))


def _segments(held):
    """``(start, seg)``: a new segment starts wherever the held position changes; ``seg`` numbers them."""
    start = held != ops.shift(held, 1, fill=0.0)
    seg = ops.cumsum(ops.where(start, 1.0, 0.0))
    return start, seg


def _at_start(value, start):
    """``value`` at the first candle of the segment each candle is in (shaped like ``start``)."""
    T = start.shape[-1]
    if ops.is_torch(start):
        import torch

        value = value.expand(start.shape) if tuple(value.shape) != tuple(start.shape) else value
        idx = torch.where(start, torch.arange(T, device=start.device), 0)
        return torch.gather(value, -1, torch.cummax(idx, dim=-1).values)
    value = np.broadcast_to(value, start.shape)
    idx = np.maximum.accumulate(np.where(start, np.arange(T), 0), axis=-1)
    return np.take_along_axis(value, idx, -1)


def simulate(close, positions, fee: float, slippage, margin: Margin | None = None, low=None, high=None,
             funding=None, periods_per_year: float = 365.0, detail: bool = False):
    """Per-candle strategy returns of ``positions`` under ``margin`` (see the module docstring).

    Same shapes and timing as :func:`tradebot.backtest.engine.simulate` (the position decided on candle ``t`` is
    held from its close to the close of ``t+1``). ``low``/``high``: the candles' extremes for the liquidation
    check (the close when None). ``funding``: historical funding per candle (fraction, shaped like ``close``),
    else ``margin.funding`` per 8 hours. ``detail``: also return the boolean mask of the candles in which a
    position was liquidated.
    """
    margin = margin or Margin()
    L, mm = margin.leverage, margin.maintenance
    positions = ops.nan_to_num(positions)
    held = ops.shift(positions, 1, fill=0.0) + 0.0 * close  # the position during each candle, broadcast
    prev_close = ops.shift(close, 1, fill=np.nan)
    prev_close = ops.where(ops.isnan(prev_close), close, prev_close)
    start, seg = _segments(held)
    entry = _at_start(prev_close + 0.0 * held, start)  # the fill at the close before the segment's first candle
    a = ops.abs_(held)
    s = ops.where(held > 0, 1.0, ops.where(held < 0, -1.0, 0.0))
    x = close / entry
    g = 1 + s * L * (x - 1)
    worst = ops.where(held > 0, close if low is None else low, close if high is None else high)
    xw = worst / entry
    liq = (a > 0) & (1 + s * L * (xw - 1) <= mm * L * xw)
    last_liq = ops.cummax(ops.where(liq, seg, 0.0))
    dead = (last_liq == seg) & (a > 0)
    g = ops.where(dead, 0.0, g)
    g_prev = ops.where(start, 1.0, ops.shift(g, 1, fill=1.0))
    x_prev = ops.where(start, 1.0, prev_close / entry)
    d_prev = 1 - a + a * g_prev  # equity before the candle, relative to the segment's start
    safe = ops.maximum(d_prev, 1e-12)
    out = ops.where(d_prev > 1e-12, a * (g - g_prev) / safe, 0.0)
    exposure = ops.where(g_prev > 0, s * a * L * x_prev / safe, 0.0)  # signed notional / equity
    # costs: the fee on the traded notional; nothing to close any more once a position was liquidated
    # and no new position while the target stays the one that was liquidated
    alive_held = ops.where(dead, 0.0, held)
    target = ops.where(dead & (positions == held), 0.0, positions)
    turnover = ops.abs_(target - alive_held) * L
    out = out - (fee + slippage) * turnover
    per_bar = periods_per_year
    if margin.borrow:
        out = out - margin.borrow / per_bar * ops.maximum(-exposure, 0.0)
    if funding is not None:
        out = out - exposure * ops.nan_to_num(funding)
    elif margin.funding:
        out = out - exposure * margin.funding * FUNDING_PER_YEAR / per_bar
    if detail:
        prev_dead = ops.shift(ops.where(dead, 1.0, 0.0), 1, fill=0.0) > 0.5
        return out, liq & ~(prev_dead & ~start)
    return out
