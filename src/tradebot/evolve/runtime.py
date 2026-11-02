# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Helpers shared by the evolver's evaluator and the plugin files it writes (#83).

Generated plugins import only this module, :mod:`tradebot.core.ops` and the registries, so they keep
working without the evolver itself.
"""

from __future__ import annotations

import numpy as np

from tradebot.core import ops
from tradebot.indicators.base import INDICATORS


def indicator(name: str, o, **params) -> dict:
    """``{output: array}`` of a registered indicator (looked up when called, so plugin indicators work too)."""
    return INDICATORS.get(name).compute(o, **params)


def always(o):
    c = o.close
    return ops.torch.ones_like(c, dtype=ops.torch.bool) if ops.is_torch(c) else np.ones(np.shape(c), bool)


def never(o):
    return ~always(o)


def cross_above_level(x, level):
    """True on the bar where ``x`` rises above the constant ``level``."""
    return (x > level) & (ops.shift(x, 1) <= level)


def cross_below_level(x, level):
    return (x < level) & (ops.shift(x, 1) >= level)


def change(x, n: int):
    """Relative change over ``n`` candles (0.05 = +5 %)."""
    return x / ops.shift(x, n) - 1


def volatility(o, n: int):
    """Standard deviation of the candle-to-candle returns over ``n`` candles."""
    return ops.rolling_std(change(o.close, 1), n)


class Trade:
    """The open trade an exit rule looks at: it started at the close of candle ``start``.

    Inside an exit rule every series is cut to the candles from the entry on (:meth:`cut`), so
    position 0 is the entry candle.
    """

    def __init__(self, close, start: int):
        self.start = start
        self.close = close[..., start:]
        self.entry_price = close[..., start]
        self.bars = np.arange(self.close.shape[-1])  #: candles since the entry (0 on the entry candle)

    def cut(self, x):
        """``x`` from the entry candle on (a constant stays as it is)."""
        return x[..., self.start:] if np.ndim(x) else x

    def stop(self, fraction):
        """The close fell ``fraction`` below the entry price (0.05 = 5 %)."""
        return self.close < self.entry_price * (1 - fraction)

    def target(self, fraction):
        """The close rose ``fraction`` above the entry price."""
        return self.close > self.entry_price * (1 + fraction)

    def trail(self, fraction):
        """The close fell ``fraction`` below the highest close since the entry (a trailing stop)."""
        return self.close < ops.cummax(self.close) * (1 - fraction)


def in_trade(entry, exit_rule, close) -> np.ndarray:
    """Positions (0/1) of a long-only strategy whose exit rule looks at the open trade.

    ``exit_rule(trade)`` returns a mask over the candles from the entry on. Same semantics as
    :func:`ops.latch`: in on an entry, out on an exit, and an exit on the entry candle itself wins.
    One NumPy evaluation per trade instead of a loop over candles.
    """
    close = np.asarray(close, dtype=float)
    n = close.shape[-1]
    entry = np.broadcast_to(np.nan_to_num(np.asarray(entry, dtype=float)) > 0, (n,))
    pos = np.zeros(n)
    t = 0
    while t < n:
        later = np.flatnonzero(entry[t:])
        if not len(later):
            break
        start = t + int(later[0])
        hits = np.flatnonzero(np.broadcast_to(np.asarray(exit_rule(Trade(close, start)), dtype=bool), (n - start,)))
        if len(hits) and hits[0] == 0:  # exit on the entry candle: no trade
            t = start + 1
            continue
        end = start + int(hits[0]) if len(hits) else n
        pos[start:end] = 1.0
        t = end + 1
    return pos
