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


# -- the low-level maths of the from-scratch set (#100) ------------------------------------------------
class Undefined(ArithmeticError):
    """A from-scratch formula divided a non-zero value by zero (or overflowed): the program is rejected."""


def const(o, value):
    """A constant as a series (so every operation accepts it)."""
    return ops.full(tuple(o.close.shape), float(value), o.close)


def diff(x, n: int):
    """``x`` minus ``x`` ``n`` candles ago."""
    return x - ops.shift(x, n)


def pos(x):
    """``max(x, 0)``; NaN stays NaN (so a smoothing seeds on the first real value)."""
    return ops.where(x < 0, 0.0, x)


def neg(x):
    """``max(-x, 0)``; NaN stays NaN."""
    return ops.where(x > 0, 0.0, -x)


def rolling_sum(x, n: int):
    """Sum over ``n`` candles (NaN while the window holds a NaN)."""
    return ops.sma(x, n) * n


def div(a, b):
    """``a / b`` as a generated plugin computes it: NaN (no signal) where ``b`` is 0 instead of inf."""
    with np.errstate(divide="ignore", invalid="ignore"):
        out = a / ops.where(b == 0, ops.NAN, b)
    return ops.where(ops.isnan(b) | (b != 0), out, ops.NAN)


def div_checked(a, b):
    """:func:`div` while evolving: ``x / 0`` with ``x != 0`` raises :class:`Undefined`, so a formula that blows up
    on real data is rejected; ``0 / 0`` (e.g. a stochastic of flat candles) is a gap, like a warm-up."""
    a_ = np.broadcast_to(np.asarray(a, dtype=float), np.shape(b)) if np.ndim(b) else np.asarray(a, dtype=float)
    if np.any((np.asarray(b) == 0) & (a_ != 0) & ~np.isnan(a_)):
        raise Undefined("division by zero")
    return finite(div(a, b))


def finite(x):
    """``x`` unless it holds an infinity (an overflow), which raises :class:`Undefined`."""
    if np.isinf(np.asarray(x, dtype=float)).any():
        raise Undefined("overflow")
    return x


def squash(x):
    """A ratio onto 0-100: ``100 * x / (1 + |x|)``; for a ratio ``rs`` >= 0 this is ``100 - 100 / (1 + rs)``, RSI's form."""
    return 100 * x / (1 + ops.abs_(x))


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
