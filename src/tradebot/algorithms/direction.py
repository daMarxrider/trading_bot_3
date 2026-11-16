# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Long/short variants of any strategy (#149).

The built-in strategies and combos say when to be *long*. :class:`Directed` wraps one and trades it in another
direction:

* ``long``: unchanged (the default);
* ``short``: short only, on the strategy's short signal;
* ``long_short``: long on its long signal, short on its short signal (stop and reverse for crossovers).

The short signal is :meth:`BaseAlgorithm.short_signals`: by default the mirror image of the long one (short
whenever the long version is out of the market, after its first entry), with real mirror rules where that
differs (``bollinger_reversion`` shorts a close above the upper band, ``ichimoku_cloud`` a close below the cloud).

A short position is backtested with the margin model (:mod:`tradebot.backtest.margin`: fixed units, a short is
liquidated when the price runs away, borrow fees and funding). Wrap it in a sizing model as usual:
``Sized(Directed(algo, "long_short"), ...)``. The live bot does not open shorts.
"""

from __future__ import annotations

from tradebot.algorithms.base import BaseAlgorithm
from tradebot.core import ops
from tradebot.core.series import OHLC

DIRECTIONS = {
    "long": "Long only",
    "short": "Short only: short on the strategy's sell signal",
    "long_short": "Long and short: long on the buy signal, short on the sell signal",
}


class Directed(BaseAlgorithm):
    """``inner`` traded short or long and short. Not registered as an algorithm."""

    poolable = False
    allow_short = True

    def __init__(self, inner: BaseAlgorithm, direction: str):
        if direction not in DIRECTIONS or direction == "long":
            raise ValueError(f"unknown direction {direction!r}; choose {', '.join(DIRECTIONS)}")
        if getattr(inner, "fills", "close") == "intrabar":
            raise ValueError("shorts cannot be combined with intrabar fills yet; use fills=close")
        self.inner, self.direction = inner, direction
        self.name = inner.name
        self.description = f"{inner.description} ({DIRECTIONS[direction].split(':')[0].lower()})"
        self.values = dict(inner.values)

    @property
    def vectorized(self) -> bool:
        return self.inner.vectorized

    @property
    def period_batched(self) -> bool:
        return self.inner.period_batched

    def vectorized_for(self, p: dict) -> bool:
        f = getattr(self.inner, "vectorized_for", None)
        return f(p) if f else self.inner.vectorized

    def __getattr__(self, name):  # e.g. a combo's ``fills``, a pool's ``selected``
        if name == "inner" or name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.inner, name)

    def param_space(self):
        return self.inner.param_space()

    def with_params(self, **params) -> "Directed":
        return Directed(self.inner.with_params(**params), self.direction)

    def signals(self, o: OHLC, p: dict):
        return self.execute(o, p)[0]

    def execute(self, o: OHLC, p: dict):
        long, fills = self.inner.execute(o, p)
        if fills is not None:
            raise ValueError("shorts cannot be combined with intrabar fills yet; use fills=close")
        long = ops.nan_to_num(long)
        short = ops.nan_to_num(self.inner.short_signals(o, p, long))
        if self.direction == "short":
            return -short, None
        return long - short, None

    def short_signals(self, o, p, long):
        return self.inner.short_signals(o, p, long)

    def chart_indicators(self):
        return self.inner.chart_indicators()

    def reuse_params(self, params: dict) -> dict:
        return self.inner.reuse_params(params) if hasattr(self.inner, "reuse_params") else params

    def __repr__(self) -> str:
        return f"{self.inner!r} {self.direction.replace('_', '/')}"


def directed(algorithm: BaseAlgorithm, direction: str | None) -> BaseAlgorithm:
    """``algorithm`` traded in ``direction`` (None or "long": unchanged)."""
    if not direction or direction == "long":
        return algorithm
    return Directed(algorithm, direction)
