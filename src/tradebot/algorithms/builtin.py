# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Built-in strategies, ported and cleaned up from the legacy bots.

* ``rsi_threshold``, ``macd_cross``, ``ichimoku_cloud`` and the ``vote`` ensemble
  come from coinbase-pro-trader (``Algorithms/Trading/trade_all_indicators.py``:
  RSI 33/66, MACD vs. signal, close above both cloud spans, 70 % majority vote).
* ``range`` is the binance-trader "range" mode (buy <= X, sell >= Y), with the
  optional stop-loss of its "profit" mode.
"""

from __future__ import annotations

from tradebot.algorithms.base import ALGORITHMS, BaseAlgorithm
from tradebot.core import ops
from tradebot.core.registry import IntParam, Param
from tradebot.indicators.builtin import MACD, RSI, SMA, Bollinger, Ichimoku


class RsiThreshold(BaseAlgorithm):
    name = "rsi_threshold"
    description = "Buy when RSI falls below `lower`, sell when it rises above `upper`"
    params = {
        "period": IntParam(14, 2, 50),
        "lower": Param(33.0, 5.0, 50.0),
        "upper": Param(66.0, 50.0, 95.0),
    }

    def signals(self, o, p):
        rsi = RSI.compute(o, period=p["period"])["value"]
        return ops.latch(rsi < p["lower"], rsi > p["upper"])

    def chart_indicators(self):
        return [RSI(period=self.values["period"])]


class MacdCross(BaseAlgorithm):
    name = "macd_cross"
    description = "Long while the MACD line is above its signal line"
    params = {"fast": IntParam(12, 2, 50), "slow": IntParam(26, 5, 120), "signal": IntParam(9, 2, 50)}

    def signals(self, o, p):
        m = MACD.compute(o, fast=p["fast"], slow=p["slow"], signal=p["signal"])
        return ops.where(m["macd"] > m["signal"], 1.0, 0.0)

    def chart_indicators(self):
        return [MACD(**self.values)]


class IchimokuCloud(BaseAlgorithm):
    name = "ichimoku_cloud"
    description = "Long while the close is above both cloud spans"
    params = Ichimoku.params

    def signals(self, o, p):
        ich = Ichimoku.compute(o, **p)
        return ops.where((o.close > ich["span_a"]) & (o.close > ich["span_b"]), 1.0, 0.0)

    def chart_indicators(self):
        return [Ichimoku(**self.values)]


class SmaCross(BaseAlgorithm):
    name = "sma_cross"
    description = "Long while the fast SMA is above the slow SMA"
    params = {"fast": IntParam(20, 2, 100), "slow": IntParam(50, 10, 300)}

    def signals(self, o, p):
        fast = SMA.compute(o, period=p["fast"])["value"]
        slow = SMA.compute(o, period=p["slow"])["value"]
        return ops.where(fast > slow, 1.0, 0.0)

    def chart_indicators(self):
        return [SMA(period=self.values["fast"]), SMA(period=self.values["slow"])]


class BollingerReversion(BaseAlgorithm):
    name = "bollinger_reversion"
    description = "Buy a close below the lower band, exit at the middle band"
    params = {"period": IntParam(20, 5, 100), "width": Param(2.0, 1.0, 3.5)}

    def signals(self, o, p):
        bb = Bollinger.compute(o, period=p["period"], width=p["width"])
        return ops.latch(o.close < bb["lower"], o.close > bb["middle"])

    def chart_indicators(self):
        return [Bollinger(**self.values)]


class RangeTrader(BaseAlgorithm):
    """Legacy binance-trader range mode: absolute buy/sell prices plus optional stop price."""

    name = "range"
    description = "Buy at or below `buy_price`, sell at or above `sell_price` (or below `stop_price`)"
    params = {
        "buy_price": Param(0.0, 0.0, 1e9),
        "sell_price": Param(1e9, 0.0, 1e9),
        "stop_price": Param(0.0, 0.0, 1e9),
    }

    def signals(self, o, p):
        exit_ = (o.high >= p["sell_price"]) | (o.low < p["stop_price"])
        return ops.latch(o.low <= p["buy_price"], exit_)


class VotingEnsemble(BaseAlgorithm):
    """Majority vote of member algorithms (coinbase-pro-trader's 70 % rule).

    Enter when at least ``threshold`` of the members are long, exit when at
    least ``threshold`` are flat, hold otherwise. Member parameters are exposed
    as ``<member>.<param>`` so the ensemble can be optimized as a whole.
    """

    name = "vote"
    description = "Majority vote over several algorithms"
    params = {"threshold": Param(0.7, 0.5, 1.0)}

    def __init__(self, members: list[BaseAlgorithm] | None = None, **params):
        self.members = members or [RsiThreshold(), MacdCross(), IchimokuCloud()]
        super().__init__(**params)

    @property
    def vectorized(self) -> bool:
        # batched [group, 1] parameters reach every member, so one loop-based member makes the whole vote loop-based
        return all(m.vectorized for m in self.members)

    def param_space(self):
        space = dict(self.params)
        for m in self.members:
            space.update({f"{m.name}.{k}": v for k, v in m.param_space().items()})
        return space

    def signals(self, o, p):
        votes = 0.0
        for m in self.members:
            mp = {k: p.get(f"{m.name}.{k}", v) for k, v in m.values.items()}
            votes = votes + (ops.nan_to_num(m.signals(o, mp)) > 0.5)
        share = votes / len(self.members)
        return ops.latch(share >= p["threshold"], (1 - share) >= p["threshold"])

    def chart_indicators(self):
        return [ind for m in self.members for ind in m.chart_indicators()]


def build_algorithm(name: str, params: dict | None = None, members: list[str] | None = None) -> BaseAlgorithm:
    """Instantiate a registered algorithm by name (``members`` only applies to ``vote``)."""
    cls = ALGORITHMS.get(name)
    params = params or {}
    if cls is VotingEnsemble:
        return VotingEnsemble([ALGORITHMS.get(m)() for m in members] if members else None, **params)
    return cls(**params)
