# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Automatic member selection (#21): a vote over a pool whose members the optimizer switches on and off.

The pool holds strategies and indicators. Strategies vote as in ``vote``.
An indicator alone does not trade, so it joins as a **filter** with a default
rule (:data:`RULES`): an entry also needs every active filter's rule to hold.
Indicators without a sensible rule are left out of the pool.

Every member gets a ``use.<member>`` gene (1 = on, 0 = off) next to its
parameters. :meth:`PoolEnsemble.canonical` enforces the constraints
(``min_members``/``max_members``, at least one strategy) and resets the
parameters of unused members to their defaults, so all genomes with the same
active subset share one cached evaluation. ``penalty`` is subtracted from the
fitness per active member: with equal scores, the smaller combination wins.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from tradebot.algorithms.base import ALGORITHMS, BaseAlgorithm
from tradebot.algorithms.builtin import VotingEnsemble
from tradebot.analysis.elliott import STATE_WAVE3
from tradebot.core import ops
from tradebot.core.registry import IntParam, Param
from tradebot.core.series import OHLC
from tradebot.indicators.base import INDICATORS
from tradebot.indicators.builtin import EMA, MACD, RSI, SMA, Bollinger, Donchian, Stochastic
from tradebot.indicators.elliott import ElliottWaves


@dataclass(frozen=True)
class Rule:
    """Default condition of a pool indicator: while it holds, the indicator allows an entry."""

    description: str
    params: dict[str, Param]
    condition: Callable[[OHLC, dict], object]
    vectorized: bool = True


RULES: dict[str, Rule] = {
    "sma": Rule("Trend filter: buy only while the close is above its simple moving average",
                {"period": IntParam(50, 5, 300, "Candles in the moving average the close must be above.")},
                lambda o, p: o.close > SMA.compute(o, period=p["period"])["value"]),
    "ema": Rule("Trend filter: buy only while the close is above its exponential moving average",
                {"period": IntParam(50, 5, 300, "Candles in the moving average the close must be above.")},
                lambda o, p: o.close > EMA.compute(o, period=p["period"])["value"]),
    "rsi": Rule("Oversold: buy only while the RSI is below a level",
                {"period": IntParam(14, 2, 50, "How many candles the RSI looks back."),
                 "level": Param(40.0, 10.0, 60.0, "Buy only while the RSI is below this.")},
                lambda o, p: RSI.compute(o, period=p["period"])["value"] < p["level"]),
    "stoch": Rule("Oversold: buy only while the stochastic %K is below a level",
                  {"period": IntParam(14, 2, 50, "Candles of the high-low range %K is measured in."),
                   "level": Param(30.0, 5.0, 60.0, "Buy only while %K is below this.")},
                  lambda o, p: Stochastic.compute(o, period=p["period"], smooth=1)["k"] < p["level"]),
    "macd": Rule("Momentum: buy only while the MACD line is above zero (short average above the long one)",
                 {"fast": IntParam(12, 2, 50, "Candles in the short average."),
                  "slow": IntParam(26, 5, 120, "Candles in the long average.")},
                 lambda o, p: MACD.compute(o, fast=p["fast"], slow=p["slow"], signal=2)["macd"] > 0),
    "bbands": Rule("Breakout: buy only while the close is above the upper Bollinger band",
                   {"period": IntParam(20, 5, 100, "Candles in the average the bands are drawn around."),
                    "width": Param(1.0, 0.0, 3.0, "Distance of the upper band from the average, in standard "
                                                  "deviations.")},
                   lambda o, p: o.close > Bollinger.compute(o, period=p["period"], width=p["width"])["upper"]),
    "donchian": Rule("Breakout: buy only while the close is above the previous highest high",
                     {"period": IntParam(20, 5, 200, "Candles the highest high is taken over.")},
                     lambda o, p: o.close > ops.shift(Donchian.compute(o, period=p["period"])["upper"], 1)),
    "elliott": Rule("Elliott: buy only while wave 3 is expected or running",
                    {"atr_period": IntParam(14, 5, 50, "ATR period for the swing size."),
                     "swing_atr": Param(4.0, 1.5, 10.0, "Minimum swing size in ATRs (zigzag sensitivity).")},
                    lambda o, p: ElliottWaves.compute(o, atr_period=p["atr_period"], swing_atr=p["swing_atr"],
                                                      min_score=0.0)["state"] == STATE_WAVE3,
                    vectorized=False),
}


class IndicatorFilter(BaseAlgorithm):
    """An indicator with its default rule, as a pool member. Not registered as an algorithm."""

    poolable = False

    def __init__(self, indicator: str, **params):
        self.name = indicator
        self.rule = RULES[indicator]
        self.description = self.rule.description
        super().__init__(**params)

    @property
    def vectorized(self) -> bool:
        return self.rule.vectorized

    def param_space(self):
        return dict(self.rule.params)

    def signals(self, o, p):
        return ops.where(self.rule.condition(o, p), 1.0, 0.0)

    def chart_indicators(self):
        cls = INDICATORS.get(self.name)
        return [cls(**{k: v for k, v in self.values.items() if k in cls.params})]


def pool_member(name: str) -> BaseAlgorithm:
    """A strategy by name, or an indicator with a default rule as a filter."""
    if name in ALGORITHMS:
        cls = ALGORITHMS.get(name)
        if not cls.poolable:
            raise ValueError(f"{name!r} can't be a pool member")
        return cls()
    if name in RULES and name in INDICATORS:
        return IndicatorFilter(name)
    if name in INDICATORS:
        raise ValueError(f"indicator {name!r} has no default rule, so it can't be a pool member")
    raise ValueError(f"unknown strategy or indicator {name!r}")


def default_pool() -> tuple[list[str], list[str]]:
    """``(members, skipped)``: every poolable strategy and every indicator with a default rule
    (plugins included), and the indicators left out for lack of a rule."""
    strategies = [cls.name for cls in ALGORITHMS if cls.poolable]
    indicators = INDICATORS.names()
    return strategies + [n for n in indicators if n in RULES], [n for n in indicators if n not in RULES]


class PoolEnsemble(VotingEnsemble):
    """Vote over the active strategies of a pool; active indicator filters gate the entries."""

    name = "pool"
    description = "Vote over a pool of strategies and indicator filters; the optimizer picks which ones to use"
    poolable = False

    def __init__(self, members: list[BaseAlgorithm | str] | None = None, *, min_members: int = 1,
                 max_members: int | None = None, penalty: float = 0.01, **params):
        pool = [pool_member(m) if isinstance(m, str) else m for m in (members or default_pool()[0])]
        if len({m.name for m in pool}) < len(pool):
            raise ValueError("pool members must be unique")
        strategies = [m for m in pool if not isinstance(m, IndicatorFilter)]
        if not strategies:
            raise ValueError("the pool needs at least one strategy; indicators only filter entries")
        if max_members is not None and not max(1, min_members) <= max_members:
            raise ValueError("need 1 <= max_members and min_members <= max_members")
        self.filters = [m for m in pool if isinstance(m, IndicatorFilter)]
        self.min_members, self.max_members, self.penalty = min_members, max_members, penalty
        super().__init__(strategies, **params)

    @property
    def pool(self) -> list[BaseAlgorithm]:
        return [*self.members, *self.filters]

    @property
    def vectorized(self) -> bool:
        return all(m.vectorized for m in self.pool)

    def vectorized_for(self, p: dict) -> bool:
        """Whether a parameter set's active members all take batched parameters."""
        return all(m.vectorized for m in self.active(p))

    def param_space(self):
        space = dict(self.params)
        for m in self.pool:
            space[f"use.{m.name}"] = IntParam(1, 0, 1, f"1 = use {m.name}, 0 = leave it out. The optimizer decides.")
            space.update({f"{m.name}.{k}": v for k, v in m.param_space().items()})
        return space

    def active(self, p: dict, scores: dict | None = None) -> list[BaseAlgorithm]:
        """Members switched on in ``p``, after applying the constraints.

        ``scores`` (the optimizer's raw genes in ``[0, 1]``) decide which members are added or dropped
        first; a member without a score (e.g. a fixed gene) counts as strongly on or off.
        """
        scores = scores or {}

        def score(m):
            key = f"use.{m.name}"
            return scores.get(key, 2.0 if p.get(key, 1) else -1.0)

        on = [m for m in self.pool if p.get(f"use.{m.name}", 1)]
        if not any(m in self.members for m in on):
            on.append(max(self.members, key=score))
        for m in sorted(on, key=score):  # too many: drop the weakest, never the last strategy
            if len(on) <= (self.max_members or len(self.pool)):
                break
            if m in self.filters or sum(x in self.members for x in on) > 1:
                on.remove(m)
        for m in sorted(self.pool, key=score, reverse=True):  # too few: add the strongest
            if len(on) >= self.min_members:
                break
            if m not in on:
                on.append(m)
        return [m for m in self.pool if m in on]

    def canonical(self, p: dict, scores: dict | None = None) -> dict:
        """``p`` with the constraints applied and the parameters of unused members at their defaults,
        so equivalent subsets share one cache entry."""
        active = self.active(p, scores)
        out = dict(p)
        for m in self.pool:
            out[f"use.{m.name}"] = int(m in active)
            if m not in active:
                out.update({f"{m.name}.{k}": v.clip(v.default) for k, v in m.param_space().items()})
        if sum(m in self.members for m in active) == 1:
            out["threshold"] = self.params["threshold"].default  # a single voter makes the threshold irrelevant
        return out

    def complexity(self, p: dict) -> float:
        """Fitness penalty of a parameter set: ``penalty`` per active member."""
        return self.penalty * len(self.active(p))

    def selected(self, p: dict | None = None) -> list[str]:
        return [m.name for m in self.active(self.values if p is None else p)]

    def signals(self, o, p):
        active = self.active(p)
        votes, entry = 0.0, True
        strategies = [m for m in active if m in self.members]
        for m in active:
            mp = {k: p.get(f"{m.name}.{k}", v) for k, v in m.values.items()}
            sig = ops.nan_to_num(m.signals(o, mp)) > 0.5
            if m in self.members:
                votes = votes + sig
            else:
                entry = entry & sig
        share = votes / len(strategies)
        return ops.latch((share >= p["threshold"]) & entry, (1 - share) >= p["threshold"])

    def chart_indicators(self):
        return [ind for m in self.active(self.values) for ind in m.chart_indicators()]

    def reuse_params(self, p: dict) -> dict:
        """The parameters that matter for the selected members (what ``backtest pool --members ...`` needs)."""
        names = set(self.selected(p))
        return {k: v for k, v in p.items() if not k.startswith("use.") and (k == "threshold" or k.partition(".")[0] in names)}
