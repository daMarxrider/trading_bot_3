# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""The building blocks evolved rules are made of (#83): a strongly typed primitive set.

Every sub-rule has a **unit**, and a primitive only accepts the units it makes sense for, so every tree
the search builds is a valid rule that also means the same on every symbol:

* ``bool`` - a condition (the entry and exit rules themselves)
* ``price`` - open/high/low/close, moving averages, bands, channels; compared with other prices only,
  never with a fixed number (an absolute price doesn't carry over to another symbol)
* ``osc`` - 0-100 oscillators (RSI, stochastic); compared with each other or with a level 0-100
* ``ratio`` - relative changes (0.05 = 5 %), e.g. ``change(close, 20)``, ``gap(close, sma(close, 50))``
* ``dist`` - a price distance (ATR), used as ``price + k * atr(n)`` (stops and bands)
* an indicator's own unit (e.g. ``macd``): comparable with its other outputs and its ``levels``

Constants (window lengths, offsets, levels, indicator parameters) are leaves with a search range.
The exit rule can also look at the open trade (``bars_in_trade``, stop loss, take profit, trailing
stop); those need a loop over trades instead of one vectorized latch, see :func:`runtime.in_trade`.

Indicators come from the registry (plugins included). An output is used only if it is **causal**: its
values on a prefix of the data equal those on the full data, which drops hindsight outputs like the
Elliott zigzag. Hidden outputs and label outputs are left out unless the indicator declares a unit
for them (:attr:`BaseIndicator.units`).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Callable

import numpy as np

from tradebot.core import ops
from tradebot.core.registry import IntParam, Param
from tradebot.core.series import OHLC
from tradebot.evolve import runtime as rt
from tradebot.indicators.base import INDICATORS, BaseIndicator

BOOL, PRICE, OSC, RATIO, DIST = "bool", "price", "osc", "ratio", "dist"

WINDOW = IntParam(20, 2, 200, "candles")
LAG = IntParam(1, 1, 20, "candles back")
MULT = Param(2.0, -6.0, 6.0, "multiple of the ATR")
FRACTION = Param(0.05, 0.005, 0.5, "fraction of the price (0.05 = 5 %)")
HOLD = IntParam(10, 1, 250, "candles in the trade")
#: The level a unit can be compared with (units without one are only compared with each other).
LEVELS = {OSC: Param(50.0, 0.0, 100.0, "level"), RATIO: Param(0.0, -0.2, 0.2, "level")}


@dataclass(frozen=True, eq=False)
class Primitive:
    """One function of the rule language.

    ``args`` holds a unit for every sub-rule and a :class:`Param` for every constant. ``kind`` says
    how ``fn`` is called: ``"op"`` with the argument values, ``"data"`` with the candles and the
    constants, ``"state"`` with the open trade and the constants (exit rules only), ``"indicator"``
    reads output ``indicator[1]`` of indicator ``indicator[0]``. ``text`` and ``code`` are format
    strings for the readable rule and the generated Python.
    """

    name: str  #: unique; the part before ":" is the family shared by all units ("gt:osc" -> "gt")
    ret: str
    args: tuple
    fn: Callable | None
    text: str
    code: str
    kind: str = "op"
    weight: float = 1.0
    prec: int = 9  #: binding strength of ``text`` (or 1 < and 2 < not 3 < comparison 4 < +/- 5 < call 9)
    indicator: tuple[str, str, tuple[str, ...]] | None = None  #: (name, output, parameter names)

    @property
    def family(self) -> str:
        return self.name.partition(":")[0]

    @property
    def subrules(self) -> list[int]:
        """Positions of the arguments that are sub-rules (the rest are constants)."""
        return [i for i, a in enumerate(self.args) if isinstance(a, str)]

    @property
    def terminal(self) -> bool:
        return not self.subrules


#: families whose two sub-rules can be swapped without changing the result (sorted when simplifying)
COMMUTATIVE = {"and", "or", "max", "min", "mid", "add"}
#: families that compare two sub-rules (``x op x`` is never true)
COMPARISONS = {"gt", "lt", "crosses_above", "crosses_below"}
LEVEL_COMPARISONS = {"above", "below", "crosses_above_level", "crosses_below_level"}


def _generic(unit: str, level: Param | None) -> list[Primitive]:
    """Transforms and comparisons every numeric unit gets."""
    u = unit
    out = [
        Primitive(f"sma:{u}", u, (u, WINDOW), ops.sma, "sma({0}, {1})", "ops.sma({0}, {1})", weight=0.5),
        Primitive(f"ema:{u}", u, (u, WINDOW), ops.ema, "ema({0}, {1})", "ops.ema({0}, {1})", weight=0.5),
        Primitive(f"highest:{u}", u, (u, WINDOW), ops.rolling_max, "highest({0}, {1})", "ops.rolling_max({0}, {1})",
                  weight=0.35),
        Primitive(f"lowest:{u}", u, (u, WINDOW), ops.rolling_min, "lowest({0}, {1})", "ops.rolling_min({0}, {1})",
                  weight=0.35),
        Primitive(f"shift:{u}", u, (u, LAG), ops.shift, "shift({0}, {1})", "ops.shift({0}, {1})", weight=0.3),
        Primitive(f"max:{u}", u, (u, u), ops.maximum, "max({0}, {1})", "ops.maximum({0}, {1})", weight=0.15),
        Primitive(f"min:{u}", u, (u, u), ops.minimum, "min({0}, {1})", "ops.minimum({0}, {1})", weight=0.15),
        Primitive(f"if_else:{u}", u, (BOOL, u, u), ops.where, "if_else({0}, {1}, {2})", "ops.where({0}, {1}, {2})",
                  weight=0.1),
        Primitive(f"gt:{u}", BOOL, (u, u), lambda a, b: a > b, "{0} > {1}", "({0} > {1})", prec=4),
        Primitive(f"lt:{u}", BOOL, (u, u), lambda a, b: a < b, "{0} < {1}", "({0} < {1})", prec=4),
        Primitive(f"crosses_above:{u}", BOOL, (u, u), ops.cross_above, "crosses_above({0}, {1})",
                  "ops.cross_above({0}, {1})"),
        Primitive(f"crosses_below:{u}", BOOL, (u, u), ops.cross_below, "crosses_below({0}, {1})",
                  "ops.cross_below({0}, {1})"),
    ]
    if level is not None:
        out += [
            Primitive(f"above:{u}", BOOL, (u, level), lambda a, c: a > c, "{0} > {1}", "({0} > {1})", prec=4),
            Primitive(f"below:{u}", BOOL, (u, level), lambda a, c: a < c, "{0} < {1}", "({0} < {1})", prec=4),
            Primitive(f"crosses_above_level:{u}", BOOL, (u, level), rt.cross_above_level, "crosses_above({0}, {1})",
                      "rt.cross_above_level({0}, {1})"),
            Primitive(f"crosses_below_level:{u}", BOOL, (u, level), rt.cross_below_level, "crosses_below({0}, {1})",
                      "rt.cross_below_level({0}, {1})"),
        ]
    return out


def _field(name: str) -> Primitive:
    return Primitive(name, PRICE, (), lambda o, _f=name: getattr(o, _f), name, f"o.{name}", kind="data")


def _core(units: set[str]) -> list[Primitive]:
    """Candles, logic, trade state and the unit-specific arithmetic."""
    out = [_field(f) for f in ("open", "high", "low", "close")]
    out += [
        Primitive("always", BOOL, (), rt.always, "always", "rt.always(o)", kind="data", weight=0.0),
        Primitive("never", BOOL, (), rt.never, "never", "rt.never(o)", kind="data", weight=0.0),
        Primitive("and", BOOL, (BOOL, BOOL), lambda a, b: a & b, "{0} and {1}", "({0} & {1})", weight=1.5, prec=2),
        Primitive("or", BOOL, (BOOL, BOOL), lambda a, b: a | b, "{0} or {1}", "({0} | {1})", weight=1.2, prec=1),
        Primitive("not", BOOL, (BOOL,), lambda a: ~a, "not {0}", "(~{0})", weight=0.4, prec=3),
        Primitive("if_else:bool", BOOL, (BOOL, BOOL, BOOL), ops.where, "if_else({0}, {1}, {2})",
                  "ops.where({0}, {1}, {2})", weight=0.1),
        Primitive("held", BOOL, (HOLD,), lambda t, n: t.bars >= n, "bars_in_trade >= {0}", "(t.bars >= {0})",
                  kind="state", weight=0.4, prec=4),
        Primitive("stop", BOOL, (FRACTION,), lambda t, f: t.stop(f), "stop_loss({0})", "t.stop({0})", kind="state",
                  weight=0.4),
        Primitive("target", BOOL, (FRACTION,), lambda t, f: t.target(f), "take_profit({0})", "t.target({0})",
                  kind="state", weight=0.3),
        Primitive("trail", BOOL, (FRACTION,), lambda t, f: t.trail(f), "trailing_stop({0})", "t.trail({0})",
                  kind="state", weight=0.4),
        Primitive("mid:price", PRICE, (PRICE, PRICE), lambda a, b: (a + b) / 2, "mid({0}, {1})", "(({0} + {1}) / 2)",
                  weight=0.15),
        Primitive("change:ratio", RATIO, (PRICE, WINDOW), rt.change, "change({0}, {1})", "rt.change({0}, {1})"),
        Primitive("gap:ratio", RATIO, (PRICE, PRICE), lambda a, b: a / b - 1, "gap({0}, {1})", "({0} / {1} - 1)"),
        Primitive("volatility:ratio", RATIO, (WINDOW,), rt.volatility, "volatility({0})", "rt.volatility(o, {0})",
                  kind="data", weight=0.5),
        Primitive("add:ratio", RATIO, (RATIO, RATIO), lambda a, b: a + b, "{0} + {1}", "({0} + {1})", weight=0.15,
                  prec=5),
        Primitive("sub:ratio", RATIO, (RATIO, RATIO), lambda a, b: a - b, "{0} - {1}", "({0} - {1})", weight=0.15,
                  prec=5),
        Primitive("abs:ratio", RATIO, (RATIO,), ops.abs_, "abs({0})", "ops.abs_({0})", weight=0.15),
    ]
    if DIST in units:
        out.append(Primitive("offset:price", PRICE, (PRICE, DIST, MULT), lambda p, d, k: p + k * d,
                             "{0} + {2} * {1}", "({0} + {2} * {1})", weight=0.4, prec=5))
    return out


# -- indicators ---------------------------------------------------------------------------------------
def _sample_ohlc(n: int, seed: int = 7) -> OHLC:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    open_ = np.r_[close[0], close[:-1]]
    spread = np.abs(rng.normal(0, 0.01, n)) * close
    return OHLC(open_, np.maximum(open_, close) + spread, np.minimum(open_, close) - spread, close)


def output_unit(cls: type[BaseIndicator], output: str) -> str | None:
    """What an indicator output measures (see :attr:`BaseIndicator.units`); None = not usable in rules."""
    if declared := cls.units.get(output):
        return declared
    if output in cls.hidden or output in cls.labels:
        return None
    if cls.overlay:
        return PRICE
    if cls.levels and all(0 < lv < 100 for lv in cls.levels):
        return OSC
    return cls.name


@lru_cache(maxsize=None)
def causal_outputs(cls: type[BaseIndicator]) -> tuple[str, ...]:
    """The outputs whose values on the first candles don't change when later candles are added."""
    full, part = _sample_ohlc(400), 250
    try:
        defaults = {k: v.default for k, v in cls.params.items()}
        with np.errstate(all="ignore"):
            a = cls.compute(full, **defaults)
            b = cls.compute(OHLC(*(x[:part] for x in (full.open, full.high, full.low, full.close))), **defaults)
    except Exception:  # a broken plugin indicator is simply not offered
        return ()
    return tuple(k for k in cls.outputs if k in a and k in b and np.allclose(
        np.asarray(a[k], float)[:part], np.asarray(b[k], float), rtol=1e-7, atol=1e-9, equal_nan=True))


def indicator_outputs(names: list[str] | None = None) -> list[tuple[type[BaseIndicator], str, str]]:
    """``(indicator, output, unit)`` of every usable output (of ``names``, default: every registered one)."""
    classes = [INDICATORS.get(n) for n in names] if names is not None else list(INDICATORS)
    out = []
    for cls in classes:
        for output in causal_outputs(cls):
            if (unit := output_unit(cls, output)) is not None:
                out.append((cls, output, unit))
    return out


def _indicator(cls: type[BaseIndicator], output: str, unit: str) -> Primitive:
    keys = tuple(cls.params)
    args = ", ".join("{%d}" % i for i in range(len(keys)))
    call = f"{cls.name}({args})" if keys else cls.name
    text = call if cls.outputs == ("value",) else f"{call}.{output}"
    kw = ", ".join(f"{k}={{{i}}}" for i, k in enumerate(keys))
    code = f'rt.indicator("{cls.name}", o{", " + kw if kw else ""})["{output}"]'
    return Primitive(f"{cls.name}.{output}:{unit}", unit, tuple(cls.params[k] for k in keys), None, text, code,
                     kind="indicator", indicator=(cls.name, output, keys))


def _level(unit: str, cls: type[BaseIndicator] | None) -> Param | None:
    if unit in LEVELS:
        return LEVELS[unit]
    if cls is not None and cls.levels:
        lo, hi = min(cls.levels), max(cls.levels)
        return Param((lo + hi) / 2, lo, hi, "level")
    return None


# -- the set ------------------------------------------------------------------------------------------
def _min_depths(primitives: list[Primitive]) -> dict[str, int]:
    """Smallest tree depth each primitive can be built with (a fixed point over the units)."""
    unit_depth: dict[str, int] = {}
    prim_depth: dict[str, int] = {}
    changed = True
    while changed:
        changed = False
        for p in primitives:
            subs = [p.args[i] for i in p.subrules]
            if all(u in unit_depth for u in subs):
                d = 1 + max((unit_depth[u] for u in subs), default=0)
                if d < prim_depth.get(p.name, 1 << 30):
                    prim_depth[p.name] = d
                    changed = True
                if p.weight > 0 and d < unit_depth.get(p.ret, 1 << 30):
                    unit_depth[p.ret] = d
                    changed = True
    return prim_depth


class PrimitiveSet:
    """The primitives of one search, indexed by name and by unit, with the depth each one needs."""

    def __init__(self, primitives: list[Primitive], state: bool = True):
        self.state = state
        primitives = [p for p in primitives if state or p.kind != "state"]
        #: smallest depth (counting sub-rules, not constants) per primitive: in exit rules, in entry rules
        self.min_depth = _min_depths(primitives)
        self.entry_depth = _min_depths([p for p in primitives if p.kind != "state"])
        # a primitive whose arguments can't all be built (e.g. a unit nothing produces) is never offered
        self.primitives = [p for p in primitives if p.name in self.min_depth]
        self.by_name = {p.name: p for p in self.primitives}
        self.by_ret: dict[str, list[Primitive]] = {}
        for p in self.primitives:
            self.by_ret.setdefault(p.ret, []).append(p)

    def __getitem__(self, name: str) -> Primitive:
        return self.by_name[name]

    def __contains__(self, name: str) -> bool:
        return name in self.by_name

    def candidates(self, unit: str, depth: int, state: bool) -> list[Primitive]:
        """Primitives returning ``unit`` that fit into ``depth`` levels (``state``: in an exit rule)."""
        table = self.min_depth if state else self.entry_depth
        return [p for p in self.by_ret.get(unit, []) if p.weight > 0 and table.get(p.name, depth + 1) <= depth]

    def same_signature(self, p: Primitive) -> list[Primitive]:
        """Other primitives taking and returning the same (for point mutation: gt <-> lt, sma <-> ema, ...)."""
        return [q for q in self.by_ret.get(p.ret, []) if q is not p and q.weight > 0 and q.kind == p.kind
                and q.args == p.args and q.name in self.min_depth]

    @classmethod
    def default(cls, indicators: list[str] | None = None, state: bool = True) -> "PrimitiveSet":
        """Candles, logic and arithmetic, plus every usable indicator output (or those of ``indicators``)."""
        found = indicator_outputs(indicators)
        units = {PRICE, OSC, RATIO} | {u for _, _, u in found}
        prims = _core(units)
        owner = {u: c for c, _, u in found if u not in (PRICE, OSC, RATIO, DIST)}
        for unit in sorted(units):
            prims += _generic(unit, _level(unit, owner.get(unit)))
        # the indicator outputs of one unit share a weight, so many outputs don't crowd out the rest; so do
        # the comparisons, so the many units don't crowd out and/or
        indicator_prims = [_indicator(c, out, u) for c, out, u in found]
        per_unit = {u: sum(p.ret == u for p in indicator_prims) for u in units}
        indicator_prims = [replace(p, weight=2.0 / per_unit[p.ret]) for p in indicator_prims]
        comparisons = sum(p.family in COMPARISONS | LEVEL_COMPARISONS for p in prims)
        prims = [replace(p, weight=6.0 / comparisons) if p.family in COMPARISONS | LEVEL_COMPARISONS else p
                 for p in prims]
        return cls(prims + indicator_prims, state)
