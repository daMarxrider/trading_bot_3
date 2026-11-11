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

**From scratch** (#100, :meth:`PrimitiveSet.from_scratch`): no ready-made indicators, only candles and
low-level maths (``diff``, ``pos``/``neg``, rolling sums, deviations, Wilder smoothing, free constants,
``+ - * /``). The arithmetic is checked by dimensional analysis (:func:`arith_unit`), so a rule still means
the same on every symbol, while RSI, Stochastic, Bollinger, ATR or MACD can be built (and found) from the
parts. Subtrees that keep showing up in good programs become **modules** (``m1``, ``m2``, ...): named,
evolved indicators whose constants are its parameters, used as one building block from then on.

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
FACTOR = Param(1.0, -5.0, 5.0, "constant")  #: a free constant of the from-scratch set (multipliers, offsets)
#: The level a unit can be compared with (units without one are only compared with each other).
LEVELS = {OSC: Param(50.0, 0.0, 100.0, "level"), RATIO: Param(0.0, -0.2, 0.2, "level")}
#: from scratch, ratios are also normalised oscillators like ``diff / abs(diff)`` (RSI = 50 + 50 * that), not just returns
SCRATCH_LEVELS = {**LEVELS, RATIO: Param(0.0, -1.0, 1.0, "level")}


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
    #: a module (``kind`` "module"): the subtree it stands for, and per parameter the paths of its constants
    body: object = None
    slots: tuple = ()

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
COMMUTATIVE = {"and", "or", "max", "min", "mid", "add", "mul"}
#: families that compare two sub-rules (``x op x`` is never true)
COMPARISONS = {"gt", "lt", "crosses_above", "crosses_below"}
LEVEL_COMPARISONS = {"above", "below", "crosses_above_level", "crosses_below_level"}


def _generic(unit: str, level: Param | None, scratch: bool = False) -> list[Primitive]:
    """Transforms and comparisons every numeric unit gets (``scratch``: plus Wilder's smoothing)."""
    u = unit
    out = [Primitive(f"wilder:{u}", u, (u, WINDOW), ops.wilder, "wilder({0}, {1})", "ops.wilder({0}, {1})",
                     weight=0.5)] if scratch else []
    out += [
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


# -- from scratch (#100) ------------------------------------------------------------------------------
#: what a unit measures for the arithmetic: (power of the price, a level rather than a difference). A price is
#: a point on the price axis, a distance the gap between two points, a ratio has no dimension.
DIMENSION = {PRICE: (1, True), DIST: (1, False), RATIO: (0, False)}
ARITH = {"add": "+", "sub": "-", "mul": "*", "div": "/"}


def arith_unit(op: str, a: str, b: str) -> str | None:
    """The unit of ``a op b`` (``op`` add, sub, mul or div), or None when it means nothing.

    Dimensional analysis as in physics, with prices as points and distances as differences:
    ``price - price`` and ``price ± dist`` are fine, ``price + price`` is not; ``price / price`` and
    ``dist / price`` are ratios, ``price * ratio`` a price, ``price * price`` nothing.
    """
    if a not in DIMENSION or b not in DIMENSION or op not in ARITH:
        return None
    (pa, la), (pb, lb) = DIMENSION[a], DIMENSION[b]
    if op in ("add", "sub"):
        if pa != pb or (la and lb and op == "add") or (lb and not la):
            return None
        power, level = pa, (la or lb) if op == "add" else la and not lb
    elif op == "mul":
        power, level = pa + pb, la or lb
        if la and lb:
            return None
    else:
        power, level = pa - pb, la and not lb
    return {(0, False): RATIO, (1, True): PRICE, (1, False): DIST}.get((power, level))


def _arithmetic() -> list[Primitive]:
    """Every meaningful ``a op b`` (one order of the commutative ones)."""
    order = [PRICE, DIST, RATIO]
    fns = {"add": lambda a, b: a + b, "sub": lambda a, b: a - b,
           "mul": lambda a, b: rt.finite(a * b), "div": rt.div_checked}
    out = []
    for op, sign in ARITH.items():
        for a in order:
            for b in order:
                if op in ("add", "mul") and order.index(b) < order.index(a):
                    continue
                if (ret := arith_unit(op, a, b)) is None:
                    continue
                code = f"rt.div({{0}}, {{1}})" if op == "div" else f"({{0}} {sign} {{1}})"
                out.append(Primitive(f"{op}:{a},{b}", ret, (a, b), fns[op], f"{{0}} {sign} {{1}}", code,
                                     weight=0.3, prec=5 if op in ("add", "sub") else 6))
    return out


def _scratch_core() -> list[Primitive]:
    """Candles, logic, trade state, constants, the arithmetic and the low-level series maths."""
    core = {p.name: p for p in _core(set())}
    keep = ("open", "high", "low", "close", "always", "never", "and", "or", "not", "if_else:bool",
            "held", "stop", "target", "trail", "mid:price")
    out = [core[k] for k in keep]
    out += [
        Primitive("const:ratio", RATIO, (FACTOR,), rt.const, "{0}", "rt.const(o, {0})", kind="data", weight=0.6),
        Primitive("pct:osc", OSC, (RATIO,), lambda x: 100 * x, "100 * {0}", "(100 * {0})", weight=0.4, prec=6),
        Primitive("squash:osc", OSC, (RATIO,), rt.squash, "squash({0})", "rt.squash({0})", weight=0.3),
    ]
    out += _arithmetic()
    for u in (PRICE, DIST, RATIO):
        spread = DIST if u == PRICE else u  # what a difference or deviation of u is: price - price = dist
        out += [
            Primitive(f"diff:{u}", spread, (u, LAG), rt.diff, "diff({0}, {1})",
                      "rt.diff({0}, {1})", weight=0.5),
            Primitive(f"rolling_std:{u}", spread, (u, WINDOW), ops.rolling_std,
                      "rolling_std({0}, {1})", "ops.rolling_std({0}, {1})", weight=0.3),
        ]
        if u != PRICE:  # a sum, a sign or a size of price levels means nothing
            out += [
                Primitive(f"rolling_sum:{u}", u, (u, WINDOW), rt.rolling_sum, "rolling_sum({0}, {1})",
                          "rt.rolling_sum({0}, {1})", weight=0.3),
                Primitive(f"pos:{u}", u, (u,), rt.pos, "pos({0})", "rt.pos({0})", weight=0.4),
                Primitive(f"neg:{u}", u, (u,), rt.neg, "neg({0})", "rt.neg({0})", weight=0.4),
                Primitive(f"abs:{u}", u, (u,), ops.abs_, "abs({0})", "ops.abs_({0})", weight=0.3),
            ]
    return out


def module_primitive(name: str, body, slots: tuple, specs: tuple) -> Primitive:
    """A module ``name`` standing for ``body`` (a tree); parameter ``i`` fills the constants at ``slots[i]``."""
    args = ", ".join("{%d}" % i for i in range(len(specs)))
    return Primitive(name, body.unit, tuple(specs), None, f"{name}({args})" if specs else name,
                     f"{name}(o{', ' if specs else ''}{args})", kind="module", body=body, slots=slots)


# -- indicators ---------------------------------------------------------------------------------------
def _sample_ohlc(n: int, seed: int = 7) -> OHLC:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    open_ = np.r_[close[0], close[:-1]]
    spread = np.abs(rng.normal(0, 0.01, n)) * close
    time = 1.42e9 + 86_400.0 * np.arange(n)  # daily candles from 2015, for calendar indicators
    return OHLC(open_, np.maximum(open_, close) + spread, np.minimum(open_, close) - spread, close, time)


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
            b = cls.compute(OHLC(*(x[:part] for x in full.fields())), **defaults)
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
    text = call if cls.plot_outputs() == ("value",) else f"{call}.{output}"
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

    def __init__(self, primitives: list[Primitive], state: bool = True, scratch: bool = False):
        self.state, self.scratch = state, scratch
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

    @classmethod
    def from_scratch(cls, state: bool = True) -> "PrimitiveSet":
        """No ready-made indicators: candles, logic, constants and low-level maths with unit-checked arithmetic,
        enough to build RSI, Stochastic, Bollinger, ATR and MACD (#100)."""
        prims = _scratch_core()
        for unit in (PRICE, DIST, RATIO, OSC):
            prims += _generic(unit, SCRATCH_LEVELS.get(unit), scratch=True)
        comparisons = sum(p.family in COMPARISONS | LEVEL_COMPARISONS for p in prims)
        prims = [replace(p, weight=6.0 / comparisons) if p.family in COMPARISONS | LEVEL_COMPARISONS else p
                 for p in prims]
        return cls(prims, state, scratch=True)

    @property
    def modules(self) -> list[Primitive]:
        """The modules (evolved building blocks) in the order they were added."""
        return [p for p in self.primitives if p.kind == "module"]

    def with_modules(self, modules: list[Primitive]) -> "PrimitiveSet":
        """A new set with ``modules`` added; the modules of one unit share a weight of 2, like indicator outputs."""
        base = [p for p in self.primitives if p.kind != "module"] + self.modules + list(modules)
        mods = [p for p in base if p.kind == "module"]
        per_unit = {u: sum(p.ret == u for p in mods) for u in {p.ret for p in mods}}
        return PrimitiveSet([replace(p, weight=2.0 / per_unit[p.ret]) if p.kind == "module" else p for p in base],
                            self.state, self.scratch)
