# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Strategy combinator (#22): a strategy built from other strategies and indicator conditions, without code.

A combo has an **entry** rule and an **exit** rule, each a small expression::

    elliott_wave & close > sma(200)
    macd_cross | stop_atr(2)
    weighted(rsi_threshold:0.5, macd_cross:0.3, sma_cross:0.2) >= 0.6

Building blocks:

* a **strategy** name (``macd_cross``, a plugin, a saved combo) stands for its signal: in the entry rule it
  holds while that strategy is in the market, in the exit rule while it is out (its exit). So
  ``--exit macd_cross`` exits when macd_cross exits; ``!macd_cross`` in the exit rule exits while it is in.
  Parameters as arguments: ``rsi_threshold(lower=30)``, or positionally in the order of ``tradebot list``.
* an **indicator value**, compared: ``close > sma(200)``, ``rsi(14) < 30``, ``elliott.state == 3``,
  ``sma(50) > sma(200) * 1.02``. ``name(args).output`` picks an output (default: the first). Only
  outputs that never change when later candles arrive are allowed (no hindsight). Prices: ``open high low
  close typical``. Arithmetic: ``+ - * /``.
* an indicator **rule**: the indicator name alone, e.g. ``rsi`` or ``sma(period=100)``, is its pool rule (#21,
  :data:`tradebot.algorithms.pool.RULES`); an alert output alone (``sma.cross_up``) is true on its bars.
* logic: ``&`` / ``and``, ``|`` / ``or``, ``!`` / ``not``, brackets; ``vote(a, b, c) >= 0.6`` (the share of
  members that hold) and ``weighted(a:0.5, b:0.3) >= 0.6`` (their weighted share, weights normalised to sum 1).
* risk exits (exit rule only, joined with ``|``): ``stop_pct(5)``, ``stop_atr(2)`` (2 ATRs below the entry),
  ``take_profit(10)`` (%), ``take_profit_atr(3)``, ``trail_pct(5)``, ``trail_atr(3)`` (below the highest close
  since the entry), ``max_hold(20)`` (candles). Take-profits can scale out in steps: ``take_profit(5, 0.5) |
  take_profit(10)`` sells half at +5% and the rest at +10% (the second number: the share of the position).
* a limit entry (entry rule only, joined with ``&``): ``limit_pct(1)`` / ``limit_atr(0.5)``: when the rest of
  the entry rule holds on a candle, buy with a limit order 1% / 0.5 ATRs below its close, for the next candle.
* ``entry`` in the exit rule is the entry rule: ``!entry | stop_pct(5)``. Without an exit rule, a combo exits
  when the entry rule stops holding (``!entry``).

Semantics, the same as every strategy here: everything is evaluated on closed candles; the entry and exit
are **latched** (once in, only the exit rule or a risk exit gets out); an entry and an exit on the same
candle: the exit wins. A condition whose indicator is still warming up is neither true nor false: ``!``
doesn't turn it into true.

Fills (#150), ``fills=`` of a combo: ``"close"`` (the default) compares the stops, take-profits and limit
entries with the close and fills them at the close. ``"intrabar"`` fills them inside the candle against its
high and low, at the level, or at the open when the candle opens beyond it (a gap); a stop and a take-profit
both inside one candle: the stop counts (the worse case). See :meth:`Combo._with_risk`.

Every number in a call is a tunable parameter, namespaced like the vote's: ``rsi_threshold.lower``,
``sma.period`` (a second, different ``sma(...)`` is ``sma_2.period``), ``stop_atr.mult``; weights are
``w.<member>`` and thresholds ``threshold`` (the second group's: ``w2.*``, ``threshold2``). So the optimizer
tunes members, weights and thresholds together. Comparison constants (``< 30``) stay fixed.

Saved combos live in ``combos.toml`` next to ``config.toml`` and are registered as strategies named after
them, so they work everywhere (list, backtest, optimize, bot, chart, GUI, and inside other combos).
"""

from __future__ import annotations

import json
import logging
import operator
import re
import threading
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from tradebot.algorithms.base import ALGORITHMS, BaseAlgorithm
from tradebot.core import ops
from tradebot.core.registry import IntParam, Param
from tradebot.core.series import OHLC
from tradebot.indicators.base import INDICATORS
from tradebot.indicators.builtin import ATR

log = logging.getLogger(__name__)

PRICES = ("open", "high", "low", "close", "typical")

_SIZE = ("Share of the position (as bought) this take-profit sells; less than 1 scales out in steps, e.g. "
         "take_profit(5, 0.5) | take_profit(10).")
#: Risk exits: name -> ((parameter, spec), ...); the first parameter is the one a single argument sets.
RISK: dict[str, tuple[tuple[str, Param], ...]] = {
    "stop_pct": (("pct", Param(5.0, 0.5, 50.0, "Stop-loss: sell when the price is this many percent below the "
                                                "entry price.")),),
    "stop_atr": (("mult", Param(2.0, 0.5, 10.0, "Stop-loss: sell when the price is this many ATRs below the "
                                                 "entry price (ATR at the entry).")),
                 ("period", IntParam(14, 2, 100, "Candles the ATR (average true range) is measured over."))),
    "take_profit": (("pct", Param(10.0, 0.5, 200.0, "Take-profit: sell when the price is this many percent above "
                                                     "the entry price.")),
                    ("size", Param(1.0, 0.05, 1.0, _SIZE))),
    "take_profit_atr": (("mult", Param(3.0, 0.5, 20.0, "Take-profit: sell when the price is this many ATRs above "
                                                        "the entry price (ATR at the entry).")),
                        ("period", IntParam(14, 2, 100, "Candles the ATR (average true range) is measured over.")),
                        ("size", Param(1.0, 0.05, 1.0, _SIZE))),
    "trail_pct": (("pct", Param(5.0, 0.5, 50.0, "Trailing stop: sell when the price is this many percent below "
                                                 "the highest close since the entry.")),),
    "trail_atr": (("mult", Param(3.0, 0.5, 10.0, "Trailing stop: sell when the price is this many ATRs below "
                                                  "the highest close since the entry.")),
                  ("period", IntParam(14, 2, 100, "Candles the ATR (average true range) is measured over."))),
    "max_hold": (("bars", IntParam(20, 1, 1000, "Sell after holding this many candles.")),),
}
STOPS = ("stop_pct", "stop_atr", "trail_pct", "trail_atr")
TAKE_PROFITS = ("take_profit", "take_profit_atr")
#: Limit entries (entry rule only, joined with ``&``): buy below the close of the candle the rule holds on.
LIMITS: dict[str, tuple[tuple[str, Param], ...]] = {
    "limit_pct": (("pct", Param(1.0, 0.0, 20.0, "Limit entry: when the entry rule holds, place a buy order this "
                                                 "many percent below the close, for the next candle.")),),
    "limit_atr": (("mult", Param(0.5, 0.0, 5.0, "Limit entry: when the entry rule holds, place a buy order this "
                                                 "many ATRs below the close, for the next candle.")),
                  ("period", IntParam(14, 2, 100, "Candles the ATR (average true range) is measured over."))),
}
#: How orders are filled: at the close only, or inside the candle against its high and low.
FILLS = ("close", "intrabar")
GROUPS = ("vote", "weighted")
WORDS = {"and": "&", "or": "|", "not": "!"}
CMP = {">": operator.gt, "<": operator.lt, ">=": operator.ge, "<=": operator.le, "==": operator.eq,
       "!=": operator.ne}
ARITH = {"+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.truediv}


# -- parsing ------------------------------------------------------------------------------------------
_TOKEN = re.compile(r"\s*(?:(\d+\.?\d*(?:[eE][-+]?\d+)?|\.\d+)|([A-Za-z_]\w*)|(>=|<=|==|!=|&&|\|\||[-+*/<>!~&|(),:.=]))")


def tokenize(text: str) -> list[str]:
    out, i, text = [], 0, text.rstrip()
    while i < len(text):
        m = _TOKEN.match(text, i)
        if not m:
            raise ValueError(f"can't read {text[i:].strip()!r}")
        tok = m.group(1) or m.group(2) or m.group(3)
        tok = {"&&": "&", "||": "|", "~": "!"}.get(tok, tok)
        if m.group(2) and tok.lower() in WORDS:
            tok = WORDS[tok.lower()]
        out.append(tok)
        i = m.end()
    return out


@dataclass
class Ast:
    """A parsed expression node, before the names in it are resolved."""

    kind: str  #: num | name | call | attr | op | not | neg | cmp | arith
    value: object = None  #: number, name, operator
    args: list = field(default_factory=list)  #: children / call arguments
    kwargs: dict = field(default_factory=dict)
    weights: list = field(default_factory=list)  #: ``weighted(a:0.5)``: one number (or None) per argument


class _Parser:
    def __init__(self, text: str):
        self.text, self.toks, self.i = text, tokenize(text), 0

    def peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else None

    def take(self, expected=None):
        tok = self.peek()
        if tok is None or (expected is not None and tok != expected):
            raise ValueError(f"expected {expected or 'more'!r} in {self.text!r}"
                             + (f", got {tok!r}" if tok is not None else ""))
        self.i += 1
        return tok

    def parse(self) -> Ast:
        if not self.toks:
            raise ValueError("empty rule")
        node = self.or_()
        if self.peek() is not None:
            raise ValueError(f"unexpected {self.peek()!r} in {self.text!r}")
        return node

    def _chain(self, op, sub):
        items = [sub()]
        while self.peek() == op:
            self.take()
            items.append(sub())
        return items[0] if len(items) == 1 else Ast("op", op, items)

    def or_(self):
        return self._chain("|", self.and_)

    def and_(self):
        return self._chain("&", self.not_)

    def not_(self):
        if self.peek() == "!":
            self.take()
            return Ast("not", args=[self.not_()])
        return self.cmp()

    def cmp(self):
        left = self.sum_()
        if self.peek() in CMP:
            op = self.take()
            return Ast("cmp", op, [left, self.sum_()])
        return left

    def sum_(self):
        node = self.prod()
        while self.peek() in ("+", "-"):
            node = Ast("arith", self.take(), [node, self.prod()])
        return node

    def prod(self):
        node = self.unary()
        while self.peek() in ("*", "/"):
            node = Ast("arith", self.take(), [node, self.unary()])
        return node

    def unary(self):
        if self.peek() == "-":
            self.take()
            inner = self.unary()
            return Ast("num", -inner.value) if inner.kind == "num" else Ast("neg", args=[inner])
        return self.atom()

    def atom(self):
        tok = self.take()
        if tok == "(":
            node = self.or_()
            self.take(")")
            return node
        if re.fullmatch(r"\d+\.?\d*(?:[eE][-+]?\d+)?|\.\d+", tok):
            return Ast("num", float(tok))
        if not re.fullmatch(r"[A-Za-z_]\w*", tok):
            raise ValueError(f"unexpected {tok!r} in {self.text!r}")
        node = Ast("name", tok)
        if self.peek() == "(":
            node = self.call(tok)
        if self.peek() == ".":
            self.take()
            out = self.take()
            if not re.fullmatch(r"[A-Za-z_]\w*", out):
                raise ValueError(f"expected an output name after {tok!r}.")
            node = Ast("attr", out, [node])
        return node

    def call(self, name):
        self.take("(")
        node = Ast("call", name)
        while self.peek() != ")":
            if self.peek() is not None and self.i + 1 < len(self.toks) and self.toks[self.i + 1] == "=" \
                    and re.fullmatch(r"[A-Za-z_]\w*", self.peek()):
                key = self.take()
                self.take("=")
                node.kwargs[key] = self.or_()
            else:
                if node.kwargs:
                    raise ValueError(f"positional argument after keyword in {name}(...)")
                node.args.append(self.or_())
                weight = None
                if self.peek() == ":":
                    self.take()
                    weight = self.unary()
                    if weight.kind != "num":
                        raise ValueError(f"a weight must be a number in {name}(...)")
                    weight = weight.value
                node.weights.append(weight)
            if self.peek() != ")":
                self.take(",")
        self.take(")")
        return node


def parse(text: str) -> Ast:
    """The expression tree of a rule; raises ``ValueError`` with a readable message."""
    return _Parser(text).parse()


# -- compiled nodes -----------------------------------------------------------------------------------
class _Eval:
    """One evaluation: the candles, the parameters, and the members' results computed so far."""

    def __init__(self, combo: "Combo", o: OHLC, p: dict):
        self.combo, self.o, self.p, self.cache = combo, o, p, {}


def _known(x):
    return None if isinstance(x, (int, float)) else ~ops.isnan(x)


def _and(a, b):
    return b if a is None else a if b is None else a & b


def _bool(x, like):
    """A truth array from a Python bool (for constants), shaped like ``like``."""
    return ops.full(like.shape, bool(x), like) > 0.5


class Node:
    def value(self, ev: _Eval):
        raise ValueError(f"{self} is a condition, not a number: compare numbers, e.g. close > sma(200)")

    def truth(self, ev: _Eval):
        """``(true, false)`` masks; a candle where the condition can't be known yet is in neither."""
        raise ValueError(f"{self} is a number, not a condition: compare it, e.g. {self} > 0")

    def walk(self):
        yield self


@dataclass(eq=False)
class Num(Node):
    v: float

    def value(self, ev):
        return self.v

    def __str__(self):
        return f"{self.v:g}"


@dataclass(eq=False)
class Price(Node):
    name: str

    def value(self, ev):
        return ev.o.typical if self.name == "typical" else getattr(ev.o, self.name)

    def __str__(self):
        return self.name


@dataclass(eq=False)
class Member(Node):
    """A strategy, indicator or rule with its own parameters (``<key>.<param>``)."""

    key: str
    space: dict  #: the member's parameter space (unprefixed)
    source: str  #: the registered strategy or indicator name

    def params(self, ev):
        return {k: ev.p[f"{self.key}.{k}"] for k in self.space}


@dataclass(eq=False)
class Strategy(Member):
    algo: BaseAlgorithm = None
    exit_side: bool = False  #: in the exit rule: holds while the strategy is out

    def value(self, ev):
        if self.key not in ev.cache:
            mp = {k: ev.p.get(f"{self.key}.{k}", v) for k, v in self.algo.values.items()}
            ev.cache[self.key] = ops.nan_to_num(self.algo.signals(ev.o, mp))
        return ev.cache[self.key]

    def truth(self, ev):
        held = self.value(ev) > 0.5
        return (~held, held) if self.exit_side else (held, ~held)

    def __str__(self):
        return self.key


@dataclass(eq=False)
class Indicator(Member):
    cls: type = None
    output: str = "value"
    alert: bool = False

    def value(self, ev):
        if self.key not in ev.cache:
            ev.cache[self.key] = self.cls.compute(ev.o, **self.params(ev))
        return ev.cache[self.key][self.output]

    def truth(self, ev):
        if not self.alert:
            return super().truth(ev)
        x = self.value(ev)
        known = ~ops.isnan(x)
        return known & (x > 0.5), known & ~(x > 0.5)

    def __str__(self):
        return f"{self.key}.{self.output}"


@dataclass(eq=False)
class Rule(Member):
    rule: object = None

    def truth(self, ev):
        held = self.rule.condition(ev.o, self.params(ev))
        held = held if not isinstance(held, bool) else _bool(held, ev.o.close)
        return held, ~held

    def __str__(self):
        return self.key


@dataclass(eq=False)
class Compare(Node):
    op: str
    a: Node
    b: Node

    def truth(self, ev):
        x, y = self.a.value(ev), self.b.value(ev)
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            return _bool(CMP[self.op](x, y), ev.o.close), _bool(not CMP[self.op](x, y), ev.o.close)
        held = CMP[self.op](x, y)
        known = _and(_known(x), _known(y))
        return (held, ~held) if known is None else (known & held, known & ~held)

    def walk(self):
        yield self
        yield from self.a.walk()
        yield from self.b.walk()

    def __str__(self):
        return f"{self.a} {self.op} {self.b}"


@dataclass(eq=False)
class Arith(Node):
    op: str
    a: Node
    b: Node

    def value(self, ev):
        return ARITH[self.op](self.a.value(ev), self.b.value(ev))

    def walk(self):
        yield self
        yield from self.a.walk()
        yield from self.b.walk()

    def __str__(self):
        return f"({self.a} {self.op} {self.b})"


@dataclass(eq=False)
class Not(Node):
    a: Node

    def truth(self, ev):
        t, f = self.a.truth(ev)
        return f, t

    def walk(self):
        yield self
        yield from self.a.walk()

    def __str__(self):
        return f"!{self.a}"


@dataclass(eq=False)
class Logic(Node):
    op: str  #: "&" or "|"
    items: list

    def truth(self, ev):
        t, f = self.items[0].truth(ev)
        for item in self.items[1:]:
            ti, fi = item.truth(ev)
            t, f = (t & ti, f | fi) if self.op == "&" else (t | ti, f & fi)  # Kleene: unknown stays unknown
        return t, f

    def walk(self):
        yield self
        for item in self.items:
            yield from item.walk()

    def __str__(self):
        return "(" + f" {self.op} ".join(map(str, self.items)) + ")"


@dataclass(eq=False)
class Group(Node):
    """``vote(...)`` / ``weighted(...)``: the (weighted) share of members that hold, against a threshold."""

    kind: str
    items: list
    weights: list  #: parameter keys (weighted) or None (vote)
    threshold: str  #: parameter key
    strict: bool = False  #: ``> threshold`` instead of ``>=``

    def value(self, ev):
        total, score = 0.0, 0.0
        for item, w in zip(self.items, self.weights or [None] * len(self.items)):
            weight = 1.0 if w is None else ev.p[w]
            held = item.truth(ev)[0]
            score = score + weight * (held if not ops.is_torch(held) else held.double())
            total = total + weight
        return score / (total + 1e-12) if self.weights else score / total

    def truth(self, ev):
        share, thr = self.value(ev), ev.p[self.threshold]
        held = share > thr if self.strict else share >= thr - 1e-9
        return held, ~held

    def walk(self):
        yield self
        for item in self.items:
            yield from item.walk()

    def __str__(self):
        return f"{self.kind}(...)"


@dataclass(eq=False)
class EntryRef(Node):
    """``entry`` inside the exit rule."""

    def truth(self, ev):
        return ev.combo._entry_truth(ev)

    def __str__(self):
        return "entry"


@dataclass
class Risk:
    key: str
    kind: str
    params: tuple = ()  #: its parameter names (a take-profit without a size has none)


# -- the strategy -------------------------------------------------------------------------------------
_building = threading.local()  #: names of the saved combos being built, against cycles


class Combo(BaseAlgorithm):
    """Entry and exit rules built from strategies, indicator conditions and risk exits (see the module docs)."""

    name = "combo"
    description = "Combine strategies and indicator conditions: entry and exit rules with and/or/not/vote/weighted"
    poolable = False  # the generic combo has no rule of its own; saved combos can be pool members
    #: the rules of a saved combo (subclasses created by :func:`register_combo`)
    entry_rule: str = "rsi_threshold & close > sma(200)"
    exit_rule: str = "rsi_threshold | stop_pct(10)"
    saved_params: dict = {}
    saved: bool = False  #: True for combos from ``combos.toml``
    fills_rule: str = "close"  #: a saved combo's fill model (:data:`FILLS`)

    def __init__(self, entry: str | None = None, exit: str | None = None, fills: str | None = None, **params):
        self.entry = (entry or self.entry_rule).strip()
        exit = self.exit_rule if exit is None and entry is None else exit
        self.exit = (exit or "").strip()
        self.fills = (fills or self.fills_rule or "close").strip().lower()
        if self.fills not in FILLS:
            raise ValueError(f"fills must be one of {', '.join(FILLS)}, not {fills!r}")
        self.limit: Risk | None = None
        self._space: dict[str, Param] = {}
        self._keys: dict[tuple, str] = {}
        self._groups = 0
        self._loop = False
        self.members: list[Member] = []
        stack = getattr(_building, "names", None)
        if stack is None:
            stack = _building.names = []
        if type(self).name in stack and type(self) is not Combo:
            raise ValueError(f"combo {type(self).name!r} contains itself: {' -> '.join([*stack, type(self).name])}")
        stack.append(type(self).name)
        try:
            self._entry = self._compile_entry(parse(self.entry))
            self.risks: list[Risk] = []
            self._exit = Not(EntryRef())
            if self.exit:
                self._exit = self._compile_exit(parse(self.exit))
        finally:
            stack.pop()
        defaults = {k: v for k, v in self.saved_params.items() if k in self._space}
        self._space = {k: replace(s, default=s.clip(defaults[k])) if k in defaults else s for k, s in self._space.items()}
        super().__init__(**params)
        with np.errstate(all="ignore"):  # a dry run, so a misused block fails now with a clear message
            self.signals(_sample(), self.values)

    # -- building
    def _compile_entry(self, ast: Ast):
        terms = ast.args if ast.kind == "op" and ast.value == "&" else [ast]
        rest = []
        for term in terms:
            if term.kind in ("call", "name") and term.value in LIMITS:
                if self.limit is not None:
                    raise ValueError("only one limit entry (limit_pct, limit_atr) per entry rule")
                key, _ = self._member("risk", dict(LIMITS[term.value]), term, "limit entry")
                self.limit = Risk(key, term.value, tuple(k for k, _ in LIMITS[term.value]))
            else:
                rest.append(term)
        if not rest:
            raise ValueError("a limit entry needs a condition to buy on, e.g. rsi(14) < 30 & limit_pct(1)")
        return self._compile(rest[0] if len(rest) == 1 else Ast("op", "&", rest), exit_rule=False)

    def _compile_exit(self, ast: Ast):
        terms = ast.args if ast.kind == "op" and ast.value == "|" else [ast]
        rest = []
        for term in terms:
            if term.kind in ("call", "name") and term.value in RISK:
                self.risks.append(self._risk(term))
            else:
                rest.append(term)
        if not rest:
            return None  # only risk exits
        node = rest[0] if len(rest) == 1 else Ast("op", "|", rest)
        return self._compile(node, exit_rule=True)

    def _risk(self, ast: Ast) -> Risk:
        kind = ast.value
        space = dict(RISK[kind])
        if kind in TAKE_PROFITS:  # several take-profits scale out; the size is a parameter only when given
            given = ast.kind == "call" and (len(ast.args) > list(space).index("size") or "size" in ast.kwargs)
            if not given:
                space.pop("size")
        elif any(r.kind == kind for r in self.risks):
            raise ValueError(f"{kind} appears twice in the exit rule")
        key, space = self._member("risk", space, ast, "risk exit")
        if any(r.key == key for r in self.risks):
            raise ValueError(f"{key} appears twice in the exit rule")
        return Risk(key, kind, tuple(space))

    def _key(self, identity: tuple, name: str) -> str:
        if identity in self._keys:
            return self._keys[identity]
        taken = set(self._keys.values())
        key, n = name, 2
        while key in taken:
            key, n = f"{name}_{n}", n + 1
        self._keys[identity] = key
        return key

    def _member(self, kind: str, space: dict, ast: Ast, what: str) -> tuple[str, dict]:
        """``(key, parameter space)`` of a member called as in ``ast``; the call's numbers become the defaults.
        Calls with the same name and values share one key (and so one set of parameters)."""
        args = list(ast.args) if ast.kind == "call" else []
        kwargs = dict(ast.kwargs) if ast.kind == "call" else {}
        names = list(space)
        if len(args) > len(names):
            raise ValueError(f"{ast.value} takes at most {len(names)} argument(s): {', '.join(names) or 'none'}")
        given = dict(zip(names, args))
        for k, v in kwargs.items():
            if k not in space:
                raise ValueError(f"{ast.value} has no parameter {k!r}; it has: {', '.join(names) or 'none'}")
            given[k] = v
        out = {}
        for k, spec in space.items():
            if k in given:
                v = given[k]
                if v.kind != "num":
                    raise ValueError(f"{ast.value}({k}=...) must be a number")
                v = int(round(v.value)) if spec.integer else v.value
                spec = replace(spec, default=v, low=min(spec.low, v), high=max(spec.high, v))
            if not spec.description:
                spec = replace(spec, description=f"The {k.replace('_', ' ')} of {ast.value} in this {what}.")
            out[k] = spec
        key = self._key((kind, ast.value, tuple(s.default for s in out.values())), ast.value)
        for k, spec in out.items():
            self._space.setdefault(f"{key}.{k}", spec)
        return key, out

    def _compile(self, ast: Ast, exit_rule: bool, number: bool = False) -> Node:
        """``number``: the node is used as a number (compared, computed with), so an indicator name is its
        value, not its pool rule."""
        c = lambda a: self._compile(a, exit_rule)  # noqa: E731
        n = lambda a: self._compile(a, exit_rule, number=True)  # noqa: E731
        if ast.kind == "num":
            return Num(ast.value)
        if ast.kind == "op":
            return Logic(ast.value, [c(a) for a in ast.args])
        if ast.kind == "not":
            return Not(c(ast.args[0]))
        if ast.kind == "neg":
            return Arith("-", Num(0.0), n(ast.args[0]))
        if ast.kind == "arith":
            return Arith(ast.value, n(ast.args[0]), n(ast.args[1]))
        if ast.kind == "cmp":
            left, right = ast.args
            if left.kind == "call" and left.value in GROUPS and right.kind == "num" and ast.value in (">=", ">"):
                return self._group(left, exit_rule, right.value, ast.value == ">")
            return Compare(ast.value, n(left), n(right))
        if ast.kind == "attr":
            inner = ast.args[0]
            if inner.value not in INDICATORS:
                raise ValueError(f"{inner.value}.{ast.value}: only indicators have outputs; {inner.value!r} is no "
                                 "indicator")
            return self._indicator(inner, ast.value)
        name = ast.value
        if name in GROUPS:
            if ast.kind != "call":
                raise ValueError(f"{name} needs members: {name}(a, b, ...)")
            return self._group(ast, exit_rule, None, False)
        if name in RISK:
            raise ValueError(f"{name} is a risk exit: use it in the exit rule, joined with |")
        if name in LIMITS:
            raise ValueError(f"{name} is a limit entry: use it in the entry rule, joined with &")
        if name == "entry":
            if not exit_rule or ast.kind == "call":
                raise ValueError("'entry' can only be used in the exit rule, for the entry rule")
            return EntryRef()
        if name in PRICES:
            if ast.kind == "call":
                raise ValueError(f"{name} takes no arguments")
            return Price(name)
        if name in ALGORITHMS:
            return self._strategy(ast, exit_rule)
        if name in INDICATORS:
            from tradebot.algorithms.pool import RULES

            if name in RULES and not number:
                return self._rule(ast, RULES[name])
            return self._indicator(ast, None)
        raise ValueError(f"unknown name {name!r}: not a strategy, indicator, price ({', '.join(PRICES)}), "
                         f"risk exit ({', '.join(RISK)}) or group ({', '.join(GROUPS)})")

    def _strategy(self, ast: Ast, exit_rule: bool) -> Strategy:
        from tradebot.algorithms.builtin import VotingEnsemble

        cls = ALGORITHMS.get(ast.value)
        if issubclass(cls, VotingEnsemble):
            raise ValueError(f"{ast.value} needs members; use vote(a, b, ...) here instead")
        if cls is Combo:
            raise ValueError("'combo' itself can't be a member; save a combo under a name and use that name")
        algo = cls()
        if not algo.vectorized:
            self._loop = True
        key, space = self._member("strategy", algo.param_space(), ast, "strategy")
        algo = algo.with_params(**{k: s.default for k, s in space.items()})
        node = Strategy(key, space, ast.value, algo, exit_rule)
        self.members.append(node)
        return node

    def _rule(self, ast: Ast, rule) -> Rule:
        if not rule.vectorized:
            self._loop = True
        key, space = self._member("rule", dict(rule.params), ast, "rule")
        node = Rule(key, space, ast.value, rule)
        self.members.append(node)
        return node

    def _indicator(self, ast: Ast, output: str | None) -> Indicator:
        from tradebot.algorithms.pool import RULES
        from tradebot.evolve.primitives import causal_outputs

        cls = INDICATORS.get(ast.value)
        output = output or (cls.plot_outputs() or cls.outputs)[0]
        if output not in cls.outputs:
            raise ValueError(f"{cls.name} has no output {output!r}; it has: {', '.join(cls.outputs)}")
        if output not in causal_outputs(cls):
            raise ValueError(f"{cls.name}.{output} uses later candles (hindsight), so it can't be used in a rule")
        if ast.value in RULES and not RULES[ast.value].vectorized:
            self._loop = True
        key, space = self._member("indicator", dict(cls.params), ast, "condition")
        node = Indicator(key, space, ast.value, cls, output, output in cls.alerts)
        self.members.append(node)
        return node

    def _group(self, ast: Ast, exit_rule: bool, threshold: float | None, strict: bool) -> Group:
        if ast.kwargs or not ast.args:
            raise ValueError(f"{ast.value}(...) takes members, e.g. {ast.value}(macd_cross, rsi_threshold)")
        self._groups += 1
        n = "" if self._groups == 1 else str(self._groups)
        items = [self._compile(a, exit_rule) for a in ast.args]
        weights = None
        if ast.value == "weighted":
            weights = []
            for i, (item, w) in enumerate(zip(items, ast.weights), 1):
                label = item.key if isinstance(item, Member) and not isinstance(item, Indicator) else str(i)
                key = f"w{n}.{label}"
                if key in self._space:
                    key = f"w{n}.{i}"
                w = 1.0 if w is None else w
                self._space[key] = Param(w, 0.0, max(1.0, w), f"Weight of {item} in the weighted vote; the "
                                                                "weights are scaled to add up to 1.")
                weights.append(key)
        elif any(w is not None for w in ast.weights):
            raise ValueError("vote(...) takes no weights; use weighted(a:0.5, b:0.3)")
        thr = f"threshold{n}"
        default = 0.5 if threshold is None else threshold
        what = "weighted share" if weights else "share of members"
        self._space[thr] = Param(default, min(0.0, default), max(1.0, default),
                                 f"The {what} that must hold for {ast.value}(...) to hold: 0.6 means 60%.")
        return Group(ast.value, items, weights, thr, strict)

    # -- the BaseAlgorithm interface
    def param_space(self):
        return dict(self._space)

    @property
    def vectorized(self) -> bool:
        return not self._loop

    def _entry_truth(self, ev):
        if "_entry" not in ev.cache:
            ev.cache["_entry"] = self._entry.truth(ev)
        return ev.cache["_entry"]

    def signals(self, o, p):
        return self.execute(o, p)[0]

    def execute(self, o, p):
        ev = _Eval(self, o, p)
        enter = self._entry_truth(ev)[0]
        leave = self._exit.truth(ev)[0] if self._exit is not None else ops.full(o.close.shape, 0.0, o.close) > 0.5
        if not self.risks and self.limit is None:
            return ops.latch(enter, leave), None  # every fill at the close, whatever the fill model
        return self._with_risk(o, p, enter, leave)

    def _with_risk(self, o, p, enter, leave):
        """The latch with risk exits and limit entries: they depend on the entry price, so this walks the candles
        (batched over parameter sets; NumPy either way, converted back to the input's backend).

        ``fills="close"``: every level is compared with the close and filled at the close. ``fills="intrabar"``
        (#150): stops, take-profits and limit entries fill inside the candle against its high and low, at the
        level, or at the open when the candle opens beyond it (a gap). The levels for a candle come from the
        candles before it only (no look-ahead). When a stop and a take-profit are both inside one candle, the
        stop is taken to have come first (the worse outcome), unless the candle opens beyond the take-profit
        (then that is sold at the open first); on the candle a limit entry fills on, a
        take-profit counts only if the close is beyond it (the high may have come before the fill), a stop if the
        low is below it (after a fill at the limit or the open, the price must have gone on down to the low). The
        exit rule and max_hold still exit at the close. Returns ``(positions, Fills | None)``."""
        from tradebot.backtest.engine import Fills

        np_ = lambda x: x.detach().cpu().numpy() if ops.is_torch(x) else np.asarray(x)  # noqa: E731
        intrabar = self.fills == "intrabar"
        close = np_(o.close).astype(float)
        ohlc_np = None

        def atr_of(period):
            nonlocal ohlc_np
            if ohlc_np is None:
                ohlc_np = OHLC(*(None if a is None else np_(a) for a in o.fields()))
            return np_(ATR.compute(ohlc_np, period=period)["value"])

        terms = []  # (kind, {param: value}, atr or None)
        for r in self.risks + ([self.limit] if self.limit is not None else []):
            vals = {k: p[f"{r.key}.{k}"] for k in r.params}
            vals = {k: v if isinstance(v, (int, float)) else np_(v) for k, v in vals.items()}
            terms.append((r.kind, vals, atr_of(vals["period"]) if "period" in vals else None))
        enter, leave = np_(enter).astype(bool), np_(leave).astype(bool)
        shapes = [enter.shape, leave.shape, close.shape] + [np.shape(v) for _, vals, _ in terms for v in vals.values()]
        shape = np.broadcast_shapes(*shapes)
        b = lambda x: np.broadcast_to(np.asarray(x, dtype=float), shape)  # noqa: E731
        enter, leave, close = np.broadcast_to(enter, shape), np.broadcast_to(leave, shape), b(close)
        if intrabar:
            price = lambda x: close if x is None else b(np_(x))  # noqa: E731  (no high/low: the close)
            open_, high, low = price(o.open), price(o.high), price(o.low)
        terms = [(kind, {k: b(v) for k, v in vals.items()}, None if atr is None else b(atr))
                 for kind, vals, atr in terms]
        stops = [t for t in terms if t[0] in STOPS]
        tps = [t for t in terms if t[0] in TAKE_PROFITS]
        max_holds = [t for t in terms if t[0] == "max_hold"]
        limit = terms[-1] if self.limit is not None else None
        lead = shape[:-1]
        pos = np.zeros(lead)
        entry_px, peak, held = np.zeros(lead), np.zeros(lead), np.zeros(lead)
        entry_atr = {id(t): np.full(lead, np.nan) for t in terms}
        tp_done = [np.zeros(lead, dtype=bool) for _ in tps]
        order = np.zeros(lead, dtype=bool)  # a limit buy for this candle, placed at the previous close
        order_px = np.full(lead, np.nan)
        out = np.zeros(shape)
        adjust, extra = (np.zeros(shape), np.zeros(shape)) if intrabar else (None, None)
        prev_c = close[..., 0]

        def stop_level(t, px, top):
            """The highest stop below ``px`` (the entry price) / ``top`` (the highest close), from candle ``t``."""
            lvl = np.full(lead, -np.inf)
            for term in stops:
                kind, vals, atr = term
                v = vals[next(iter(vals))][..., t]
                lvl = np.fmax(lvl, px * (1 - v / 100) if kind == "stop_pct" else
                                 px - v * entry_atr[id(term)] if kind == "stop_atr" else
                                 top * (1 - v / 100) if kind == "trail_pct" else
                                 top - v * atr[..., t])
            return np.nan_to_num(lvl, nan=-np.inf)

        def tp_level(term, t):
            kind, vals, _ = term
            v = vals[next(iter(vals))][..., t]
            return entry_px * (1 + v / 100) if kind == "take_profit" else entry_px + v * entry_atr[id(term)]

        def tp_size(term, t):
            return term[1]["size"][..., t] if "size" in term[1] else 1.0

        for t in range(shape[-1]):
            c = close[..., t]
            was = pos > 0  # in at the previous close
            cur = pos.copy()
            if intrabar:  # the legs of this candle: hold ``cur`` from ``cur_px`` on, compounded into ``g``
                g, cur_px, turn = np.ones(lead), prev_c.copy(), np.zeros(lead)

                def leg(mask, px, new):
                    nonlocal g, cur_px, cur, turn
                    px = np.where(mask, px, cur_px)
                    g = np.where(mask, g * (1 + cur * (px / cur_px - 1)), g)
                    turn = np.where(mask, turn + np.abs(new - cur), turn)
                    cur = np.where(mask, new, cur)
                    cur_px = px

            if was.any():
                held = held + was
                if intrabar:  # the levels are known at the previous close
                    o_, h, lo = open_[..., t], high[..., t], low[..., t]
                    lvl = stop_level(t - 1, entry_px, peak)
                    levels = [tp_level(term, t - 1) for term in tps]
                    ranks = np.argsort(np.stack(levels), axis=0) if tps else []  # lowest take-profit first

                    def take_profits(when, at_open: bool):
                        for i in ranks:
                            for k, term in enumerate(tps):
                                pick = when & (i == k) & ~tp_done[k] & (cur > 0) & (
                                    o_ >= levels[k] if at_open else h >= levels[k])
                                leg(pick, np.maximum(o_, levels[k]), np.maximum(cur - tp_size(term, t - 1), 0.0))
                                tp_done[k] |= pick

                    take_profits(was & (o_ > lvl), at_open=True)  # opens above a take-profit: sold at the open
                    hit = was & (cur > 0) & (lo <= lvl)  # a stop inside the candle comes before any take-profit
                    leg(hit, np.minimum(o_, lvl), 0.0)  # opens below the stop: sold at the open
                    take_profits(was & ~hit, at_open=False)
                peak = np.where(was, np.maximum(peak, c), peak)
                full = was & leave[..., t]
                for term in max_holds:
                    full |= was & (held >= term[1]["bars"][..., t])
                if not intrabar:  # every level against the close (the levels as of this close)
                    full |= was & (c <= stop_level(t, entry_px, peak))
                    for k, term in enumerate(tps):
                        pick = was & ~full & ~tp_done[k] & (c >= tp_level(term, t))
                        cur = np.where(pick, np.maximum(cur - tp_size(term, t), 0.0), cur)
                        tp_done[k] |= pick
                        full |= pick & (cur <= 0)
                    cur = np.where(full, 0.0, cur)
                else:
                    leg(full & (cur > 0), c, 0.0)
            # entries: only from flat at the previous close (no re-entry on a candle with an exit)
            flat = ~was
            if limit is None:
                started = flat & enter[..., t] & ~leave[..., t]
                fill_px = c
                if intrabar:
                    leg(started, c, 1.0)
            elif intrabar:
                started = flat & order & (low[..., t] <= order_px)
                fill_px = np.minimum(open_[..., t], order_px)  # opens below the limit: filled at the open
                leg(started, fill_px, 1.0)
            else:
                started = flat & order & ~leave[..., t] & (c <= order_px)
                fill_px = c
            if started.any():
                entry_px = np.where(started, fill_px, entry_px)
                peak = np.where(started, np.maximum(fill_px, c) if limit is not None else c, peak)
                held = np.where(started, 0.0, held)
                at = t if limit is None else max(t - 1, 0)  # a limit order's ATR: at the candle it was placed on
                for term in terms:
                    if term[2] is not None:
                        entry_atr[id(term)] = np.where(started, term[2][..., at], entry_atr[id(term)])
                tp_done = [d & ~started for d in tp_done]
                if limit is None:
                    cur = np.where(started, 1.0, cur)
                elif intrabar:  # the rest of the candle the limit filled on (see the docstring)
                    lvl = stop_level(at, entry_px, entry_px)  # as of the order (a trailing ATR too)
                    hit = started & (low[..., t] <= lvl)
                    leg(hit, lvl, 0.0)
                    for k, term in enumerate(tps):
                        tp = tp_level(term, at)
                        pick = started & ~hit & ~tp_done[k] & (cur > 0) & (c >= tp)
                        leg(pick, tp, np.maximum(cur - tp_size(term, at), 0.0))
                        tp_done[k] |= pick
                    leg(started & (cur > 0) & leave[..., t], c, 0.0)
                else:
                    cur = np.where(started, 1.0, cur)
            if intrabar:
                g = g * (1 + cur * (c / cur_px - 1))
                adjust[..., t] = (g - 1) - pos * (c / prev_c - 1)
                extra[..., t] = turn - np.abs(cur - pos)
            if limit is not None:  # the limit buy for the next candle
                kind, vals, atr = limit
                v = vals[next(iter(vals))][..., t]
                order = (cur <= 0) & enter[..., t] & ~leave[..., t]
                order_px = c * (1 - v / 100) if kind == "limit_pct" else c - v * atr[..., t]
                order &= ~np.isnan(order_px)
            pos = cur
            prev_c = c
            out[..., t] = pos
        wrap = (lambda x: ops.asarray(x, o.close)) if ops.is_torch(o.close) else (lambda x: x)
        fills = Fills(wrap(adjust), wrap(extra)) if intrabar else None
        return wrap(out), fills

    def chart_indicators(self):
        out, seen = [], set()
        for m in self.members:
            if isinstance(m, Strategy):
                inds = m.algo.with_params(**{k: self.values[f"{m.key}.{k}"] for k in m.space}).chart_indicators()
            else:
                cls = INDICATORS.get(m.source)
                inds = [cls(**{k: self.values[f"{m.key}.{k}"] for k in m.space if k in cls.params})]
            for ind in inds:
                if ind.label not in seen:
                    seen.add(ind.label)
                    out.append(ind)
        return out

    def spec(self) -> dict:
        """The rules as saved: ``{"entry": ..., "exit": ..., "fills": ...}``."""
        return {"entry": self.entry, "exit": self.exit, "fills": self.fills}

    def __repr__(self) -> str:
        return (f"{self.name}(entry: {self.entry}" + (f" · exit: {self.exit}" if self.exit else "")
                + (f" · fills: {self.fills})" if self.fills != "close" else ")"))


def _sample(n: int = 300) -> OHLC:
    rng = np.random.default_rng(22)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    open_ = np.r_[close[0], close[:-1]]
    spread = np.abs(rng.normal(0, 0.01, n)) * close
    return OHLC(open_, np.maximum(open_, close) + spread, np.minimum(open_, close) - spread, close,
                1.42e9 + 86_400.0 * np.arange(n))


def check(entry: str, exit: str | None = None, fills: str | None = None) -> Combo:
    """A combo from the rules, or ``ValueError`` explaining what's wrong."""
    try:
        return Combo(entry=entry, exit=exit or "", fills=fills)
    except KeyError as e:
        raise ValueError(str(e).strip("'\"")) from None


# -- combining blocks (the GUI's builder) --------------------------------------------------------------
LOGIC = ("and", "or", "vote", "weighted")


def terms(text: str) -> tuple[list[str], list[str]]:
    """``(terms, risk exits)`` at the top level of a rule: the members of a group, or what ``&``/``|`` join."""
    toks = tokenize(text) if text.strip() else []
    if not toks:
        return [], []
    if toks[0] in GROUPS and len(toks) > 1 and toks[1] == "(":
        depth, end = 0, None
        for i, tok in enumerate(toks):
            depth += tok == "("
            depth -= tok == ")"
            if depth == 0 and i > 0:
                end = i
                break
        if end is not None and all(t in (">=", ">") or re.fullmatch(r"[\d.eE+-]+", t) for t in toks[end + 1:]):
            parts = _split(toks[2:end], {","})
            parts = [p[:-2] if len(p) > 2 and p[-2] == ":" else p for p in parts]
            return [_join(p) for p in parts], []
    out, risk = [], []
    for part in _split(toks, {"&", "|"}):
        (risk if part and part[0] in RISK else out).append(_join(part))
    return out, risk


def _split(toks, seps):
    parts, cur, depth = [], [], 0
    for tok in toks:
        if depth == 0 and tok in seps:
            parts.append(cur)
            cur = []
            continue
        depth += tok == "("
        depth -= tok == ")"
        cur.append(tok)
    parts.append(cur)
    return [p for p in parts if p]


def _join(toks) -> str:
    out, prev = "", None
    for tok in toks:
        tight = prev is None or prev in ("(", ".", "!", ":") or tok in (")", ",", ".", ":") \
            or (tok == "(" and re.fullmatch(r"\w+", prev) is not None)
        out += tok if tight else " " + tok
        prev = tok
    return out


def combine(items: list[str], logic: str, risk: list[str] = (), threshold: float = 0.5) -> str:
    """A rule joining ``items`` with ``logic`` (:data:`LOGIC`), with the risk exits OR-ed on."""
    items = [i.strip() for i in items if i.strip()]
    wrap = lambda i: f"({i})" if len(items) > 1 and len(terms(i)[0]) > 1 else i  # noqa: E731
    if not items:
        body = ""
    elif logic == "and":
        body = " & ".join(map(wrap, items))
    elif logic == "or":
        body = " | ".join(map(wrap, items))
    elif logic in GROUPS:
        members = items if logic == "vote" else [f"{i}:1" for i in items]
        body = f"{logic}({', '.join(members)}) >= {threshold:g}"
    else:
        raise ValueError(f"logic must be one of {', '.join(LOGIC)}")
    return " | ".join(([body] if body else []) + list(risk))


# -- saved combos -------------------------------------------------------------------------------------
NAME = re.compile(r"[a-z_][a-z0-9_]*")
RESERVED = {*PRICES, *RISK, *LIMITS, *GROUPS, *WORDS, "entry", "combo"}


def combos_path() -> Path:
    """``combos.toml`` next to ``config.toml`` (``~/.config/tradebot/combos.toml``)."""
    from tradebot.config import config_path

    return config_path().parent / "combos.toml"


def load_saved(path: Path | None = None) -> dict[str, dict]:
    import tomllib

    path = path or combos_path()
    if not path.is_file():
        return {}
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    return {k: v for k, v in data.items() if isinstance(v, dict) and isinstance(v.get("entry"), str)}


def _write(saved: dict[str, dict], path: Path) -> None:
    lines = ["# Saved strategy combos (tradebot combo save / the Backtest lab's combo builder).", ""]
    for name, c in saved.items():
        lines.append(f"[{name}]")
        for k in ("entry", "exit", "description", "fills"):
            if c.get(k) and not (k == "fills" and c[k] == "close"):
                lines.append(f"{k} = {json.dumps(c[k], ensure_ascii=False)}")
        if c.get("params"):
            lines.append(f"[{name}.params]")
            lines += [f"{json.dumps(k)} = {float(v)!r}" for k, v in c["params"].items()]
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def register_combo(name: str, entry: str, exit: str = "", description: str = "",
                   params: dict | None = None, fills: str = "close") -> type[Combo]:
    """Register a saved combo as the strategy ``name`` (a :class:`Combo` subclass); replaces an earlier one."""
    attrs = {"name": name, "entry_rule": entry, "exit_rule": exit or "", "saved_params": dict(params or {}),
             "poolable": True, "saved": True, "fills_rule": fills or "close",
             "description": description or f"Combo: enter on {entry}" + (f", exit on {exit}" if exit else ""),
             "__module__": __name__}
    with ALGORITHMS.reloading():
        return type(f"Combo_{name}", (Combo,), attrs)


def saved_combos() -> list[str]:
    """Names of the registered saved combos."""
    return [cls.name for cls in ALGORITHMS if issubclass(cls, Combo) and getattr(cls, "saved", False)]


def load_combos(path: Path | None = None) -> list[str]:
    """Register every combo in ``combos.toml``; a broken one is logged and skipped."""
    try:
        saved = load_saved(path)
    except Exception:
        log.exception("can't read %s", path or combos_path())
        return []
    names = []
    for name, c in saved.items():
        if not NAME.fullmatch(name) or name in RESERVED or (name in ALGORITHMS and name not in saved_combos()):
            log.warning("combo %r skipped: the name is taken or not allowed", name)
            continue
        register_combo(name, c["entry"], c.get("exit", ""), c.get("description", ""), c.get("params"),
                       c.get("fills", "close"))
        names.append(name)
    for name in names:  # every name is known now, so combos may use each other
        try:
            ALGORITHMS.get(name)()
        except (ValueError, KeyError) as e:
            log.warning("combo %r skipped: %s", name, e)
            ALGORITHMS.unregister(name)
    return [n for n in names if n in ALGORITHMS]


def save_combo(name: str, entry: str, exit: str = "", description: str = "", params: dict | None = None,
               path: Path | None = None, fills: str = "close") -> type[Combo]:
    """Check, save to ``combos.toml`` and register a combo; ``ValueError`` if the rules or the name are bad."""
    if not NAME.fullmatch(name):
        raise ValueError(f"{name!r} can't be a combo name: use lower-case letters, digits and _")
    if name in RESERVED or name in INDICATORS or (name in ALGORITHMS and name not in saved_combos()):
        raise ValueError(f"{name!r} is already taken by a strategy, indicator or keyword")
    probe = check(entry, exit, fills)  # the rules on their own
    params = {k: v for k, v in (params or {}).items() if k in probe.param_space()}
    old = ALGORITHMS.get(name) if name in ALGORITHMS else None
    cls = register_combo(name, entry, exit, description, params, probe.fills)
    try:
        cls()  # catches a combo that would contain itself
    except (ValueError, KeyError) as e:
        if old is not None:
            with ALGORITHMS.reloading():
                ALGORITHMS.register(old)
        else:
            ALGORITHMS.unregister(name)
        raise ValueError(str(e)) from None
    path = path or combos_path()
    saved = load_saved(path)
    saved[name] = {"entry": entry, "exit": exit or "", "description": description, "params": params,
                   "fills": probe.fills}
    _write(saved, path)
    return cls


def delete_combo(name: str, path: Path | None = None) -> bool:
    """Remove a saved combo from ``combos.toml`` and the registry; False if there is none by that name."""
    path = path or combos_path()
    saved = load_saved(path)
    found = saved.pop(name, None) is not None
    if found:
        _write(saved, path)
    if name in saved_combos():
        ALGORITHMS.unregister(name)
        found = True
    return found
