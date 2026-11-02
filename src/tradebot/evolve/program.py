# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Evolved programs (#83): an entry rule and an exit rule, each an expression tree.

A :class:`Node` applies a primitive (:mod:`tradebot.evolve.primitives`) to sub-rules and constants.
Trees are immutable; :func:`simplify` rewrites them into a canonical form (constant folding, ``x > x``
is ``never``, sorted arguments of ``and``/``or``/``max``, ...), so equivalent programs share one cache
entry and print the same. :class:`Evaluator` turns a program into positions on one symbol's candles,
caching every sub-rule it computes: a population shares most of its building blocks (``rsi(14)``,
``sma(close, 50)``), so most of a generation is looked up instead of computed.
"""

from __future__ import annotations

import heapq
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

from tradebot.core import ops
from tradebot.core.registry import Param
from tradebot.core.series import OHLC
from tradebot.evolve import runtime as rt
from tradebot.evolve.primitives import BOOL, COMMUTATIVE, COMPARISONS, Primitive, PrimitiveSet


@dataclass(frozen=True, eq=False)
class Node:
    """A primitive applied to ``args`` (sub-rules and constants), or a constant (``prim`` None)."""

    prim: Primitive | None
    args: tuple["Node", ...] = ()
    value: float | int | None = None
    spec: Param | None = None  #: a constant's range

    @property
    def is_const(self) -> bool:
        return self.prim is None

    @property
    def unit(self) -> str | None:
        return None if self.prim is None else self.prim.ret

    @cached_property
    def key(self) -> str:
        """Exact, canonical identity (cache key)."""
        if self.prim is None:
            return repr(self.value)
        return f"{self.prim.name}({','.join(a.key for a in self.args)})" if self.args else self.prim.name

    @cached_property
    def size(self) -> int:
        return 1 + sum(a.size for a in self.args)

    @cached_property
    def depth(self) -> int:
        """Levels of sub-rules (constants don't count)."""
        return 0 if self.prim is None else 1 + max((a.depth for a in self.args), default=0)

    @cached_property
    def uses_state(self) -> bool:
        """Looks at the open trade (then it can only be evaluated trade by trade)."""
        return self.prim is not None and (self.prim.kind == "state" or any(a.uses_state for a in self.args))

    @property
    def prec(self) -> int:
        return 9 if self.prim is None else self.prim.prec

    def __str__(self) -> str:
        if self.prim is None:
            return format_value(self.value)
        parts = []
        for i, a in enumerate(self.args):
            s = str(a)
            # infix forms: parentheses where the child binds weaker, and for the right side of a minus
            if self.prec < 9 and (a.prec < self.prec or (a.prec == self.prec and self.prim.family == "sub" and i == 1)):
                s = f"({s})"
            parts.append(s)
        return self.prim.text.format(*parts).replace("+ -", "- ")

    def nodes(self, path: tuple[int, ...] = ()):
        """``(path, node)`` of this node and every descendant, depth first."""
        yield path, self
        for i, a in enumerate(self.args):
            yield from a.nodes((*path, i))

    def at(self, path: tuple[int, ...]) -> "Node":
        node = self
        for i in path:
            node = node.args[i]
        return node

    def replace(self, path: tuple[int, ...], new: "Node") -> "Node":
        if not path:
            return new
        args = list(self.args)
        args[path[0]] = args[path[0]].replace(path[1:], new)
        return Node(self.prim, tuple(args), self.value, self.spec)


def format_value(v) -> str:
    if isinstance(v, str):  # a parameter name in place of the value
        return v
    return str(v) if isinstance(v, (int, np.integer)) else f"{float(v):.4g}"


def const(spec: Param, value) -> Node:
    return Node(None, value=quantize(spec, value), spec=spec)


def quantize(spec: Param, value):
    """Integers as they are; floats on a grid of 1/200 of the range, so near-equal constants share a cache entry."""
    if spec.integer:
        return int(spec.clip(value))
    step = (spec.high - spec.low) / 200 or 1.0
    return float(round(spec.clip(spec.low + round((value - spec.low) / step) * step), 10))


@dataclass(frozen=True, eq=False)
class Program:
    """A long-only strategy: in on ``entry``, out on ``exit`` (an exit on the entry candle wins)."""

    entry: Node
    exit: Node

    @cached_property
    def key(self) -> str:
        return f"{self.entry.key} | {self.exit.key}"

    @property
    def size(self) -> int:
        return self.entry.size + self.exit.size

    @property
    def depth(self) -> int:
        return max(self.entry.depth, self.exit.depth)

    @property
    def uses_state(self) -> bool:
        return self.exit.uses_state

    def rules(self) -> dict[str, str]:
        return {"entry": str(self.entry), "exit": str(self.exit)}

    def __str__(self) -> str:
        return f"entry = {self.entry}\nexit  = {self.exit}"


# -- random trees -------------------------------------------------------------------------------------
def random_const(spec: Param, rng: np.random.Generator) -> Node:
    if spec.integer:
        return const(spec, int(rng.integers(int(spec.low), int(spec.high) + 1)))
    return const(spec, rng.uniform(spec.low, spec.high))


def random_tree(ps: PrimitiveSet, unit: str, depth: int, rng: np.random.Generator, *, full: bool = False,
                state: bool = False) -> Node:
    """A random, type-valid tree returning ``unit`` with at most ``depth`` levels.

    ``full`` keeps choosing primitives with sub-rules until the depth is used up ("full" initialisation),
    otherwise any primitive that fits is chosen by weight ("grow"). ``state`` allows trade-state
    primitives (exit rules only).
    """
    options = ps.candidates(unit, depth, state)
    if not options:
        raise ValueError(f"no primitive returns {unit!r} within depth {depth}")
    if full and depth > 1:
        deeper = [p for p in options if not p.terminal]
        options = deeper or options
    weights = np.array([p.weight for p in options])
    prim = options[int(rng.choice(len(options), p=weights / weights.sum()))]
    args = tuple(random_tree(ps, a, depth - 1, rng, full=full, state=state) if isinstance(a, str)
                 else random_const(a, rng) for a in prim.args)
    return simplify(Node(prim, args), ps)


def random_program(ps: PrimitiveSet, depth: int, rng: np.random.Generator, full: bool = False) -> Program:
    def rule(state):
        for _ in range(50):  # simplifying rarely rebuilds a part deeper than it was
            tree = random_tree(ps, BOOL, depth, rng, full=full, state=state)
            if tree.depth <= depth:
                return tree
        return random_tree(ps, BOOL, 2, rng, state=state)

    return Program(rule(False), rule(ps.state))


def check(node: Node, unit: str, ps: PrimitiveSet | None = None, state: bool = True) -> None:
    """Raise ``TypeError`` unless ``node`` is a valid tree returning ``unit`` (``state``: trade state allowed)."""
    if node.is_const:
        raise TypeError(f"constant {node.value} where a {unit} rule is expected")
    p = node.prim
    if p.ret != unit:
        raise TypeError(f"{p.name} returns {p.ret}, expected {unit}")
    if p.kind == "state" and not state:
        raise TypeError(f"{p.name} looks at the open trade, which only exit rules can")
    if ps is not None and p.name not in ps:
        raise TypeError(f"unknown primitive {p.name}")
    if len(node.args) != len(p.args):
        raise TypeError(f"{p.name} takes {len(p.args)} arguments, got {len(node.args)}")
    for a, want in zip(node.args, p.args):
        if isinstance(want, str):
            check(a, want, ps, state)
        elif not a.is_const or a.spec != want or not want.low <= a.value <= want.high:
            raise TypeError(f"{p.name}: bad constant {a.key}")


# -- simplification -----------------------------------------------------------------------------------
def simplify(node: Node, ps: PrimitiveSet) -> Node:
    """Canonical, smaller form of ``node`` (its children are assumed simplified already)."""
    if node.is_const or not node.args:
        return node
    fam = node.prim.family
    a = node.args
    always, never = Node(ps["always"]), Node(ps["never"])

    def is_(x, name):
        return x.prim is not None and x.prim.name == name

    if fam in ("and", "or"):
        absorbing, neutral = (never, always) if fam == "and" else (always, never)
        terms: dict[str, Node] = {}
        for x in (*_chain(a[0], fam), *_chain(a[1], fam)):  # flattened, without duplicates
            terms.setdefault(x.key, x)
        if absorbing.key in terms or any(is_(x, "not") and x.args[0].key in terms for x in terms.values()):
            return absorbing  # x and never / x and not x
        terms.pop(neutral.key, None)
        if not terms:
            return neutral
        # rebuilt with the least depth: always join the two shallowest parts (ties by key, so it stays canonical)
        heap = [(x.depth, k, x) for k, x in terms.items()]
        heapq.heapify(heap)
        while len(heap) > 1:
            (_, k1, x), (_, k2, y) = heapq.heappop(heap), heapq.heappop(heap)
            joined = Node(node.prim, (x, y) if k1 <= k2 else (y, x))
            heapq.heappush(heap, (joined.depth, joined.key, joined))
        return heap[0][2]
    elif fam == "not":
        x = a[0]
        if is_(x, "not"):
            return x.args[0]
        if is_(x, "always"):
            return never
        if is_(x, "never"):
            return always
    elif fam in COMPARISONS and a[0].key == a[1].key:
        return never
    elif fam in ("max", "min", "mid") and a[0].key == a[1].key:
        return a[0]
    elif fam == "if_else":
        c, x, y = a
        if is_(c, "always") or x.key == y.key:
            return x
        if is_(c, "never"):
            return y
    if fam in COMMUTATIVE - {"and", "or"} and a[0].key > a[1].key:
        return Node(node.prim, (a[1], a[0]))
    return node


def _chain(node: Node, family: str) -> list[Node]:
    """The operands of a chain ``a and b and c`` (the node itself if it isn't one)."""
    if node.prim is not None and node.prim.family == family:
        return [*_chain(node.args[0], family), *_chain(node.args[1], family)]
    return [node]


def simplify_tree(node: Node, ps: PrimitiveSet) -> Node:
    """:func:`simplify` bottom-up over a whole tree."""
    if node.is_const or not node.args:
        return node
    return simplify(Node(node.prim, tuple(simplify_tree(x, ps) for x in node.args), node.value, node.spec), ps)


def constants(program: Program) -> list[tuple[str, tuple[int, ...], Node]]:
    """``(rule, path, constant)`` of every constant, entry rule first."""
    return [(rule, path, n) for rule in ("entry", "exit") for path, n in getattr(program, rule).nodes() if n.is_const]


# -- evaluation ---------------------------------------------------------------------------------------
@dataclass
class Evaluator:
    """Computes programs on one symbol's candles, caching sub-rules (least recently used dropped first)."""

    ohlc: OHLC
    cache_size: int = 3000
    _cache: OrderedDict = field(default_factory=OrderedDict, repr=False)
    _indicators: dict = field(default_factory=dict, repr=False)

    def positions(self, program: Program) -> np.ndarray:
        """0/1 per candle."""
        with np.errstate(all="ignore"):
            entry = self.value(program.entry)
            if not program.exit.uses_state:
                return np.asarray(ops.latch(entry, self.value(program.exit)), dtype=float)
            return rt.in_trade(entry, lambda t: self.value(program.exit, t), self.ohlc.close)

    def value(self, node: Node, trade: rt.Trade | None = None):
        if node.is_const:
            return node.value
        if trade is not None and node.uses_state:
            return self._apply(node, [self.value(a, trade) if a.uses_state or a.is_const else trade.cut(self.value(a))
                                      for a in node.args], trade)
        hit = self._cache.get(node.key)
        if hit is not None:
            self._cache.move_to_end(node.key)
            return hit
        out = self._apply(node, [self.value(a) for a in node.args], None)
        self._cache[node.key] = out
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return out

    def _apply(self, node: Node, args: list, trade):
        p = node.prim
        if p.kind == "op":
            return p.fn(*args)
        if p.kind == "data":
            return p.fn(self.ohlc, *args)
        if p.kind == "state":
            return p.fn(trade, *args)
        name, output, keys = p.indicator
        params = dict(zip(keys, args))
        k = (name, tuple(params.items()))
        if k not in self._indicators:
            if len(self._indicators) > self.cache_size:
                self._indicators.clear()
            self._indicators[k] = rt.indicator(name, self.ohlc, **params)
        return self._indicators[k][output]
