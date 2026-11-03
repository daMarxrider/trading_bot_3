# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Evolved building blocks (#100): subtrees that keep showing up in good programs become named modules.

A from-scratch RSI is a tree of ~15 nodes; a strategy made of two such pieces outgrows any sensible depth
and size limit, and crossover would tear the pieces apart. So every few generations the best programs
are searched for shared subtrees. A subtree is compared by its **template**: the tree with its constants
replaced by parameters, where equal constants of the same kind become one parameter. So ``wilder(pos(diff(
close, 1)), 14) / (wilder(pos(diff(close, 1)), 14) + wilder(neg(diff(close, 1)), 14))`` is
``m1(n1, n2)`` with ``n1 = 1`` (the lag) and ``n2 = 14`` (the window), whatever the window is in each program.

A frequent template becomes a module: one node whose constants still evolve (mutation nudges ``n2`` for all
three smoothings at once). The population is rewritten to use it, new random subtrees can pick it, and
the exported plugin defines it as a function.
"""

from __future__ import annotations

from collections import Counter

from tradebot.evolve.primitives import BOOL, Primitive, PrimitiveSet, module_primitive
from tradebot.evolve.program import Node, Program, const


class Template:
    """A subtree with its constants as parameters (equal constants of the same range share one)."""

    def __init__(self, node: Node):
        self.node = node
        slots: dict[tuple, list] = {}
        for path, n in node.nodes():
            if n.is_const:
                slots.setdefault((id(n.spec), n.value), []).append(path)
        self.slots = tuple(tuple(paths) for paths in slots.values())
        self.specs = tuple(node.at(paths[0]).spec for paths in self.slots)
        self.values = tuple(node.at(paths[0]).value for paths in self.slots)
        names = {path: f"${i}" for i, paths in enumerate(self.slots) for path in paths}
        self.key = _key(node, (), names)

    @property
    def operations(self) -> int:
        """Nodes that are not constants."""
        return sum(not n.is_const for _, n in self.node.nodes())


def _key(node: Node, path: tuple, names: dict) -> str:
    if node.is_const:
        return names[path]
    if not node.args:
        return node.prim.name
    return f"{node.prim.name}({','.join(_key(a, (*path, i), names) for i, a in enumerate(node.args))})"


def _eligible(node: Node, min_operations: int) -> bool:
    """A numeric subtree (an evolved indicator; conditions stay open to evolution) without trade state."""
    return (not node.is_const and bool(node.args) and not node.uses_state and node.unit != BOOL
            and sum(not n.is_const for _, n in node.nodes()) >= min_operations)


def discover(programs: list[Program], known: set[str], *, min_share: float = 0.1, min_count: int = 3,
             min_operations: int = 3, limit: int = 2) -> list[Template]:
    """Templates found in at least ``min_share`` (and ``min_count``) of ``programs``, with at least
    ``min_operations`` operations and not in ``known``; the ``limit`` that save the most nodes first."""
    counts: Counter = Counter()
    templates: dict[str, Template] = {}
    for program in programs:
        seen = set()
        for tree in (program.entry, program.exit):
            for _, n in tree.nodes():
                if not _eligible(n, min_operations):
                    continue
                t = Template(n)
                if t.key not in seen:
                    seen.add(t.key)
                    counts[t.key] += 1
                    templates.setdefault(t.key, t)
    need = max(min_count, min_share * len(programs))
    found = [templates[k] for k, c in counts.items() if c >= need and k not in known]
    found.sort(key=lambda t: (-counts[t.key] * (t.node.size - 1 - len(t.slots)), t.key))
    out: list[Template] = []
    for t in found:  # not a part of a template already picked in this round (the bigger one covers it)
        if not any(t.key in o.key for o in out):
            out.append(t)
        if len(out) == limit:
            break
    return out


def make_module(name: str, template: Template) -> Primitive:
    return module_primitive(name, template.node, template.slots, template.specs)


def rewrite(node: Node, modules: dict[str, Primitive]) -> Node:
    """``node`` with every subtree matching a module's template (``modules``: template key -> module) replaced
    by that module, outermost first."""
    if node.is_const or not node.args:
        return node
    if modules and not node.uses_state:
        t = Template(node)
        if (m := modules.get(t.key)) is not None:
            return Node(m, tuple(const(spec, v) for spec, v in zip(t.specs, t.values)))
    args = tuple(rewrite(a, modules) for a in node.args)
    return node if all(x is y for x, y in zip(args, node.args)) else Node(node.prim, args, node.value, node.spec)


def rewrite_program(program: Program, modules: dict[str, Primitive]) -> Program:
    entry, exit_ = rewrite(program.entry, modules), rewrite(program.exit, modules)
    return program if entry is program.entry and exit_ is program.exit else Program(entry, exit_)


class Library:
    """The modules of one run, with their templates for :func:`rewrite`."""

    def __init__(self, prefix: str = "m"):
        self.prefix = prefix
        self.by_template: dict[str, Primitive] = {}

    @property
    def modules(self) -> list[Primitive]:
        return list(self.by_template.values())

    def grow(self, programs: list[Program], ps: PrimitiveSet, **kwargs) -> tuple[PrimitiveSet, list[Primitive]]:
        """Promote the frequent templates of ``programs``; the new primitive set and the new modules."""
        new = []
        for t in discover(programs, set(self.by_template), **kwargs):
            m = make_module(f"{self.prefix}{len(self.by_template) + 1}", t)
            self.by_template[t.key] = m
            new.append(m)
        if not new:
            return ps, []
        ps = ps.with_modules(new)
        # the set re-weights its modules (new objects); the library hands out the set's own
        self.by_template = {k: ps[m.name] for k, m in self.by_template.items()}
        return ps, [ps[m.name] for m in new]

    def rewrite(self, program: Program) -> Program:
        return rewrite_program(program, self.by_template)
