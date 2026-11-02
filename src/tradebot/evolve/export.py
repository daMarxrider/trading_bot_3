# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""An evolved program as Python (#83): a ``BaseAlgorithm`` plugin file.

The generated class is ordinary strategy code against :mod:`tradebot.core.ops`, so it works wherever a
hand-written strategy does: ``backtest``, the chart, the Backtest lab and the bot. Every constant of the
program becomes a parameter with the range it was evolved in (``n1``, ``n2``, ... for whole numbers,
``c1``, ... for the rest), so ``tradebot optimize <name>`` can fine-tune them, batched on the GPU unless
the exit rule looks at the open trade (that needs a loop over the trades, ``vectorized = False``).
"""

from __future__ import annotations

import keyword
import re
import types

from tradebot.core.registry import Param
from tradebot.evolve.program import Node, Program, format_value
from tradebot.evolve import runtime as rt

NAME = re.compile(r"[a-z_][a-z0-9_]*\Z")  #: see :func:`valid_name`


def parameter_names(program: Program) -> dict[tuple[str, tuple[int, ...]], str]:
    """``(rule, path)`` of every constant -> its parameter name (``n1``... integers, ``c1``... floats)."""
    names, counts = {}, {"n": 0, "c": 0}
    for rule in ("entry", "exit"):
        for path, node in getattr(program, rule).nodes():
            if node.is_const:
                prefix = "n" if node.spec.integer else "c"
                counts[prefix] += 1
                names[(rule, path)] = f"{prefix}{counts[prefix]}"
    return names


def parameters(program: Program) -> dict[str, Param]:
    """The program's constants as parameters: default = the evolved value, range = the evolved range,
    description = where the constant sits (e.g. ``in rsi(n1) < c1``)."""
    names = parameter_names(program)
    out = {}
    for (rule, path), name in names.items():
        tree = getattr(program, rule)
        node, parent = tree.at(path), tree.at(path[:-1])
        shown = Node(parent.prim, tuple(Node(None, value=names.get((rule, (*path[:-1], i)), "?"), spec=a.spec)
                                        if a.is_const else a for i, a in enumerate(parent.args)))
        spec = node.spec
        what = parent.prim.indicator[2][path[-1]] if parent.prim.kind == "indicator" else spec.description
        out[name] = Param(node.value, spec.low, spec.high, f"{what} in {shown}", integer=spec.integer)
    return out


def _code(node: Node, rule: str, path: tuple, names: dict, in_trade: bool, hoisted: dict) -> str:
    if node.is_const:
        return f'p["{names[(rule, path)]}"]'
    if in_trade and not node.uses_state:  # computed once on all candles, then cut to the trade
        code = _code(node, rule, path, names, False, hoisted)
        return f"t.cut({hoisted.setdefault(code, f'x{len(hoisted) + 1}')})"
    args = [_code(a, rule, (*path, i), names, in_trade, hoisted) for i, a in enumerate(node.args)]
    return node.prim.code.format(*args)


def signals_source(program: Program, indent: str = "") -> str:
    """``def signals(self, o, p): ...`` for the program."""
    names = parameter_names(program)
    lines = ["def signals(self, o, p):", f"    entry = {_code(program.entry, 'entry', (), names, False, {})}"]
    if program.exit.uses_state:
        hoisted: dict[str, str] = {}
        body = _code(program.exit, "exit", (), names, True, hoisted)
        lines += [f"    {var} = {code}" for code, var in hoisted.items()]
        lines += ["", "    def exit_(t):  # t: the open trade (tradebot.evolve.runtime.Trade)", f"        return {body}",
                  "", "    return rt.in_trade(entry, exit_, o.close)"]
    else:
        lines += [f"    exit_ = {_code(program.exit, 'exit', (), names, False, {})}", "    return ops.latch(entry, exit_)"]
    return "\n".join(indent + line if line else "" for line in lines)


def compile_signals(program: Program):
    """``signals(o, p)`` of the generated code, with the program's constants as default ``p``."""
    from tradebot.core import ops

    ns = {"ops": ops, "rt": rt}
    exec(signals_source(program), ns)  # noqa: S102 - code generated from our own primitives
    defaults = {k: v.default for k, v in parameters(program).items()}
    fn = ns["signals"]
    return lambda o, p=None: fn(None, o, defaults if p is None else {**defaults, **p})


def chart_indicators_source(program: Program) -> list[str]:
    """``INDICATORS.get(...)(...)`` for every indicator the rules use (and every moving average of the close)."""
    names = parameter_names(program)
    out = []
    for rule in ("entry", "exit"):
        for path, node in getattr(program, rule).nodes():
            if node.is_const:
                continue
            p = node.prim
            if p.kind == "indicator":
                ind, _, keys = p.indicator
                kw = [f'{k}=v["{names[(rule, (*path, i))]}"]' for i, k in enumerate(keys)]
            elif p.family in ("sma", "ema") and node.args[0].prim is not None and node.args[0].prim.name == "close":
                ind, kw = p.family, [f'period=v["{names[(rule, (*path, 1))]}"]']
            else:
                continue
            code = f'INDICATORS.get("{ind}")({", ".join(kw)})'
            if code not in out:
                out.append(code)
    return out


def class_name(name: str) -> str:
    return "".join(part.capitalize() for part in name.split("_") if part) or "Evolved"


def valid_name(name: str) -> bool:
    """Usable as an algorithm and file name: lower-case letters, digits and _, not a Python keyword."""
    return bool(NAME.match(name)) and not keyword.iskeyword(name)


def plugin_source(program: Program, name: str, notes: list[str] = ()) -> str:
    """A plugin file defining the algorithm ``name``; ``notes`` go into its docstring (how it was evolved)."""
    if not valid_name(name):
        raise ValueError(f"{name!r} is not a valid algorithm name: lower-case letters, digits and _")
    params = parameters(program)
    doc = [f"Evolved strategy {name!r}, generated by ``tradebot evolve`` (#83).", "",
           "Rules::", "", *(f"    {line}" for line in str(program).splitlines())]
    if notes:
        doc += ["", *notes]
    doc += ["", "Copy or edit it like any plugin; ``tradebot optimize <symbol> " + name + "`` fine-tunes the constants."]
    rules = program.rules()
    lines = [
        "# Generated by tradebot evolve (#83). Safe to edit.",
        '"""' + "\n".join(doc).replace('"""', "'''") + '\n"""',
        "",
        "from tradebot.algorithms.base import BaseAlgorithm",
        "from tradebot.core import ops",
        "from tradebot.core.registry import Param",
        "from tradebot.evolve import runtime as rt",
        "from tradebot.indicators.base import INDICATORS",
        "",
        "",
        f"class {class_name(name)}(BaseAlgorithm):",
        f"    name = {name!r}",
        f"    description = {('Evolved: buy when ' + rules['entry'] + '; sell when ' + rules['exit'])!r}",
    ]
    if program.uses_state:
        lines.append("    vectorized = False  # the exit rule looks at the open trade: one evaluation per trade")
    lines.append("    params = {")
    for key, spec in params.items():
        default = format_value(spec.default) if spec.integer else repr(spec.default)
        lines.append(f"        {key!r}: Param({default}, {spec.low!r}, {spec.high!r}, {spec.description!r}"
                     + (", integer=True" if spec.integer else "") + "),")
    lines += ["    }", "", signals_source(program, "    ")]
    if charts := chart_indicators_source(program):
        lines += ["", "    def chart_indicators(self):", "        v = self.values",
                  "        shown = [", *(f"            {c}," for c in charts), "        ]",
                  "        return list({i.label: i for i in shown}.values())  # each one once"]
    return "\n".join(lines) + "\n"


def load_plugin(source: str, module_name: str = "tradebot_evolved", register: bool = True):
    """Execute a generated plugin source; returns its algorithm class, registered like any plugin unless
    ``register`` is False (then the registry is left as it was, e.g. for a preview on the chart)."""
    from tradebot.algorithms.base import ALGORITHMS, BaseAlgorithm

    before = dict(ALGORITHMS._items)
    module = types.ModuleType(module_name)
    exec(compile(source, f"<{module_name}>", "exec"), module.__dict__)  # noqa: S102
    if not register:
        ALGORITHMS._items.clear()
        ALGORITHMS._items.update(before)
    return next(v for v in module.__dict__.values()
                if isinstance(v, type) and issubclass(v, BaseAlgorithm) and v is not BaseAlgorithm)
