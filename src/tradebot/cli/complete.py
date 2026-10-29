# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Shell completion callbacks for the CLI (install with ``tradebot --install-completion``).

The shell runs a fresh ``tradebot`` process for every TAB, so these callbacks import
lazily and keep slow lookups on disk (see :func:`tradebot.symbols.cli_symbol_search`).
Each returns ``(value, help)`` pairs; fish and zsh show the help next to the value.
"""

from __future__ import annotations

import typer

Completion = list[tuple[str, str]]


def _plugins_loaded() -> None:
    """Plugin algorithms, indicators and brokers complete like built-ins (the CLI callback doesn't run here)."""
    from tradebot.core.registry import load_plugins

    load_plugins()


def _starting(incomplete: str, items: Completion) -> Completion:
    low = incomplete.lower()
    return [(value, help_) for value, help_ in items if value.lower().startswith(low)]


def symbols(incomplete: str) -> Completion:
    from tradebot.config import watchlist
    from tradebot.data.cache import DiskCache
    from tradebot.symbols import cli_symbol_search

    if not incomplete.strip():  # nothing typed yet: the watchlist, no network
        return [(s, "watchlist") for s in dict.fromkeys(watchlist())]
    cache = DiskCache.from_config()
    search = cli_symbol_search(watchlist, cache.symbols_dir if cache else None)
    # one letter: watchlist only; from two letters on also crypto pairs and Yahoo
    matches = search.search(incomplete) if len(incomplete.strip()) >= 2 else search.local(incomplete)
    # shells match suggestions against the typed word, not the description, so a search by company
    # name can't surface in a terminal; tickers are matched case-insensitively ("sol" -> SOL/USDT)
    return _starting(incomplete, [(m.symbol, m.label[len(m.symbol):].lstrip(" —·")) for m in matches])


def algorithms(incomplete: str) -> Completion:
    _plugins_loaded()
    from tradebot.algorithms import ALGORITHMS

    return _starting(incomplete, [(cls.name, cls.description) for cls in ALGORITHMS])


def members(incomplete: str) -> Completion:
    """Comma-separated algorithm names for ``--members``: completes the last one."""
    head, _, last = incomplete.rpartition(",")
    prefix = f"{head}," if head else ""
    taken = set(head.split(","))
    return [(prefix + name, help_) for name, help_ in algorithms(last) if name not in taken]


def indicator_specs(incomplete: str) -> Completion:
    """``-i`` specs: indicator names, then ``name:param=`` for their parameters."""
    _plugins_loaded()
    from tradebot.indicators import INDICATORS

    name, colon, args = incomplete.partition(":")
    if not colon:
        return _starting(incomplete, [(cls.name, cls.description) for cls in INDICATORS])
    if name not in INDICATORS:
        return []
    head, _, last = args.rpartition(",")
    given = {kv.partition("=")[0] for kv in head.split(",") if kv}
    prefix = f"{name}:{head}," if head else f"{name}:"
    space = INDICATORS.get(name).params
    return _starting(incomplete, [(f"{prefix}{key}=", _param_help(p)) for key, p in space.items() if key not in given])


def algorithm_params(ctx: typer.Context, incomplete: str) -> Completion:
    """``-p`` / ``--fix``: ``key=`` for the parameters of the algorithm given earlier on the line."""
    _plugins_loaded()
    from tradebot.algorithms import ALGORITHMS, build_algorithm

    # while completing, positional arguments stay in ctx.args: "backtest AAPL rsi_threshold -p <TAB>"
    name = ctx.params.get("algo") or (ctx.args[1] if len(ctx.args) >= 2 else None)
    members_ = ctx.params.get("members")
    if not name or name not in ALGORITHMS:
        return []
    algo = build_algorithm(name, members=members_.split(",") if members_ else None)
    return _starting(incomplete, [(f"{key}=", _param_help(p)) for key, p in algo.param_space().items()])


def brokers(incomplete: str) -> Completion:
    _plugins_loaded()
    from tradebot.brokers import BROKERS

    return _starting(incomplete, [(cls.name, f"{cls.execution_mode.value} · {', '.join(a.value for a in cls.asset_classes)}")
                                  for cls in BROKERS])


def timeframes(incomplete: str) -> Completion:
    from tradebot.core.timeframes import TIMEFRAMES

    return _starting(incomplete, [(tf, "") for tf in TIMEFRAMES])


def choices(*items: tuple[str, str]):
    """Completion for a fixed set of values, e.g. ``--method``."""
    return lambda incomplete: _starting(incomplete, list(items))


def match_case_insensitively() -> None:
    """Let ``sol`` complete to ``SOL/USDT``: Typer keeps only suggestions starting with the typed text,
    case-sensitively. Replaces Typer's private wrapper with a case-insensitive one; if a Typer update
    removes it, completion still works, just case-sensitively (tests/test_cli_complete.py catches that).
    """
    import typer.core
    from typer._click.shell_completion import CompletionItem

    if not hasattr(typer.core, "_typer_param_setup_autocompletion_compat"):
        return

    def setup(self, *, autocompletion=None):
        if autocompletion is None:
            return

        def shell_complete(ctx, param, incomplete):
            items = [CompletionItem(c[0], help=c[1]) if isinstance(c, tuple) else CompletionItem(c)
                     for c in autocompletion(ctx, [], incomplete)]
            return [i for i in items if i.value.lower().startswith(incomplete.lower())]

        self._custom_shell_complete = shell_complete

    typer.core._typer_param_setup_autocompletion_compat = setup


def _param_help(p) -> str:
    return f"{p.description + ' · ' if p.description else ''}default {p.default:g}, {p.low:g}–{p.high:g}"
