# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Automatic member selection: the ``pool`` and its genes in GA/PSO (#21)."""

import shlex

import numpy as np
import pytest

from tradebot.algorithms import ALGORITHMS, build_algorithm
from tradebot.algorithms.base import BaseAlgorithm
from tradebot.algorithms.pool import RULES, IndicatorFilter, PoolEnsemble, default_pool
from tradebot.core import ops
from tradebot.indicators import INDICATORS
from tradebot.optimize import OPTIMIZERS, GeneticOptimizer, compare
from tradebot.optimize.base import OptimizationResult

from .conftest import make_ohlc


class Oracle(BaseAlgorithm):
    """Peeks one bar ahead: long exactly before up-moves. The only pool member with an edge."""

    def signals(self, o, p):
        return ops.where(ops.shift(o.close, -1) > o.close, 1.0, 0.0)


Oracle.name = "oracle"  # set after the class body, so it isn't registered for the other tests


def test_default_pool_holds_poolable_strategies_and_ruled_indicators():
    members, skipped = default_pool()
    assert {"rsi_threshold", "macd_cross", "elliott_wave", "sma", "rsi", "elliott"} <= set(members)
    assert not {"vote", "pool", "range"} & set(members)
    assert "atr" in skipped and set(members) | set(skipped) >= set(INDICATORS.names())
    assert set(RULES) <= set(INDICATORS.names())


@pytest.mark.parametrize("name", sorted(RULES))
def test_indicator_rules_produce_positions(name):
    f = IndicatorFilter(name)
    pos = f.run(make_ohlc(1000, seed=5))  # long enough for an Elliott wave 3
    assert set(np.unique(pos)) <= {0.0, 1.0} and pos.sum() > 0
    assert f.chart_indicators()[0].name == name


def test_pool_member_errors():
    with pytest.raises(ValueError, match="no default rule"):
        PoolEnsemble(["macd_cross", "atr"])
    with pytest.raises(ValueError, match="at least one strategy"):
        PoolEnsemble(["sma", "rsi"])
    with pytest.raises(ValueError, match="can't be a pool member"):
        PoolEnsemble(["vote"])
    with pytest.raises(ValueError):
        PoolEnsemble(["macd_cross"], min_members=3, max_members=2)


def test_filters_gate_entries_only():
    df = make_ohlc(600, seed=2)
    alone = build_algorithm("pool", members=["sma_cross"]).run(df)
    filtered = build_algorithm("pool", members=["sma_cross", "rsi"]).run(df)
    entries = lambda pos: (pos.diff() > 0).sum()  # noqa: E731
    assert entries(filtered) <= entries(alone)
    assert ((filtered == 1) <= (alone == 1)).all()  # a filter never keeps a position the vote has left


def test_constraints_are_respected():
    pool = PoolEnsemble(["sma_cross", "macd_cross", "rsi_threshold", "sma", "rsi", "stoch"],
                        min_members=2, max_members=3)
    opt = GeneticOptimizer(pool, make_ohlc(300), backend="numpy", seed=0)
    genomes = np.random.default_rng(1).random((300, len(opt.keys)))
    genomes[:20, [opt.keys.index(f"use.{n}") for n in ("sma_cross", "macd_cross", "rsi_threshold")]] = 0.0
    for g in genomes:
        chosen = pool.selected(opt.decode(g))
        assert 2 <= len(chosen) <= 3
        assert {"sma_cross", "macd_cross", "rsi_threshold"} & set(chosen)  # at least one strategy


def test_equivalent_subsets_share_one_evaluation():
    pool = PoolEnsemble(["sma_cross", "macd_cross", "rsi"])
    opt = GeneticOptimizer(pool, make_ohlc(300), backend="numpy", seed=0, min_trades=0)
    g = np.random.default_rng(2).random(len(opt.keys))
    g[opt.keys.index("use.sma_cross")], g[opt.keys.index("use.macd_cross")] = 0.9, 0.1
    other = g.copy()
    for k in ("macd_cross.fast", "macd_cross.slow", "macd_cross.signal"):  # params of the unused member
        other[opt.keys.index(k)] = 1 - g[opt.keys.index(k)]
    assert opt.decode(g) == opt.decode(other)
    fit = opt.evaluate(np.stack([g, other]))
    assert fit[0] == fit[1] and opt.evaluations == 1


def test_complexity_penalty_prefers_fewer_members():
    pool = PoolEnsemble(["sma_cross", "macd_cross"], penalty=0.5)
    params = dict(pool.values)
    assert pool.complexity(params) == 1.0
    assert pool.complexity({**params, "use.macd_cross": 0}) == 0.5


@pytest.mark.parametrize("method", ["ga", "pso"])
def test_search_selects_the_only_member_with_an_edge(method):
    df = make_ohlc(500, seed=6)
    pool = PoolEnsemble([Oracle(), "sma_cross", "rsi_threshold", "bollinger_reversion", "sma"])
    res = OPTIMIZERS[method](pool, df, backend="numpy", population=40, generations=8, seed=0, min_trades=0).run()
    assert pool.selected(res.best_params) == ["oracle"]
    assert res.test.metrics["sharpe"] > 3  # it keeps its edge out-of-sample


def test_loop_based_members_are_only_looped_when_active():
    pool = PoolEnsemble(["sma_cross", "elliott_wave", "elliott"])
    assert not pool.vectorized
    assert pool.vectorized_for({**pool.values, "use.elliott_wave": 0, "use.elliott": 0})
    res = GeneticOptimizer(pool, make_ohlc(400, seed=3), backend="numpy", population=12, generations=2,
                           seed=0, min_trades=0).run()
    assert res.test is not None


def _fake(method, params, fit):
    class R:
        metrics = {"sharpe": 0.0, "total_return": 0.0, "cagr": 0.0, "max_drawdown": 0.1}

    return OptimizationResult(params, fit, [fit], None, R(), R(), 100, method)


def test_comparison_reports_member_agreement():
    pool = PoolEnsemble(["sma_cross", "macd_cross", "rsi"])
    base = pool.canonical(pool.values)
    ga = {**base, "use.rsi": 0, "sma_cross.fast": 10}
    pso = {**base, "use.macd_cross": 0, "macd_cross.fast": 40, "sma_cross.fast": 12}
    cmp = compare(_fake("ga", pool.canonical(ga), 1.0), _fake("pso", pool.canonical(pso), 1.0), pool.param_space())
    assert cmp.members == {"sma_cross": (True, True), "macd_cross": (True, False), "rsi": (False, True)}
    assert "sma_cross.fast" in cmp.param_diff
    assert not any(k.startswith(("macd_cross.", "rsi.")) for k in cmp.param_diff)  # used by only one method
    assert cmp.param_diff["use.macd_cross"] == 1 and cmp.verdict != "consistent"


def test_cli_optimize_pool_and_reuse(monkeypatch):
    from typer.testing import CliRunner

    from tradebot.cli import main as cli

    monkeypatch.setattr(cli, "load_data", lambda *a, **k: make_ohlc(600, seed=11))
    env = {"COLUMNS": "400"}
    res = CliRunner().invoke(cli.app, ["optimize", "X", "--pool", "sma_cross,macd_cross,sma", "--max-members", "2",
                                       "--population", "16", "--generations", "2", "--backend", "numpy",
                                       "--min-trades", "0"], env=env)
    assert res.exit_code == 0, res.output
    assert "pool members" in res.output and "filter" in res.output
    reuse = res.output.split("reuse with: ")[1].splitlines()[0]
    args = shlex.split(reuse)[1:]
    assert args[:3] == ["backtest", "X", "pool"] and "--members" in args
    again = CliRunner().invoke(cli.app, [*args, "--no-plot"], env=env)
    assert again.exit_code == 0, again.output

    both = CliRunner().invoke(cli.app, ["optimize", "X", "--pool", "sma_cross,macd_cross", "--method", "both",
                                        "--population", "12", "--generations", "2", "--backend", "numpy",
                                        "--min-trades", "0"], env=env)
    assert both.exit_code == 0, both.output
    assert "agree" in both.output and "use." not in both.output.split("GA vs PSO: parameters")[1].split("results")[0]

    bad = CliRunner().invoke(cli.app, ["optimize", "X", "sma_cross", "--pool", "sma"], env=env)
    assert bad.exit_code != 0 and "leave out the algorithm" in bad.output


def test_pool_is_a_registered_algorithm():
    assert "pool" in ALGORITHMS and not ALGORITHMS.get("pool").poolable
    algo = build_algorithm("pool", {"threshold": 0.6}, ["macd_cross", "sma"])
    assert algo.selected() == ["macd_cross", "sma"] and algo.values["threshold"] == 0.6
