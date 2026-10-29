# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Optimizing on several symbols at once (#84): aggregated fitness, per-symbol results, skipped histories."""

import numpy as np
import pytest

from tradebot.algorithms import build_algorithm
from tradebot.optimize import GeneticOptimizer, run_comparison
from tradebot.optimize.base import AGGREGATES, carried_by_few, select_frames

from .conftest import make_ohlc


def frames(n=3, length=500):
    return {f"S{i}": make_ohlc(length, seed=10 + i) for i in range(n)}


def test_aggregates():
    f = np.array([[1.0, 4.0], [2.0, 0.0], [6.0, 2.0]])  # 3 symbols x 2 candidates
    np.testing.assert_allclose(AGGREGATES["mean"](f), [3.0, 2.0])
    np.testing.assert_allclose(AGGREGATES["median"](f), [2.0, 2.0])
    np.testing.assert_allclose(AGGREGATES["min"](f), [1.0, 0.0])
    np.testing.assert_allclose(AGGREGATES["mean-std"](f), [3.0 - np.std([1, 2, 6]), 2.0 - np.std([4, 0, 2])])


def test_single_frame_is_unchanged():
    df = make_ohlc(500, seed=3)
    genomes = np.random.default_rng(0).random((32, 3))
    algo = build_algorithm("rsi_threshold")
    kw = dict(backend="numpy", seed=1, min_trades=0)
    single = GeneticOptimizer(algo, df, **kw)
    as_dict = GeneticOptimizer(algo, {"X": df}, **kw)
    np.testing.assert_allclose(single.evaluate(genomes), as_dict.evaluate(genomes))
    assert single.run().symbols == [] and len(GeneticOptimizer(algo, {"X": df}, **kw, generations=1).run().symbols) == 1


@pytest.mark.parametrize("aggregate", list(AGGREGATES))
def test_fitness_is_the_aggregate_of_the_symbols(aggregate):
    data = frames()
    algo = build_algorithm("rsi_threshold")
    genomes = np.random.default_rng(1).random((16, 3))
    per = np.stack([GeneticOptimizer(algo, df, backend="numpy", min_trades=0).evaluate(genomes) for df in data.values()])
    multi = GeneticOptimizer(algo, data, backend="numpy", min_trades=0, aggregate=aggregate).evaluate(genomes)
    np.testing.assert_allclose(multi, AGGREGATES[aggregate](per))


def test_robust_parameters_beat_ones_that_fit_one_symbol():
    """A parameter set great on one symbol but bad on the others loses to one that is decent everywhere."""
    class Fake(GeneticOptimizer):  # fitness per symbol from a table instead of a backtest
        table = {0: {"S0": 3.0, "S1": -1.0, "S2": -1.0}, 1: {"S0": 0.5, "S1": 0.5, "S2": 0.5}}

        def _symbol_fitness(self, d, group):
            return np.array([self.table[int(p["fast"] > 50)][d.symbol] for p in group])

    data = frames()
    space = build_algorithm("sma_cross").param_space()["fast"]
    genome = lambda fast: [(fast - space.low) / (space.high - space.low), 0.9]  # noqa: E731
    one_chart, robust = np.array([genome(10), genome(80)])
    for aggregate in ("mean", "median", "min", "mean-std"):
        fit = Fake(build_algorithm("sma_cross"), data, backend="numpy", aggregate=aggregate).evaluate(
            np.array([one_chart, robust]))
        assert fit[1] > fit[0], aggregate
    # on S0 alone the one-chart settings win
    alone = Fake(build_algorithm("sma_cross"), {"S0": data["S0"]}, backend="numpy").evaluate(np.array([one_chart, robust]))
    assert alone[0] > alone[1]


def test_short_histories_are_skipped_or_aligned():
    data = {"LONG": make_ohlc(600, seed=1), "SHORT": make_ohlc(600, seed=2).iloc[-150:], "EMPTY": make_ohlc(10).iloc[:0]}
    kept, skipped = select_frames(data)
    assert list(kept) == ["LONG"] and set(skipped) == {"SHORT", "EMPTY"} and "150 of 600" in skipped["SHORT"]
    kept, skipped = select_frames(data, align=True)
    assert set(kept) == {"LONG", "SHORT"} and len(kept["LONG"]) == 150 and set(skipped) == {"EMPTY"}
    assert kept["LONG"].index[0] == data["SHORT"].index[0]
    with pytest.raises(ValueError, match="enough data"):
        select_frames({"EMPTY": data["EMPTY"]})


def test_result_per_symbol_with_baselines():
    data = {**frames(3), "SHORT": make_ohlc(100, seed=9)}
    res = GeneticOptimizer(build_algorithm("sma_cross"), data, backend="numpy", population=16, generations=3,
                           seed=0, min_trades=0).run()
    assert [r.symbol for r in res.symbols] == ["S0", "S1", "S2"] and "SHORT" in res.skipped
    for r in res.symbols:
        assert len(r.train.equity) == 350 and len(r.test.equity) == 150
        assert r.test_baselines is not None
    assert res.train is res.symbols[0].train
    assert set(res.objectives("test")) == {"S0", "S1", "S2"}
    assert res.aggregated("train") == pytest.approx(np.mean(list(res.objectives("train").values())))
    assert set(res.beats()) == {"buy & hold", "monthly investing"} and all(0 <= v <= 3 for v in res.beats().values())
    rows = res.rows()
    assert [r["symbol"] for r in rows] == ["S0", "S1", "S2"] and {"vs buy & hold", "trades", "xirr"} <= set(rows[0])


def test_carried_by_few():
    assert carried_by_few([3.0, 0.1, 0.0, 0.1])  # one symbol makes the mean
    assert not carried_by_few([1.0, 0.9, 1.1])
    assert not carried_by_few([3.0, 0.0])  # too few symbols to tell
    assert not carried_by_few([-1.0, -2.0, -3.0])


def test_comparison_uses_the_aggregate_out_of_sample():
    cmp = run_comparison(build_algorithm("sma_cross"), frames(), population=12, generations=2, seed=2,
                         backend="numpy", min_trades=0, aggregate="median")
    assert cmp.ga.aggregate == "median" and len(cmp.pso.symbols) == 3
    assert cmp.oos["ga"] == pytest.approx(cmp.ga.aggregated("test"))


def test_cli_several_symbols_and_watchlists(monkeypatch):
    from typer.testing import CliRunner

    from tradebot.cli import main as cli

    seeds = iter(range(100))
    monkeypatch.setattr(cli, "load_data", lambda s, *a, **k: make_ohlc(100 if s == "NEW" else 500, seed=next(seeds)))
    monkeypatch.setattr("tradebot.config.watchlists", lambda: {"Main": ["A", "B"], "Crypto": ["C", "A"]})
    run = lambda *args: CliRunner().invoke(cli.app, [*args, "--population", "12", "--generations", "2",  # noqa: E731
                                                     "--backend", "numpy", "--min-trades", "0"], env={"COLUMNS": "160"})
    res = run("optimize", "A,B,NEW", "sma_cross", "--aggregate", "min")
    assert res.exit_code == 0, res.output
    assert "per symbol" in res.output and "skipped NEW" in res.output and "beats on" in res.output
    assert "tradebot backtest A,B sma_cross" in res.output  # the reuse command leaves out skipped symbols
    res = run("optimize", "-w", "all", "sma_cross")  # the symbols argument can be left out
    assert res.exit_code == 0, res.output
    assert "3 symbols" in res.output
    assert "backtest A,B,C sma_cross" in res.output
    res = run("optimize", "X", "-w", "Crypto", "sma_cross", "--method", "both")
    assert res.exit_code == 0, res.output
    assert "per symbol (GA)" in res.output and "per symbol (PSO)" in res.output and "verdict:" in res.output
    assert run("optimize", "-w", "nope", "sma_cross").exit_code != 0
    assert run("optimize", "A,B", "sma_cross", "--aggregate", "max").exit_code != 0
    bt = CliRunner().invoke(cli.app, ["backtest", "A,B", "sma_cross"], env={"COLUMNS": "160"})
    assert bt.exit_code == 0, bt.output
    assert "beats buy & hold on" in bt.output


def test_watchlist_completion(monkeypatch):
    from tradebot.cli import complete

    monkeypatch.setattr("tradebot.config.watchlists", lambda: {"Main": ["A", "B"], "Crypto": ["C"]})
    assert [v for v, _ in complete.watchlists("")] == ["Main", "Crypto", "all"]
    assert [v for v, _ in complete.watchlists("cr")] == ["Crypto"]
