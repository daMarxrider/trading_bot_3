# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Strategy evolution (#83): typed trees, simplifier, evaluator, generated plugins, the search and its split."""

import re

import numpy as np
import pandas as pd
import pytest

from tradebot.core import ops
from tradebot.core.series import OHLC
from tradebot.evolve import Evaluator, Evolver, Node, Program, PrimitiveSet, evolve, plugin_source, shuffle_candles
from tradebot.evolve import runtime as rt
from tradebot.evolve.export import compile_signals, load_plugin, parameters
from tradebot.evolve.primitives import BOOL, FRACTION, HOLD, LEVELS, OSC, WINDOW, causal_outputs, indicator_outputs
from tradebot.evolve.program import check, const, random_program, simplify_tree
from tradebot.indicators.base import INDICATORS

from .conftest import make_ohlc

PS = PrimitiveSet.default()
SMALL = dict(population=24, generations=3, min_trades=0, seed=0)


def node(name, *args):
    return Node(PS[name], tuple(args))


CLOSE = node("close")


def sma(n, x=CLOSE):
    return node("sma:price", x, const(WINDOW, n))


def planted(n: int = 1500, seed: int = 0) -> pd.DataFrame:
    """Prices that drift up while the close is above its 20-candle average and down otherwise."""
    rng = np.random.default_rng(seed)
    close = [100.0] * 25
    for _ in range(25, n):
        drift = 0.004 if close[-1] > np.mean(close[-20:]) else -0.004
        close.append(close[-1] * np.exp(drift + rng.normal(0, 0.01)))
    close = np.array(close)
    open_ = np.r_[close[0], close[:-1]]
    spread = np.abs(rng.normal(0, 0.004, n)) * close
    return pd.DataFrame({"open": open_, "high": np.maximum(open_, close) + spread,
                         "low": np.minimum(open_, close) - spread, "close": close},
                        index=pd.date_range("2015-01-01", periods=n, freq="D", tz="UTC"))


# -- primitives and trees -----------------------------------------------------------------------------
def test_hindsight_outputs_are_left_out():
    assert set(causal_outputs(INDICATORS.get("elliott"))) == {"state"}  # zigzag and wave labels look ahead
    used = {(cls.name, out) for cls, out, _ in indicator_outputs()}
    assert ("rsi", "value") in used and ("macd", "hist") in used
    assert not any(name == "elliott" for name, _ in used)  # state is hidden and declares no unit


def test_units_follow_the_indicator():
    units = {(cls.name, out): unit for cls, out, unit in indicator_outputs()}
    assert units[("rsi", "value")] == units[("stoch", "k")] == OSC
    assert units[("bbands", "lower")] == "price" and units[("atr", "value")] == "dist"
    assert units[("macd", "signal")] == "macd"  # its own unit, compared with its other outputs and level 0


def test_random_trees_are_type_valid_and_within_limits():
    rng = np.random.default_rng(1)
    for i in range(200):
        p = random_program(PS, int(rng.integers(2, 7)), rng, full=bool(i % 2))
        check(p.entry, BOOL, PS, state=False)  # entry rules never look at the open trade
        check(p.exit, BOOL, PS)
        assert p.depth <= 6


def test_crossover_and_mutation_keep_trees_valid():
    ev = Evolver(make_ohlc(500), population=40, generations=1, max_depth=5, max_size=25, seed=2, min_trades=0)
    pop = ev.initial_population()
    assert len({p.key for p in pop}) == len(pop) == 40  # no duplicates
    fit = np.array([ev.fitness(p) for p in pop])
    for _ in range(4):
        pop = ev.breed(pop, fit, 40, set())
        fit = np.array([ev.fitness(p) for p in pop])
        for p in pop:
            check(p.entry, BOOL, ev.ps, state=False)
            check(p.exit, BOOL, ev.ps)
            assert p.depth <= 5 and p.entry.size <= 25 and p.exit.size <= 25


def test_without_state_no_rule_looks_at_the_trade():
    ps = PrimitiveSet.default(state=False)
    rng = np.random.default_rng(3)
    assert not any(random_program(ps, 5, rng).uses_state for _ in range(100))


def test_simplifier_canonical_forms():
    rsi = node("above:osc", node("rsi.value:osc", const(INDICATORS.get("rsi").params["period"], 9)),
               const(LEVELS[OSC], 50))
    up = node("gt:price", CLOSE, sma(20))
    s = lambda n: simplify_tree(n, PS)  # noqa: E731
    assert s(node("gt:price", sma(20), sma(20))).key == "never"
    assert s(node("and", up, up)).key == up.key
    assert s(node("not", node("not", up))).key == up.key
    assert s(node("or", up, node("not", up))).key == "always"
    assert s(node("and", up, node("never"))).key == "never"
    # and-chains are flattened, de-duplicated and sorted: the order the parts were combined in doesn't matter
    assert s(node("and", node("and", up, rsi), up)).key == s(node("and", rsi, up)).key
    assert str(s(node("and", rsi, up))) in ("close > sma(close, 20) and rsi(9) > 50", "rsi(9) > 50 and close > sma(close, 20)")


def test_text_and_parameters():
    p = Program(node("crosses_above:price", node("ema:price", CLOSE, const(WINDOW, 13)), sma(41)),
                node("or", node("lt:price", CLOSE, node("shift:price", node("lowest:price", node("low"), const(WINDOW, 7)),
                                                         const(PS["shift:price"].args[1], 1))),
                     node("held", const(HOLD, 30))))
    assert str(p.entry) == "crosses_above(ema(close, 13), sma(close, 41))"
    assert "bars_in_trade >= 30" in str(p.exit) and "close < shift(lowest(low, 7), 1)" in str(p.exit)
    params = parameters(p)
    assert list(params) == ["n1", "n2", "n3", "n4", "n5"] and params["n1"].default == 13
    assert "ema(close, n1)" in params["n1"].description


# -- evaluation ---------------------------------------------------------------------------------------
def test_generated_code_matches_the_evaluator():
    o = OHLC.from_frame(make_ohlc(800, seed=4))
    ev = Evaluator(o)
    rng = np.random.default_rng(5)
    with_state = 0
    for _ in range(150):
        p = random_program(PS, int(rng.integers(2, 6)), rng)
        with_state += p.uses_state
        with np.errstate(all="ignore"):
            code = np.nan_to_num(np.asarray(compile_signals(p)(o), float))
        np.testing.assert_array_equal(ev.positions(p), np.broadcast_to(code, (800,)), err_msg=str(p))
    assert with_state > 10  # trade-state exits are covered too


def test_trade_loop_equals_the_latch_without_state():
    df = make_ohlc(600, seed=6)
    close = df.close.to_numpy()
    rng = np.random.default_rng(0)
    entry, exit_ = rng.random(600) < 0.05, rng.random(600) < 0.08
    exit_[np.flatnonzero(entry)[:3]] = True  # entry and exit on the same candle: the exit wins
    np.testing.assert_array_equal(rt.in_trade(entry, lambda t: t.cut(exit_), close), ops.latch(entry, exit_))


def test_trade_state_exits():
    close = np.r_[np.full(5, 100.0), np.linspace(100, 120, 10), np.linspace(120, 90, 10)]
    entry = np.zeros(len(close), bool)
    entry[5] = True
    held = rt.in_trade(entry, lambda t: t.bars >= 4, close)
    assert held[5:9].all() and held.sum() == 4
    stop = rt.in_trade(entry, lambda t: t.stop(0.05), close)  # out when the close falls 5 % below 100
    assert stop[close < 95].sum() == 0 and stop.sum() > 10
    trail = rt.in_trade(entry, lambda t: t.trail(0.05), close)  # out 5 % below the high of 120
    assert trail.sum() < stop.sum() and close[np.flatnonzero(trail)[-1] + 1] < 120 * 0.95


def test_no_look_ahead():
    """Positions on the first candles don't change when later candles are added."""
    full = OHLC.from_frame(make_ohlc(700, seed=8))
    part = OHLC(*(x[:450] for x in full.fields()))
    rng = np.random.default_rng(9)
    ev_full, ev_part = Evaluator(full), Evaluator(part)
    for _ in range(80):
        p = random_program(PS, 5, rng)
        np.testing.assert_array_equal(ev_full.positions(p)[:450], ev_part.positions(p), err_msg=str(p))


# -- export -------------------------------------------------------------------------------------------
def _sample_programs():
    stateless = Program(node("crosses_above:price", CLOSE, sma(30)), node("lt:price", CLOSE, sma(30)))
    stateful = Program(node("gt:price", CLOSE, sma(30)), node("or", node("stop", const(FRACTION, 0.04)),
                                                              node("held", const(HOLD, 12))))
    return stateless, stateful


@pytest.mark.parametrize("program", _sample_programs(), ids=["stateless", "trade-state"])
def test_saved_plugin_reproduces_the_positions(program, tmp_path):
    from tradebot.algorithms import ALGORITHMS
    from tradebot.core.registry import load_plugins

    name = "evolved_test_" + ("state" if program.uses_state else "plain")
    (tmp_path / f"{name}.py").write_text(plugin_source(program, name, ["made in a test"]))
    assert load_plugins([tmp_path])
    algo = ALGORITHMS.get(name)()
    assert algo.vectorized == (not program.uses_state)
    df = make_ohlc(500, seed=10)
    expected = Evaluator(OHLC.from_frame(df)).positions(program)
    np.testing.assert_array_equal(algo.run(df).to_numpy(), expected)
    assert {"n1"} <= set(algo.param_space()) and [i.name for i in algo.chart_indicators()] == ["sma"]


def test_exported_strategy_can_be_fine_tuned():
    from tradebot.optimize import GeneticOptimizer

    stateless, stateful = _sample_programs()
    for p in (stateless, stateful):
        algo = load_plugin(plugin_source(p, "evolved_tune"))()
        res = GeneticOptimizer(algo, make_ohlc(600, seed=12), population=12, generations=2, seed=0, backend="numpy",
                               min_trades=0).run()
        assert set(res.best_params) == set(algo.param_space())


def test_invalid_plugin_names_are_refused():
    with pytest.raises(ValueError):
        plugin_source(_sample_programs()[0], "Not-Valid")


# -- the search ---------------------------------------------------------------------------------------
def test_recovers_a_planted_rule():
    df = planted()
    res = Evolver(df, population=120, generations=12, seed=0).run()
    c = df.close
    avg = c.rolling(20).mean()
    truth = ops.latch((c > avg).to_numpy(), (c < avg).to_numpy())
    test = res.symbols[0].test.positions.to_numpy()
    assert (test == truth[-len(test):]).mean() > 0.85, str(res.program)
    assert res.aggregated("test") > 2 and res.beats("test")["buy & hold"] == 1


def test_split_by_date_and_no_leak_into_training():
    data = {"A": make_ohlc(1000, seed=1), "B": make_ohlc(800, seed=2)}
    ev = Evolver(data, validation=0.2, test=0.2, segments=3, **SMALL)
    assert ev.validation_start == data["A"].index[600] and ev.test_start == data["A"].index[800]
    a, b = ev.data
    assert a.train_end == 600 and a.validation == (600, 800) and a.test == (800, 1000) and len(a.segments) == 3
    assert b.test is None  # B ends before the test part starts
    program = Program(node("gt:price", CLOSE, sma(10)), node("lt:price", CLOSE, sma(10)))
    before = ev.fitness(program)
    changed = {k: f.copy() for k, f in data.items()}
    for f in changed.values():  # anything after the training part must not change the fitness
        f.iloc[600:, :4] *= np.random.default_rng(0).uniform(0.5, 1.5, (len(f) - 600, 1))
    assert Evolver(changed, validation=0.2, test=0.2, segments=3, **SMALL).fitness(program) == before


def test_too_few_trades_are_rejected():
    ev = Evolver(make_ohlc(600), population=8, generations=1, min_trades=4, seed=0)
    never = Program(node("never"), node("never"))
    assert ev.fitness(never) < -1e8


def test_result_reports_parts_baselines_and_history():
    res = evolve({"A": make_ohlc(900, seed=3), "B": make_ohlc(900, seed=4)}, **SMALL)
    assert len(res.history) == 3 and res.evaluations > 0 and res.seed == 0
    assert {r.symbol for r in res.symbols} == {"A", "B"}
    for part in ("train", "validation", "test"):
        assert set(res.objectives(part)) == {"A", "B"}
    row = res.rows()[0]
    assert {"train", "validation", "test", "xirr", "vs buy & hold", "trades"} <= set(row)
    assert res.program.key in {c.program.key for c in res.hall_of_fame}
    picked = next(c for c in res.hall_of_fame if c.program.key == res.program.key)
    assert picked.validation == max(c.validation for c in res.hall_of_fame)  # chosen on validation
    assert res.algorithm("evolved_report").run(make_ohlc(300)).isin([0.0, 1.0]).all()


def test_without_unseen_data_it_says_so():
    res = evolve(make_ohlc(500), validation=0, test=0, **SMALL)
    assert res.unseen is None and "in-sample" in res.warnings()[0]


def test_shuffled_candles_keep_the_moves():
    df = make_ohlc(300, seed=5)
    out = shuffle_candles(df, np.random.default_rng(0))
    ret = lambda f: np.sort(np.diff(np.log(f.close.to_numpy())))  # noqa: E731
    assert out.index.equals(df.index) and not np.allclose(out.close, df.close)
    assert (out.high >= out[["open", "close"]].max(axis=1) - 1e-9).all()
    np.testing.assert_allclose(out.close.iloc[-1], df.close.iloc[-1], rtol=1e-9)  # same moves, same end
    assert ret(out).shape == ret(df).shape


def test_permutation_test():
    phases = []
    res = evolve(make_ohlc(700, seed=6), permutations=2, on_phase=phases.append, **SMALL)
    pt = res.permutation
    assert len(pt.shuffled) == 2 and 1 / 3 <= pt.p <= 1 and pt.part == "test"
    assert phases == ["evolving", "shuffled 1/2", "shuffled 2/2"]


def test_bad_settings_say_why():
    with pytest.raises(ValueError, match="at most 0.8"):
        Evolver(make_ohlc(500), validation=0.5, test=0.5)
    with pytest.raises(ValueError, match="training candles"):
        Evolver(make_ohlc(30), segments=5)


def plain(result) -> str:
    """CLI output without colour codes (CI's terminal gets colours, which split words)."""
    return re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", result.output)


def test_cli_evolve_and_save(monkeypatch, tmp_path):
    from typer.testing import CliRunner

    import tradebot.cli.main as cli

    monkeypatch.setattr(cli, "load_data", lambda *a, **k: make_ohlc(800, seed=7))
    args = ["evolve", "X", "--population", "16", "--generations", "2", "--min-trades", "0", "--seed", "1",
            "--save", "evolved_cli", "--plugin-dir", str(tmp_path)]
    res = CliRunner().invoke(cli.app, args, env={"COLUMNS": "200"})
    assert res.exit_code == 0, res.output
    out = plain(res)
    assert "evolved rules" in out and "validation" in out and "seed 1" in out
    assert "tradebot backtest X evolved_cli" in out
    assert "class EvolvedCli(BaseAlgorithm)" in (tmp_path / "evolved_cli.py").read_text()
    again = CliRunner().invoke(cli.app, args)
    assert again.exit_code != 0 and "exists" in plain(again)  # never overwritten without --force
    for name in ("sma_cross", "class"):  # an existing algorithm, a Python keyword: refused before the search
        bad = CliRunner().invoke(cli.app, ["evolve", "X", "--save", name, "--plugin-dir", str(tmp_path)])
        assert bad.exit_code != 0 and "--save" in plain(bad)
    monkeypatch.setenv("TRADEBOT_PLUGIN_DIR", str(tmp_path))
    bt = CliRunner().invoke(cli.app, ["backtest", "X", "evolved_cli", "--no-plot"], env={"COLUMNS": "200"})
    assert bt.exit_code == 0, bt.output


def test_best_result_settings_cover_the_evolver_and_scale_with_data():
    import inspect

    from tradebot.evolve import Evolver, best_result_settings

    small, big = best_result_settings(500), best_result_settings(6000)
    assert big["max_depth"] > small["max_depth"] and big["min_trades"] >= small["min_trades"]
    assert big["folds"] > small["folds"] == 0
    assert best_result_settings(3000, from_scratch=True)["max_depth"] >= 8  # an RSI fits
    assert best_result_settings(3000, 5)["aggregate"] == "mean-std"
    allowed = set(inspect.signature(Evolver).parameters) | {"permutations", "folds"}
    for kw in (small, big, best_result_settings(3000, from_scratch=True)):
        assert set(kw) <= allowed
        assert kw["population"] > 300 and kw["generations"] > 40 and kw["permutations"] >= 19
        extra = {k: v for k, v in kw.items() if k not in ("permutations", "folds")}
        Evolver(make_ohlc(900), **{**extra, "population": 8}, seed=1)  # valid


def test_best_result_cli_flag_is_overridable(monkeypatch):
    from typer.testing import CliRunner

    from tradebot.cli import main as cli

    seen = {}

    def fake(df, **kw):
        seen.update(kw)
        raise ValueError("stop here")

    monkeypatch.setattr("tradebot.evolve.evolve", fake)
    monkeypatch.setattr(cli, "load_data", lambda *a, **k: make_ohlc(900))
    CliRunner().invoke(cli.app, ["evolve", "X", "--best-result", "--generations", "3"])
    assert seen["generations"] == 3 and seen["population"] > 300 and seen["permutations"] == 20
