# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Evolve from scratch (#100): unit-checked arithmetic, low-level primitives that rebuild the classic indicators,
evolved building blocks (modules) and the overfitting safeguards."""

import re

import numpy as np
import pandas as pd
import pytest

from tradebot.backtest.metrics import deflated_sharpe
from tradebot.core.series import OHLC
from tradebot.evolve import Evaluator, Evolver, Node, Program, PrimitiveSet, evolve
from tradebot.evolve import runtime as rt
from tradebot.evolve.export import compile_signals, load_plugin, plugin_source
from tradebot.evolve.modules import Library, Template, discover, rewrite
from tradebot.evolve.primitives import (BOOL, DIST, FACTOR, HOLD, LAG, LEVELS, OSC, PRICE, RATIO, WINDOW,
                                        arith_unit)
from tradebot.evolve.program import check, const, expand, random_program, simplify_tree
from tradebot.evolve.search import SCRATCH_JOINT, SCRATCH_LIMITS, SCRATCH_NORMALISE, SCRATCH_REFINE
from tradebot.indicators.base import INDICATORS

from .conftest import make_ohlc
from .test_evolve import planted, plain

PS = PrimitiveSet.from_scratch()
SMALL = dict(population=24, generations=3, min_trades=0, seed=0, from_scratch=True)


def node(name, *args):
    return Node(PS[name], tuple(args))


CLOSE, HIGH, LOW = node("close"), node("high"), node("low")


def wilder(x, n):
    return node(f"wilder:{x.unit}", x, const(WINDOW, n))


def rsi_tree(n=14):
    move = node("diff:price", CLOSE, const(LAG, 1))
    gain, loss = wilder(node("pos:dist", move), n), wilder(node("neg:dist", move), n)
    return node("pct:osc", node("div:dist,dist", gain, node("add:dist,dist", gain, loss)))


def stoch_tree(n=14):
    ll, hh = node("lowest:price", LOW, const(WINDOW, n)), node("highest:price", HIGH, const(WINDOW, n))
    return node("pct:osc", node("div:dist,dist", node("sub:price,price", CLOSE, ll), node("sub:price,price", hh, ll)))


def bollinger_tree(n=20, k=2.0):
    width = node("mul:dist,ratio", node("rolling_std:price", CLOSE, const(WINDOW, n)), node("const:ratio", const(FACTOR, k)))
    return node("add:price,dist", node("sma:price", CLOSE, const(WINDOW, n)), width)


def atr_tree(n=14):
    prev = node("shift:price", CLOSE, const(LAG, 1))
    gap = lambda p: node("abs:dist", node("sub:price,price", p, prev))  # noqa: E731
    tr = node("max:dist", node("sub:price,price", HIGH, LOW), node("max:dist", gap(HIGH), gap(LOW)))
    return wilder(tr, n)


def macd_tree(fast=12, slow=26):
    return node("sub:price,price", node("ema:price", CLOSE, const(WINDOW, fast)), node("ema:price", CLOSE, const(WINDOW, slow)))


# -- units ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("op, a, b, unit", [
    ("div", PRICE, PRICE, RATIO), ("div", DIST, PRICE, RATIO), ("div", DIST, DIST, RATIO),
    ("mul", RATIO, RATIO, RATIO), ("div", RATIO, RATIO, RATIO), ("mul", PRICE, RATIO, PRICE),
    ("mul", DIST, RATIO, DIST), ("add", PRICE, DIST, PRICE), ("sub", PRICE, DIST, PRICE),
    ("sub", PRICE, PRICE, DIST), ("add", DIST, DIST, DIST), ("sub", RATIO, RATIO, RATIO),
    ("mul", PRICE, PRICE, None), ("add", PRICE, RATIO, None), ("add", PRICE, PRICE, None),
    ("sub", DIST, PRICE, None), ("div", RATIO, PRICE, None), ("mul", PRICE, DIST, None), ("add", OSC, OSC, None),
])
def test_dimensional_analysis(op, a, b, unit):
    assert arith_unit(op, a, b) == unit


def test_the_set_has_no_ready_made_indicators_and_only_valid_arithmetic():
    assert not any(p.kind == "indicator" for p in PS.primitives)
    assert not {"change:ratio", "gap:ratio", "volatility:ratio"} & set(PS.by_name)
    arith = [p for p in PS.primitives if p.family in ("add", "sub", "mul", "div")]
    assert {p.name for p in arith} >= {"div:dist,dist", "mul:price,ratio", "sub:price,price", "add:price,dist"}
    assert all(arith_unit(p.family, *p.args) == p.ret for p in arith)
    assert "mul:price,price" not in PS and "add:price,price" not in PS
    with pytest.raises(TypeError):  # a tree mixing units is refused like any other type error
        check(node("lt:price", CLOSE, node("sub:price,price", CLOSE, LOW)), BOOL, PS)
    rng = np.random.default_rng(3)
    for i in range(150):  # random trees are valid trees of the set
        p = random_program(PS, int(rng.integers(2, 9)), rng, full=bool(i % 2))
        check(p.entry, BOOL, PS, state=False)
        check(p.exit, BOOL, PS)


# -- the classic indicators, rebuilt ---------------------------------------------------------------------
@pytest.mark.parametrize("tree, unit, name, params, output", [
    (rsi_tree(), OSC, "rsi", {"period": 14}, "value"),
    (stoch_tree(), OSC, "stoch", {"period": 14, "smooth": 3}, "k"),
    (bollinger_tree(), PRICE, "bbands", {"period": 20, "width": 2.0}, "upper"),
    (atr_tree(), DIST, "atr", {"period": 14}, "value"),
    (macd_tree(), DIST, "macd", {"fast": 12, "slow": 26, "signal": 9}, "macd"),
])
def test_classic_indicators_can_be_built_from_scratch(tree, unit, name, params, output):
    check(tree, unit, PS)
    o = OHLC.from_frame(make_ohlc(1200, seed=4))
    ours = np.asarray(Evaluator(o).value(tree), dtype=float)
    ref = np.asarray(INDICATORS.get(name).compute(o, **params)[output], dtype=float)
    both = np.isfinite(ours) & np.isfinite(ref)
    assert both.sum() > 1100
    # ATR seeds its smoothing one candle earlier in the built-in (true range of the first candle); that fades out
    np.testing.assert_allclose(ours[both][300:], ref[both][300:], rtol=1e-9, atol=1e-9)
    assert np.isfinite(ours[400:]).all()


def test_rsi_prints_as_the_formula():
    assert str(rsi_tree()) == ("100 * wilder(pos(diff(close, 1)), 14) / (wilder(pos(diff(close, 1)), 14) + "
                               "wilder(neg(diff(close, 1)), 14))")
    width = node("mul:dist,ratio", node("sub:price,price", HIGH, LOW), node("const:ratio", const(FACTOR, 2.0)))
    assert str(node("div:dist,dist", node("sub:price,price", CLOSE, LOW), width)) == "(close - low) / ((high - low) * 2)"


# -- division, overflow, constants -------------------------------------------------------------------------
def test_division_by_zero_rejects_but_zero_by_zero_is_a_gap():
    a, b = np.array([1.0, 0.0, 2.0, np.nan]), np.array([2.0, 0.0, 4.0, 1.0])
    np.testing.assert_array_equal(rt.div_checked(a, b), [0.5, np.nan, 0.5, np.nan])
    with pytest.raises(rt.Undefined):
        rt.div_checked(np.array([1.0, 2.0]), np.array([1.0, 0.0]))
    np.testing.assert_array_equal(rt.div(np.array([1.0, 2.0]), np.array([1.0, 0.0])), [1.0, np.nan])  # plugins
    with pytest.raises(rt.Undefined):
        rt.finite(np.array([1.0, np.inf]))
    # close / (close - open) divides by zero on a doji: the program is rejected, not an error
    df = make_ohlc(400, seed=1)
    df.iloc[200, df.columns.get_loc("open")] = df["close"].iloc[200]
    shaky = node("div:price,ratio", CLOSE, node("div:dist,price", node("sub:price,price", CLOSE, node("open")), CLOSE))
    ev = Evolver(df, **SMALL)
    prog = Program(node("gt:price", shaky, CLOSE), node("never"))
    assert ev._positions(ev.data[0], prog) is None and ev.fitness(prog) < -1e8


def test_operations_on_constants_fold():
    two, three = node("const:ratio", const(FACTOR, 2.0)), node("const:ratio", const(FACTOR, 3.0))
    folded = simplify_tree(node("highest:ratio", node("add:ratio,ratio", two, three), const(WINDOW, 20)), PS)
    assert folded.prim.name == "const:ratio" and folded.args[0].value == 5.0
    assert simplify_tree(node("gt:ratio", two, three), PS).prim.name == "never"
    assert simplify_tree(node("diff:ratio", two, const(LAG, 3)), PS).args[0].value == 0.0


# -- modules ---------------------------------------------------------------------------------------------
def test_templates_tie_equal_constants():
    t = Template(rsi_tree(14))
    assert len(t.slots) == 2 and set(t.values) == {1, 14}  # the lag and the window, each used three times
    assert Template(rsi_tree(9)).key == t.key  # same shape, another window
    assert Template(stoch_tree()).key != t.key


def test_frequent_subtrees_become_modules_and_programs_use_them():
    level = const(LEVELS[OSC], 30.0)
    programs = [Program(node("below:osc", rsi_tree(n), level), node("held", const(HOLD, h)))
                for n, h in ((14, 5), (9, 3), (21, 8), (14, 12))]
    found = discover(programs, set(), min_count=3)
    assert found and found[0].key == Template(rsi_tree()).key  # the biggest shared part first
    lib = Library()
    ps, new = lib.grow(programs, PS)
    rsi_module = next(m for m in new if m.ret == OSC)
    assert rsi_module.kind == "module" and rsi_module.name in ps and ps.modules
    short = lib.rewrite(programs[1])
    assert short.entry.args[0].prim is rsi_module and short.size < programs[1].size
    assert [a.value for a in short.entry.args[0].args] == [1, 9]  # lag 1, window 9
    o = OHLC.from_frame(make_ohlc(600, seed=2))
    ev = Evaluator(o)
    np.testing.assert_array_equal(ev.positions(short), Evaluator(o).positions(programs[1]))
    assert expand(rsi_module, (1, 9)).key == rsi_tree(9).key
    text = str(short)
    assert f"where\n  {rsi_module.name}(n1, n2) = 100 * wilder(pos(diff(close, n1)), n2)" in text
    assert rewrite(rsi_tree(9), lib.by_template).prim is rsi_module


def test_plugins_define_the_modules():
    lib = Library()
    level = const(LEVELS[OSC], 30.0)
    programs = [Program(node("below:osc", rsi_tree(n), level), node("held", const(HOLD, 5))) for n in (14, 9, 21)]
    lib.grow(programs, PS)
    program = lib.rewrite(programs[0])
    assert program.modules
    source = plugin_source(program, "scratch_plugin_test")
    assert re.search(r"^def m\d+\(o, n1, n2\):", source, re.M)
    o = OHLC.from_frame(make_ohlc(500, seed=5))
    np.testing.assert_array_equal(compile_signals(program)(o), Evaluator(o).positions(program))
    algo = load_plugin(source, "tradebot_test_scratch_plugin", register=False)()
    assert algo.run(make_ohlc(500, seed=5)).isin([0.0, 1.0]).all()


# -- the search and its safeguards ---------------------------------------------------------------------------
def test_evolver_from_scratch_settings():
    ev = Evolver(make_ohlc(800, seed=3), **SMALL)
    assert (ev.max_depth, ev.max_size) == SCRATCH_LIMITS and ev.ps.scratch
    assert ev.library is not None and ev.dedupe and ev.deflate and ev.tune == 3
    assert (ev.joint, ev.normalise, ev.polish) == (SCRATCH_JOINT, SCRATCH_NORMALISE, SCRATCH_REFINE)
    plain_ev = Evolver(make_ohlc(800, seed=3), population=24, generations=1)
    assert (plain_ev.max_depth, plain_ev.max_size) == (6, 40) and plain_ev.library is None and not plain_ev.dedupe
    assert not plain_ev.deflate and plain_ev.tune == 0
    assert (plain_ev.joint, plain_ev.normalise, plain_ev.polish) == (0, 0, 0)
    with pytest.raises(ValueError, match="indicators"):
        Evolver(make_ohlc(800), from_scratch=True, indicators=["rsi"])


def test_islands_evolve_apart_and_swap_their_best():
    ev = Evolver(make_ohlc(800, seed=3), **{**SMALL, "population": 40, "islands": 4, "migrants": 2})
    assert ev.islands == 4 and Evolver(make_ohlc(800, seed=3), population=40, generations=1).islands == 1
    assert Evolver(make_ohlc(800, seed=3), **{**SMALL, "population": 8, "islands": 4}).islands == 2  # >= 4 each
    islands = [(pop := ev.initial_population(10), np.array([ev.fitness(p) for p in pop])) for _ in range(4)]
    moved = ev._migrate(islands)
    for i, (pop, fit) in enumerate(moved):
        src_pop, src_fit = islands[i - 1]
        best = [src_pop[j].key for j in np.argsort(-src_fit, kind="stable")[:2]]
        assert len(pop) == 10 and [p.key for p in pop[-2:]] == best  # the neighbour's best replaced the worst
        np.testing.assert_array_equal(fit, [ev.fitness(p) for p in pop])
    res = ev.run()
    assert len(res.history) == 3 and res.history[-1]["distinct"] > 10 and ev.settings["islands"] == 4


def test_duplicates_count_once():
    ev = Evolver(make_ohlc(800, seed=3), **SMALL)
    a = Program(node("gt:price", CLOSE, node("sma:price", CLOSE, const(WINDOW, 20))), node("never"))
    b = Program(node("lt:price", node("sma:price", CLOSE, const(WINDOW, 20)), CLOSE), node("never"))  # same trades
    fit = np.array([ev.fitness(a), ev.fitness(b)])
    assert fit[0] == pytest.approx(fit[1]) and ev._signature[a.key] == ev._signature[b.key]
    deduped = ev._distinct([a, b], fit)
    assert deduped[0] == fit[0] and deduped[1] < -1e8


def ratio_rule(n, level, smooth="wilder", m=None):
    """``smooth(diff(close, 1), n) / smooth(abs(diff(close, 1)), m) < level``: with wilder and ``m == n`` exactly
    ``RSI(n) < 50 * (1 + level)``; ``ema(x, 2n - 1)`` is Wilder's ``n`` up to the start."""
    move = node("diff:price", CLOSE, const(LAG, 1))
    ratio = node("div:dist,dist", node(f"{smooth}:dist", move, const(WINDOW, n)),
                 node(f"{smooth}:dist", node("abs:dist", move), const(WINDOW, m or n)))
    return node("below:ratio", ratio, const(PS["below:ratio"].args[1], level))


def test_normalised_ratios_are_generic_shapes_with_one_window():
    ev = Evolver(make_ohlc(800, seed=3), **SMALL)
    families = set()
    for _ in range(40):
        r = ev.normalised()
        check(r, RATIO, ev.ps)
        top, bottom = ({n.value for _, n in side.nodes() if n.is_const and n.spec is WINDOW} for side in r.args)
        assert top and top == bottom  # numerator and denominator smoothed (or ranged) over one window
        families.add(r.args[0].prim.name.split(":")[0])
    assert families >= {"wilder", "ema", "sma", "sub"}  # RSI-like and stochastic-like shapes
    cond = ev.compared(ev.normalised())
    check(cond, BOOL, ev.ps)


def test_joint_tuning_finds_the_ridge_one_step_at_a_time_misses():
    df = hidden_rsi()
    ev = Evolver(df, **{**SMALL, "seed": 1})
    start = Program(ratio_rule(7, -0.4), node("held", const(HOLD, 5)))  # the right shape, the wrong window
    fit = ev.fitness(start)
    tuned, tuned_fit = ev.joint_tune(start, fit, draws=40)
    assert tuned_fit >= fit + 1 and tuned_fit == ev.fitness(tuned)
    assert ev.hill_climb(start, fit)[1] < tuned_fit


def test_refine_ties_near_equal_constants_and_lands_on_the_hidden_rule():
    df = hidden_rsi()
    ev = Evolver(df, **{**SMALL, "seed": 1})
    truth = Program(ratio_rule(14, -0.4), node("held", const(HOLD, 5)))
    near = Program(ratio_rule(13, -0.42), node("held", const(HOLD, 5)))  # RSI(13) < 29
    p, f = ev.refine(near, ev.fitness(near))
    assert p.key == truth.key and f == ev.fitness(truth)
    untied = Program(ratio_rule(26, -0.41, "ema", 27), node("held", const(HOLD, 5)))
    p, f = ev.refine(untied, ev.fitness(untied))
    windows = {n.value for _, n in p.entry.nodes() if n.is_const and n.spec is WINDOW}
    assert len(windows) == 1 and f >= ev.fitness(untied) * 0.995  # one window now (ema 27 is wilder 14)
    assert Evolver(df, **{**SMALL, "polish": 0}).polish == 0


def test_a_run_from_scratch_reports_its_safeguards():
    phases = []
    res = evolve(planted(1200), on_phase=phases.append, **{**SMALL, "generations": 6, "library_every": 2})
    assert phases == ["evolving", *(f"shuffled {i}/5" for i in range(1, 6)), *(f"fold {i}/3" for i in range(1, 4))]
    assert res.from_scratch and res.permutation is not None and len(res.permutation.shuffled) == 5
    assert res.deflated is not None and res.deflated.trials > 10 and 0 <= res.deflated.probability <= 1
    assert res.walk_forward is not None and len(res.walk_forward.folds) == 3
    assert all(f.test_start is not None for f in res.walk_forward.folds)
    starts = [f.test_start for f in res.walk_forward.folds]
    assert starts == sorted(starts)  # rolling forward
    keys = [c.program.key for c in res.hall_of_fame]
    assert len(keys) == len(set(keys))
    notes = "\n".join(res.notes())
    assert "deflated Sharpe ratio" in notes and "walk-forward" in notes
    assert "history" not in notes and "modules" in res.history[-1]


def test_deflated_sharpe():
    rng = np.random.default_rng(0)
    edge = rng.normal(0.002, 0.01, 2000)  # per-candle Sharpe 0.2
    trials = rng.normal(0, 0.02, 50)
    few, _ = deflated_sharpe(edge, trials, 10)
    many, luck = deflated_sharpe(edge, trials, 100_000)
    assert few > 0.95 and many < few and luck > 0  # more tries: luck reaches higher, the edge counts for less
    noise, _ = deflated_sharpe(rng.normal(0, 0.01, 2000), trials, 1000)
    assert noise < 0.5


def test_cli_from_scratch(monkeypatch):
    from typer.testing import CliRunner

    import tradebot.cli.main as cli

    monkeypatch.setattr(cli, "load_data", lambda *a, **k: make_ohlc(900, seed=7))
    args = ["evolve", "X", "--from-scratch", "--population", "16", "--generations", "2", "--min-trades", "0",
            "--seed", "1", "--permutations", "1", "--folds", "2", "--islands", "2"]
    res = CliRunner().invoke(cli.app, args, env={"COLUMNS": "220"})
    assert res.exit_code == 0, res.output
    out = plain(res)
    assert "deflated Sharpe ratio" in out and "walk-forward" in out and "permutation test" in out
    bad = CliRunner().invoke(cli.app, ["evolve", "X", "--from-scratch", "--indicators", "rsi"])
    assert bad.exit_code != 0 and "indicators" in plain(bad)


# -- rediscovery (slow) ----------------------------------------------------------------------------------
def hidden_rsi(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    """Prices that drift down, except for the 5 candles after each RSI(14) < 30, which rise strongly."""
    rng = np.random.default_rng(seed)
    close, gain, loss, boost = np.empty(n), 0.0, 0.0, 0
    close[0] = 100.0
    for t in range(1, n):
        close[t] = close[t - 1] * np.exp((0.012 if boost > 0 else -0.0015) + rng.normal(0, 0.012))
        boost -= 1
        move = close[t] - close[t - 1]
        if t <= 14:  # Wilder's smoothing starts with a plain average
            gain, loss = gain + max(move, 0) / 14, loss + max(-move, 0) / 14
        else:
            gain, loss = gain + (max(move, 0) - gain) / 14, loss + (max(-move, 0) - loss) / 14
        if t >= 14 and 100 * gain / (gain + loss) < 30:
            boost = 5
    open_ = np.r_[close[0], close[:-1]]
    spread = np.abs(rng.normal(0, 0.003, n)) * close
    return pd.DataFrame({"open": open_, "high": np.maximum(open_, close) + spread,
                         "low": np.minimum(open_, close) - spread, "close": close},
                        index=pd.date_range("2015-01-01", periods=n, freq="D", tz="UTC"))


@pytest.mark.slow
def test_from_scratch_rediscovers_a_hidden_rsi_rule():
    """From scratch, evolve finds a hidden ``RSI(14) < 30`` rule (held 5 candles) itself (#105).

    Newly built normalised ratios (``smooth(x) / smooth(abs(x))``) get their constants tuned jointly, so the right
    shape is judged at its ridge, not at a random window and level, and :meth:`Evolver.refine` polishes the result
    (``RSI(13) < 29`` to ``RSI(14) < 30``). The evolved entry rule is RSI in another form (``ema(x, 27)`` is Wilder's
    14): over seeds 1-6 its signals overlap the hidden rule's by 0.97 0.98 1.0 0.92 0.22 1.0 (Jaccard); seed 6 finds
    the whole rule, positions and all. Before, no seed found RSI (positions overlapped 0.45 0.37 0.32 0.41). Where the
    entry is found, the positions overlap by 0.78-0.83, not more, because the evolved exits beat ``held 5`` on
    training *and* validation (the hidden boost lasts 5 candles after the *last* signal), so the fittest rule rightly
    isn't the hidden one candle for candle. Single seeds can still miss (seed 5), so 3 of 4 must find the entry rule
    (floating-point differences between CPUs change the runs)."""
    df = hidden_rsi()
    truth = Program(node("below:osc", rsi_tree(14), const(LEVELS[OSC], 30.0)), node("held", const(HOLD, 5)))
    ev0 = Evaluator(OHLC.from_frame(df))
    held, signal = ev0.positions(truth) > 0, np.nan_to_num(np.asarray(ev0.value(truth.entry), dtype=float)) > 0
    runs = []
    for seed in (1, 2, 3, 4):
        ev = Evolver(df, from_scratch=True, population=500, generations=60, min_trades=0, seed=seed)
        res = ev.run()
        found = ev0.positions(res.program) > 0
        fires = np.nan_to_num(np.asarray(ev0.value(res.program.entry), dtype=float)) > 0
        runs.append(((signal & fires).sum() / (signal | fires).sum(), (held & found).sum() / (held | found).sum(),
                     res.fitness / ev.fitness(truth)))
    runs = np.array(runs)
    found_rsi = runs[:, 0] >= 0.9
    assert found_rsi.sum() >= 3, np.round(runs, 2).tolist()  # most seeds find RSI(14) < 30 as the entry
    assert (runs[found_rsi, 1] >= 0.7).all() and (runs[found_rsi, 2] >= 1.0).all(), np.round(runs, 2).tolist()
