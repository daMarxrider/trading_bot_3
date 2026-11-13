# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""The strategy combinator (#22): rules, operators, latching, risk exits, saved combos, optimizer."""

import numpy as np
import pandas as pd
import pytest

from tradebot.algorithms import ALGORITHMS, build_algorithm
from tradebot.algorithms.base import BaseAlgorithm
from tradebot.algorithms.combo import (Combo, check, combine, combos_path, delete_combo, load_combos, load_saved,
                                       save_combo, saved_combos, terms)
from tradebot.core import ops
from tradebot.core.series import OHLC
from tests.conftest import make_ohlc


class _Fixed(BaseAlgorithm):
    """A strategy that replays fixed positions (registered for these tests only)."""

    name = "_combo_fixed"
    poolable = False
    series: dict = {}

    def __init__(self, key: str = "a", **params):
        self.key = key
        super().__init__(**params)

    def signals(self, o, p):
        return np.resize(np.asarray(_Fixed.series[self.key], dtype=float), len(o))  # the dry run uses longer data


ALGORITHMS.unregister(_Fixed.name)  # only the per-test subclasses below belong in the registry: the lab lists everything in it


def fixed(name: str, values) -> None:
    """Register ``name`` as a strategy whose positions are ``values``."""
    _Fixed.series[name] = values
    type(f"Fixed_{name}", (_Fixed,), {"name": name, "__init__": lambda self, **p: _Fixed.__init__(self, name, **p)})


@pytest.fixture(autouse=True)
def _cleanup():
    before = dict(ALGORITHMS._items)
    yield
    ALGORITHMS._items.clear()
    ALGORITHMS._items.update(before)


def frame(close) -> pd.DataFrame:
    close = np.asarray(close, dtype=float)
    idx = pd.date_range("2024-01-01", periods=len(close), freq="D", tz="UTC")
    return pd.DataFrame({"open": close, "high": close, "low": close, "close": close}, index=idx)


def positions(entry, exit=None, close=None, n=8) -> list:
    df = frame(close if close is not None else np.full(n, 100.0))
    return build_algorithm("combo", entry=entry, exit=exit).run(df).astype(int).tolist()


# -- operators -------------------------------------------------------------------------------------------
A, B, C = [1, 1, 0, 0, 1, 0, 1, 0], [1, 0, 1, 0, 1, 1, 0, 0], [1, 1, 1, 0, 0, 0, 1, 1]


@pytest.fixture
def abc():
    for name, values in zip("abc", (A, B, C)):
        fixed(f"t_{name}", values)


@pytest.mark.parametrize("rule, expected", [
    ("t_a & t_b", [x & y for x, y in zip(A, B)]),
    ("t_a and t_b", [x & y for x, y in zip(A, B)]),
    ("t_a | t_b", [x | y for x, y in zip(A, B)]),
    ("t_a OR t_b", [x | y for x, y in zip(A, B)]),
    ("!t_a", [1 - x for x in A]),
    ("not t_a & t_b", [(1 - x) & y for x, y in zip(A, B)]),
    ("!(t_a | t_b)", [1 - (x | y) for x, y in zip(A, B)]),
    ("t_a & t_b | t_c", [(x & y) | z for x, y, z in zip(A, B, C)]),
    ("vote(t_a, t_b, t_c) >= 0.6", [int(x + y + z >= 2) for x, y, z in zip(A, B, C)]),
    ("vote(t_a, t_b, t_c) >= 1", [x & y & z for x, y, z in zip(A, B, C)]),
    ("vote(t_a, t_b, t_c) > 0.3", [x | y | z for x, y, z in zip(A, B, C)]),
    ("weighted(t_a:0.5, t_b:0.3, t_c:0.2) >= 0.5", [int(0.5 * x + 0.3 * y + 0.2 * z >= 0.5) for x, y, z in zip(A, B, C)]),
    ("weighted(t_a:2, t_b:1, t_c:1) >= 0.75", [int((2 * x + y + z) / 4 >= 0.75) for x, y, z in zip(A, B, C)]),
])
def test_truth_tables(abc, rule, expected):
    """Without an exit rule a combo holds exactly while its entry rule holds, so the positions are the truth table."""
    assert positions(rule) == expected


def test_a_strategy_alone_reproduces_it(ohlc):
    for name in ("rsi_threshold", "macd_cross", "bollinger_reversion"):
        expected = build_algorithm(name).run(ohlc)
        np.testing.assert_array_equal(build_algorithm("combo", entry=name).run(ohlc), expected)
        np.testing.assert_array_equal(build_algorithm("combo", entry=name, exit=name).run(ohlc), expected)


def test_strategy_in_the_exit_rule_means_its_exit(abc):
    #          A = [1, 1, 0, 0, 1, 0, 1, 0]   enter while t_a is in
    #          C = [1, 1, 1, 0, 0, 0, 1, 1]   exit while t_c is out
    assert positions("t_a", "t_c") == [1, 1, 1, 0, 0, 0, 1, 1]
    assert positions("t_a", "!t_c") == [0, 0, 0, 0, 1, 1, 0, 0]  # exit while t_c is in


def test_vote_and_weighted_in_the_exit_count_exits(abc):
    fixed("t_on", [1] * 8)
    # exit when at least 2 of 3 members are out: the vote ensemble's exit
    expected = build_algorithm("vote", {"threshold": 0.6}, ["t_a", "t_b", "t_c"]).run(frame([100.0] * 8))
    assert positions("vote(t_a, t_b, t_c) >= 0.6", "vote(t_a, t_b, t_c) >= 0.6") == expected.astype(int).tolist()


# -- latching ------------------------------------------------------------------------------------------
def test_entry_and_exit_are_latched_and_the_exit_wins():
    fixed("e_in", [0, 1, 0, 0, 0, 1, 0, 1, 0, 0])
    fixed("e_out", [1, 1, 1, 0, 1, 1, 1, 0, 1, 1])  # out (= exit) on bars 3 and 7
    # in on 1, held through 2, out on 3; in on 5, held; bar 7: entry and exit together -> exit wins
    assert positions("e_in", "e_out", n=10) == [0, 1, 1, 0, 0, 1, 1, 0, 0, 0]


def test_entry_reference_in_the_exit_rule(abc):
    assert positions("t_a", "!entry") == A
    # in the exit rule, t_a means "t_a is out", which is what !entry says here
    assert positions("t_a", "!entry & t_c") == positions("t_a", "t_a & t_c")


# -- indicator conditions --------------------------------------------------------------------------------
def test_indicator_conditions_match_hand_computed(ohlc):
    from tradebot.indicators.builtin import RSI, SMA

    o = OHLC.from_frame(ohlc)
    rsi, sma = RSI.compute(o, period=14)["value"], SMA.compute(o, period=50)["value"]
    want = ops.latch(rsi < 30, rsi > 60)
    np.testing.assert_array_equal(build_algorithm("combo", entry="rsi(14) < 30", exit="rsi(14) > 60").run(ohlc), want)
    above = (o.close > sma * 1.01).astype(float)
    np.testing.assert_array_equal(build_algorithm("combo", entry="close > sma(50) * 1.01").run(ohlc), above)


def test_pool_rules_and_alerts(ohlc):
    from tradebot.algorithms.pool import IndicatorFilter

    rule = IndicatorFilter("rsi", period=10, level=35).run(ohlc)
    np.testing.assert_array_equal(build_algorithm("combo", entry="rsi(10, 35)").run(ohlc), rule)
    combo = build_algorithm("combo", entry="sma(20).cross_up", exit="sma(20).cross_down")
    from tradebot.indicators.builtin import SMA

    s = SMA.compute(OHLC.from_frame(ohlc), period=20)
    np.testing.assert_array_equal(combo.run(ohlc), ops.latch(s["cross_up"] > 0.5, s["cross_down"] > 0.5))


def test_warm_up_is_unknown_so_not_does_not_enter(ohlc):
    combo = build_algorithm("combo", entry="!(close < sma(100))")
    assert not combo.run(ohlc).iloc[:99].any()  # sma(100) is NaN there: neither true nor false
    assert combo.run(ohlc).iloc[99:].equals((ohlc.close >= ohlc.close.rolling(100).mean()).iloc[99:].astype(float))


def test_same_calls_share_parameters_different_ones_dont():
    combo = check("sma(20) < close & sma.cross_up & sma(50) > sma(200)", "sma(50).cross_down")
    assert [k for k in combo.param_space()] == ["sma.period", "sma_2.period", "sma_3.period"]
    assert combo.values == {"sma.period": 20, "sma_2.period": 50, "sma_3.period": 200}


@pytest.mark.parametrize("entry, exit, message", [
    ("foo", None, "unknown name"),
    ("stop_pct(5)", None, "risk exit"),
    ("close", None, "not a condition"),
    ("atr(14)", None, "not a condition"),
    ("rsi(14) <", None, "expected"),
    ("elliott.wave > 0", None, "hindsight"),
    ("sma(1, 2) > close", None, "at most 1"),
    ("sma(length=3) > close", None, "no parameter"),
    ("vote(rsi_threshold:1, macd_cross)", None, "no weights"),
    ("entry", None, "only be used in the exit"),
    ("macd_cross", "stop_pct(5) | stop_pct(3)", "twice"),
    ("vote", None, "needs members"),
    ("rsi_threshold & ", None, "expected"),
    ("close > 1 @", None, "can't read"),
])
def test_bad_rules_say_what_is_wrong(entry, exit, message):
    with pytest.raises(ValueError, match=message):
        check(entry, exit)


# -- risk exits ------------------------------------------------------------------------------------------
def test_stop_loss_and_take_profit_on_constructed_series():
    fixed("always", [1] * 8)
    close = [100, 100, 97, 94, 96, 99, 100, 100]
    # in at bar 0 (100); bar 3 closes 6 % below -> stop; flat that bar, then the entry (still true) re-enters
    assert positions("always", "stop_pct(5)", close) == [1, 1, 1, 0, 1, 1, 1, 1]
    up = [100, 103, 106, 111, 100, 100, 100, 100]
    assert positions("always", "take_profit(10)", up) == [1, 1, 1, 0, 1, 1, 1, 1]


def test_trailing_stop_and_max_hold():
    fixed("always", [1] * 8)
    close = [100, 110, 120, 115, 113, 120, 120, 120]
    # peak 120 at bar 2; 5 % trail -> 114: bar 4 (113) exits
    assert positions("always", "trail_pct(5)", close) == [1, 1, 1, 1, 0, 1, 1, 1]
    assert positions("always", "max_hold(3)", [100.0] * 8) == [1, 1, 1, 0, 1, 1, 1, 0]


def test_atr_stop_uses_the_atr_at_the_entry():
    from tradebot.indicators.builtin import ATR

    df = make_ohlc(300, seed=4)
    combo = build_algorithm("combo", entry="close > sma(20)", exit="stop_atr(2, 14)")
    pos = combo.run(df).to_numpy()
    close = df.close.to_numpy()
    atr = np.asarray(ATR.compute(OHLC.from_frame(df), period=14)["value"])
    entries = np.flatnonzero(np.diff(np.r_[0, pos]) > 0)
    exits = np.flatnonzero(np.diff(np.r_[pos, 0]) < 0) + 1  # first flat bar after a trade
    assert len(entries) > 2
    for e, x in zip(entries, exits):
        if x < len(close):
            assert close[x] <= close[e] - 2 * atr[e]  # the stop fired
            assert (close[e + 1:x] > close[e] - 2 * atr[e]).all()  # and not earlier


def test_risk_exits_combine_with_the_exit_rule(ohlc):
    plain = build_algorithm("combo", entry="rsi_threshold", exit="rsi_threshold").run(ohlc)
    with_stop = build_algorithm("combo", entry="rsi_threshold", exit="rsi_threshold | stop_pct(2)").run(ohlc)
    assert with_stop.sum() < plain.sum() and (with_stop <= plain + (with_stop > plain)).all()
    loop = build_algorithm("combo", entry="rsi_threshold", exit="rsi_threshold | max_hold(100000)").run(ohlc)
    np.testing.assert_array_equal(loop, plain)  # the risk loop without a risk exit firing equals the latch


# -- honesty ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("entry, exit", [
    ("elliott_wave & close > sma(200)", "macd_cross | stop_atr(2)"),
    ("weighted(rsi_threshold:0.5, macd_cross:0.3, sma_cross:0.2) >= 0.6", None),
    ("rsi(14) < 35 & !(close < sma(50))", "rsi(14) > 60 | trail_pct(4) | max_hold(30) | take_profit_atr(3)"),
    ("vote(macd_cross, sma_cross, ichimoku_cloud) >= 0.6 & elliott.state == 3", "!entry | stop_pct(5)"),
])
def test_no_look_ahead_on_truncated_data(entry, exit):
    df = make_ohlc(700, seed=3)
    combo = build_algorithm("combo", entry=entry, exit=exit)
    full = combo.run(df).to_numpy()
    for cut in (350, 500, 650):
        np.testing.assert_array_equal(combo.run(df.iloc[:cut]).to_numpy(), full[:cut])


def test_batched_parameters_match_one_by_one(ohlc):
    combo = build_algorithm("combo", entry="weighted(rsi_threshold:0.5, sma_cross:0.5) >= 0.5 & close > sma(30)",
                            exit="macd_cross | trail_pct(5)")
    o = OHLC.from_frame(ohlc)
    weights = np.array([[0.1], [0.5], [0.9]])
    trail = np.array([[2.0], [5.0], [9.0]])
    batched = combo.signals(o, {**combo.values, "w.rsi_threshold": weights, "trail_pct.pct": trail})
    assert batched.shape == (3, len(ohlc))
    for i in range(3):
        one = combo.signals(o, {**combo.values, "w.rsi_threshold": float(weights[i, 0]), "trail_pct.pct": float(trail[i, 0])})
        np.testing.assert_array_equal(batched[i], one)


@pytest.mark.skipif(ops.torch is None, reason="torch not installed")
def test_torch_matches_numpy(ohlc):
    combo = build_algorithm("combo", entry="vote(rsi_threshold, sma_cross) >= 0.5 & close > sma(30)",
                            exit="macd_cross | stop_pct(3)")
    o = OHLC.from_frame(ohlc)
    want = combo.signals(o, combo.values)
    got = combo.signals(OHLC.from_frame(ohlc, device="cpu"), combo.values)
    assert ops.is_torch(got)
    np.testing.assert_array_equal(got.numpy(), want)


# -- optimizer -------------------------------------------------------------------------------------------
def test_weights_thresholds_and_members_are_optimizable(ohlc):
    from tradebot.optimize import GeneticOptimizer

    combo = build_algorithm("combo", entry="weighted(rsi_threshold:0.5, macd_cross:0.3, sma_cross:0.2) >= 0.6",
                            exit="macd_cross | trail_pct(8)")
    space = combo.param_space()
    for key in ("w.rsi_threshold", "w.macd_cross", "w.sma_cross", "threshold", "rsi_threshold.lower",
                "macd_cross.fast", "trail_pct.pct"):
        assert key in space and space[key].description
    assert combo.values["threshold"] == 0.6 and combo.values["w.rsi_threshold"] == 0.5
    res = GeneticOptimizer(combo, ohlc, population=24, generations=2, seed=3, backend="numpy").run()
    assert set(res.best_params) == set(space)
    assert res.train.metrics is not None
    tuned = combo.with_params(**res.best_params)
    assert tuned.run(ohlc).isin([0.0, 1.0]).all()


def test_loop_based_members_make_it_loop_based():
    assert build_algorithm("combo", entry="macd_cross").vectorized
    assert not build_algorithm("combo", entry="elliott_wave & close > sma(200)").vectorized
    assert not build_algorithm("combo", entry="elliott").vectorized  # the elliott pool rule


# -- the builder helpers ---------------------------------------------------------------------------------
def test_terms_and_combine():
    assert terms("a & b | c") == (["a", "b", "c"], [])
    assert terms("weighted(rsi_threshold:0.5, macd_cross:0.3) >= 0.6") == (["rsi_threshold", "macd_cross"], [])
    assert terms("x | stop_pct(5) | max_hold(3)") == (["x"], ["stop_pct(5)", "max_hold(3)"])
    assert terms("rsi(14) < 30 & !macd_cross") == (["rsi(14) < 30", "!macd_cross"], [])
    assert combine(["rsi(14) < 30", "a | b"], "and", ["stop_pct(5)"]) == "rsi(14) < 30 & (a | b) | stop_pct(5)"
    assert combine(["a", "b"], "vote", threshold=0.6) == "vote(a, b) >= 0.6"
    assert combine(["a", "b"], "weighted") == "weighted(a:1, b:1) >= 0.5"
    assert combine([], "or", ["stop_pct(5)"]) == "stop_pct(5)"


# -- saved combos ----------------------------------------------------------------------------------------
def test_save_register_nest_and_delete(ohlc):
    save_combo("trend_dip", "rsi_threshold & close > sma(200)", "rsi_threshold | stop_pct(8)",
               params={"sma.period": 150, "nonsense": 1})
    assert combos_path().is_file()
    saved = load_saved()
    assert saved["trend_dip"]["entry"] == "rsi_threshold & close > sma(200)"
    assert saved["trend_dip"]["params"] == {"sma.period": 150}
    algo = build_algorithm("trend_dip")
    assert algo.values["sma.period"] == 150 and "trend_dip" in saved_combos()
    # a saved combo is a strategy: a member of another combo, of a vote, and of the pool
    outer = build_algorithm("combo", entry="trend_dip & macd_cross")
    assert "trend_dip.sma.period" in outer.param_space()
    build_algorithm("vote", members=["trend_dip", "macd_cross"]).run(ohlc)
    from tradebot.algorithms.pool import default_pool

    assert "trend_dip" in default_pool()[0] and "combo" not in default_pool()[0]
    # reload from disk, as at startup
    ALGORITHMS.unregister("trend_dip")
    assert load_combos() == ["trend_dip"]
    np.testing.assert_array_equal(build_algorithm("trend_dip").run(ohlc), algo.run(ohlc))
    assert delete_combo("trend_dip") and "trend_dip" not in ALGORITHMS and "trend_dip" not in load_saved()


def test_save_rejects_bad_names_rules_and_cycles():
    with pytest.raises(ValueError, match="name"):
        save_combo("Bad Name", "macd_cross")
    with pytest.raises(ValueError, match="taken"):
        save_combo("macd_cross", "rsi_threshold")
    with pytest.raises(ValueError, match="taken"):
        save_combo("sma", "rsi_threshold")
    with pytest.raises(ValueError, match="unknown name"):
        save_combo("broken", "nope")
    save_combo("inner", "macd_cross")
    save_combo("outer", "inner & close > sma(50)")
    with pytest.raises(ValueError, match="contains itself"):
        save_combo("inner", "outer")  # inner -> outer -> inner
    assert build_algorithm("inner").entry == "macd_cross"  # the old one stays
    assert load_saved()["inner"]["entry"] == "macd_cross"


def test_broken_saved_combos_are_skipped(caplog):
    combos_path().parent.mkdir(parents=True, exist_ok=True)
    combos_path().write_text('[good]\nentry = "macd_cross"\n\n[bad]\nentry = "no_such_strategy"\n\n'
                             '[macd_cross]\nentry = "rsi_threshold"\n')
    assert load_combos() == ["good"]
    assert "bad" not in ALGORITHMS and ALGORITHMS.get("macd_cross") is not ALGORITHMS.get("good")
    assert "bad" in caplog.text


def test_toml_round_trip_keeps_quotes_and_unicode():
    entry = 'close > sma(200) & rsi(14) < 30'
    save_combo("quoted", entry, description='says "hi" · ü')
    assert load_saved()["quoted"]["description"] == 'says "hi" · ü'
    assert issubclass(ALGORITHMS.get("quoted"), Combo)


# -- CLI -------------------------------------------------------------------------------------------------
def _cli(monkeypatch, *args):
    from typer.testing import CliRunner

    import tradebot.cli.main as cli

    monkeypatch.setattr(cli, "load_data", lambda *a, **k: make_ohlc(600, seed=5))
    return CliRunner().invoke(cli.app, list(args), env={"COLUMNS": "200"})


def test_cli_backtest_optimize_and_saved_combos(monkeypatch):
    res = _cli(monkeypatch, "backtest", "AAPL", "combo", "--entry", "rsi_threshold & close>sma(200)",
               "--exit", "macd_cross | stop_atr(2)", "--no-plot")
    assert res.exit_code == 0, res.output
    assert "combo(entry: rsi_threshold & close>sma(200)" in res.output and "total return" in res.output
    bad = _cli(monkeypatch, "backtest", "AAPL", "combo", "--entry", "nope", "--no-plot")
    assert bad.exit_code != 0 and "unknown name" in bad.output
    assert _cli(monkeypatch, "backtest", "AAPL", "macd_cross", "--entry", "x", "--no-plot").exit_code != 0
    opt = _cli(monkeypatch, "optimize", "AAPL", "combo", "--entry", "weighted(rsi_threshold:0.5, sma_cross:0.5) >= 0.6",
               "--exit", "trail_pct(5)", "--population", "12", "--generations", "2", "--backend", "numpy")
    assert opt.exit_code == 0, opt.output
    assert "w.rsi_threshold" in opt.output
    assert "--entry 'weighted(rsi_threshold:0.5, sma_cross:0.5) >= 0.6' --exit 'trail_pct(5)'" in opt.output

    saved = _cli(monkeypatch, "combo", "save", "trend_dip", "--entry", "rsi_threshold & close>sma(200)",
                 "--exit", "rsi_threshold | stop_pct(8)", "-p", "sma.period=150")
    assert saved.exit_code == 0, saved.output
    assert "sma.period=150" in _cli(monkeypatch, "combo", "list").output
    assert "trend_dip" in _cli(monkeypatch, "list").output  # loaded at startup like a plugin
    res = _cli(monkeypatch, "backtest", "AAPL", "trend_dip", "--no-plot")
    assert res.exit_code == 0 and "trend_dip(entry: rsi_threshold & close>sma(200)" in res.output
    assert _cli(monkeypatch, "combo", "delete", "trend_dip").exit_code == 0
    assert _cli(monkeypatch, "combo", "delete", "trend_dip").exit_code != 0
