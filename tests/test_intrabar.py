# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Intrabar fills (#150): stops, take-profits (also in steps) and limit entries against the candle's high/low."""

import numpy as np
import pandas as pd
import pytest

from tradebot.algorithms import build_algorithm
from tradebot.algorithms.combo import check, load_combos, save_combo
from tradebot.backtest import backtest
from tradebot.backtest.engine import Fills, simulate
from tradebot.core import ops
from tradebot.core.series import OHLC
from tests.conftest import make_ohlc

ALWAYS = "close > 0"


@pytest.fixture(autouse=True)
def _cleanup():
    from tradebot.algorithms import ALGORITHMS

    before = dict(ALGORITHMS._items)
    yield
    ALGORITHMS._items.clear()
    ALGORITHMS._items.update(before)


def frame(rows) -> pd.DataFrame:
    """Candles from ``(open, high, low, close)`` rows or plain closes."""
    rows = [(r, r, r, r) if np.isscalar(r) else r for r in rows]
    idx = pd.date_range("2024-01-01", periods=len(rows), freq="D", tz="UTC")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx, dtype=float)


def run(rows, exit, entry=ALWAYS, fills="intrabar"):
    res = backtest(frame(rows), check(entry, exit, fills), fee=0.0)
    return res.positions.tolist(), res.returns.to_numpy(), res


FLAT = (100, 100, 100, 100)


def test_stop_fills_at_the_level_inside_the_candle():
    pos, ret, _ = run([FLAT, FLAT, (99, 99, 94, 98), FLAT], "stop_pct(5)")
    assert pos[:3] == [1, 1, 0]
    assert ret[2] == pytest.approx(95 / 100 - 1)  # at the stop, not the close (98)
    pos, ret, _ = run([FLAT, FLAT, (99, 99, 94, 98), FLAT], "stop_pct(5)", fills="close")
    assert pos[:3] == [1, 1, 1]  # the close (98) never reached the stop


def test_a_gap_below_the_stop_fills_at_the_open():
    _, ret, _ = run([FLAT, FLAT, (90, 92, 89, 91), FLAT], "stop_pct(5)")
    assert ret[2] == pytest.approx(90 / 100 - 1)


def test_take_profit_at_the_level_and_gap_up_at_the_open():
    pos, ret, _ = run([FLAT, FLAT, (101, 111, 100, 102), FLAT], "take_profit(10)")
    assert pos[2] == 0 and ret[2] == pytest.approx(0.10)
    _, ret, _ = run([FLAT, FLAT, (112, 113, 111, 112), FLAT], "take_profit(10)")
    assert ret[2] == pytest.approx(0.12)


def test_stop_and_take_profit_in_one_candle_takes_the_stop():
    _, ret, _ = run([FLAT, FLAT, (100, 111, 94, 105), FLAT], "stop_pct(5) | take_profit(10)")
    assert ret[2] == pytest.approx(-0.05)
    # unless it opens beyond the take-profit: that open is the first price of the candle
    _, ret, _ = run([FLAT, FLAT, (112, 113, 90, 95), FLAT], "stop_pct(5) | take_profit(10)")
    assert ret[2] == pytest.approx(0.12)
    # a half take-profit at the open, then the stop for the rest
    pos, ret, _ = run([FLAT, FLAT, (106, 107, 90, 95), FLAT], "stop_pct(5) | take_profit(5, 0.5) | take_profit(20)")
    assert pos[2] == 0 and ret[2] == pytest.approx(1.06 * (1 + 0.5 * (95 / 106 - 1)) - 1)


def test_trailing_stop_uses_the_highest_close_before_the_candle():
    rows = [FLAT, (100, 110, 100, 110), (110, 125, 109, 120), (119, 119, 113, 118), FLAT]
    pos, ret, _ = run(rows, "trail_pct(5)")
    # the peak before candle 3 is 120 (not candle 3's own high): stop at 114
    assert pos[:4] == [1, 1, 1, 0]
    assert ret[3] == pytest.approx(114 / 120 - 1)


def test_scale_out_in_steps():
    exit = "take_profit(5, 0.5) | take_profit(10)"
    pos, ret, _ = run([FLAT, FLAT, (100, 106, 100, 104), (104, 111, 103, 108), FLAT], exit)
    assert pos[:4] == [1, 1, 0.5, 0]
    assert ret[2] == pytest.approx(1.05 * (1 + 0.5 * (104 / 105 - 1)) - 1)
    assert ret[3] == pytest.approx(0.5 * (110 / 104 - 1))
    # both steps in one candle: half at 105, half at 110
    pos, ret, _ = run([FLAT, FLAT, (100, 112, 100, 108), FLAT], exit)
    assert pos[2] == 0 and ret[2] == pytest.approx(1.05 * (1 + 0.5 * (110 / 105 - 1)) - 1)
    # at the close: the first step sells half on a close above it
    pos, _, _ = run([100, 100, 106, 111, 100], exit, fills="close")
    assert pos[:4] == [1, 1, 0.5, 0]


def test_take_profit_size_is_a_parameter_only_when_given():
    assert "take_profit.size" not in check(ALWAYS, "take_profit(10)").param_space()
    space = check(ALWAYS, "take_profit(5, 0.5) | take_profit(10)").param_space()
    assert space["take_profit.size"].default == 0.5 and "take_profit_2.pct" in space
    with pytest.raises(ValueError, match="twice"):
        check(ALWAYS, "take_profit(10) | take_profit(10)")


def test_limit_entry_fills_below_the_close_or_at_a_gap():
    entry = ALWAYS + " & limit_pct(2)"
    pos, ret, _ = run([FLAT, (100, 101, 97.5, 101), FLAT], "", entry)
    assert pos[:2] == [0, 1]  # the order from candle 0 (limit 98) fills in candle 1
    assert ret[1] == pytest.approx(101 / 98 - 1)
    _, ret, _ = run([FLAT, (97, 101, 96, 101), FLAT], "", entry)
    assert ret[1] == pytest.approx(101 / 97 - 1)
    pos, _, _ = run([FLAT, (100, 101, 98.5, 101), FLAT], "", entry)
    assert pos[:2] == [0, 0]  # not low enough: no fill
    # at the close: filled only on a close at or below the limit
    pos, _, _ = run([100, 99, 97, 100], "", entry, fills="close")
    assert pos[:3] == [0, 0, 1]


def test_limit_entry_and_stop_in_one_candle_is_a_round_trip():
    pos, ret, res = run([FLAT, (99, 100, 90, 95), FLAT], "stop_pct(5)", ALWAYS + " & limit_pct(2)")
    assert pos[1] == 0
    assert ret[1] == pytest.approx(98 * 0.95 / 98 - 1)
    assert len(res.trades) >= 1 and res.trades.iloc[0]["entry"] == res.trades.iloc[0]["exit"]
    # the fee counts both sides
    fee = backtest(frame([FLAT, (99, 100, 90, 95), FLAT]), check(ALWAYS + " & limit_pct(2)", "stop_pct(5)",
                                                                 "intrabar"), fee=0.01).returns.iloc[1]
    assert fee == pytest.approx(ret[1] - 0.02)


def test_take_profit_on_the_limit_candle_needs_the_close_beyond_it():
    entry = ALWAYS + " & limit_pct(2)"
    pos, _, _ = run([FLAT, (100, 120, 97, 99), FLAT], "take_profit(5)", entry)
    assert pos[1] == 1  # the high may have come before the fill
    pos, ret, _ = run([FLAT, (100, 120, 97, 104), FLAT], "take_profit(5)", entry)
    assert pos[1] == 0 and ret[1] == pytest.approx(0.05)


def test_close_mode_is_unchanged_and_has_no_fills():
    df = make_ohlc(400, seed=5)
    exit = "rsi(14) > 60 | trail_pct(4) | stop_atr(2) | take_profit(8)"
    combo = check("rsi(14) < 40", exit)
    assert combo.fills == "close" and backtest(df, combo).fills is None
    intrabar = backtest(df, check("rsi(14) < 40", exit, "intrabar"))
    assert intrabar.fills is not None and intrabar.metrics["trades"] > 0


@pytest.mark.parametrize("entry, exit", [
    ("rsi(14) < 40", "rsi(14) > 60 | trail_atr(2) | stop_pct(3) | take_profit(5, 0.5) | take_profit_atr(3)"),
    ("close > sma(20) & limit_atr(0.5)", "trail_pct(4) | max_hold(15)"),
])
def test_no_look_ahead(entry, exit):
    df = make_ohlc(600, seed=7)
    combo = check(entry, exit, "intrabar")
    full = backtest(df, combo, fee=0.001)
    for cut in (250, 400, 550):
        part = backtest(df.iloc[:cut], combo, fee=0.001)
        np.testing.assert_array_equal(part.positions.to_numpy(), full.positions.to_numpy()[:cut])
        np.testing.assert_allclose(part.returns.to_numpy(), full.returns.to_numpy()[:cut], atol=1e-12)


def test_intrabar_returns_are_bounded_by_the_candles():
    """Every fill is inside the candle's range (or at its open), so a long can't beat high/prev close."""
    df = make_ohlc(500, seed=9)
    res = backtest(df, check("close > sma(10) & limit_pct(1)", "stop_pct(2) | take_profit(3, 0.5) | take_profit(6)",
                             "intrabar"), fee=0.0)
    prev = df.close.shift(1).bfill()
    best = np.maximum(df.high / prev, df.close / df.low) - 1
    assert (res.returns <= best + 1e-12).all()


def test_batched_parameters_match_one_by_one():
    df = make_ohlc(400, seed=2)
    combo = check("close > sma(20) & limit_pct(1)", "stop_pct(3) | take_profit(5, 0.5) | take_profit(9)", "intrabar")
    o = OHLC.from_frame(df)
    stop = np.array([[1.0], [3.0], [6.0]])
    pos, fills = combo.execute(o, {**combo.values, "stop_pct.pct": stop})
    assert pos.shape == (3, len(df)) and fills.adjust.shape == (3, len(df))
    for i in range(3):
        one_pos, one = combo.execute(o, {**combo.values, "stop_pct.pct": float(stop[i, 0])})
        np.testing.assert_array_equal(pos[i], one_pos)
        np.testing.assert_allclose(fills.adjust[i], one.adjust)
        np.testing.assert_allclose(simulate(o.close, pos[i], 0.001, Fills(fills.adjust[i], fills.turnover[i])),
                                   simulate(o.close, one_pos, 0.001, one))


@pytest.mark.skipif(ops.torch is None, reason="torch not installed")
def test_torch_matches_numpy():
    df = make_ohlc(300, seed=4)
    combo = check("close > sma(20)", "stop_pct(3) | take_profit(5)", "intrabar")
    want, wf = combo.execute(OHLC.from_frame(df), combo.values)
    got, gf = combo.execute(OHLC.from_frame(df, device="cpu"), combo.values)
    assert ops.is_torch(got) and ops.is_torch(gf.adjust)
    np.testing.assert_array_equal(got.numpy(), want)
    np.testing.assert_allclose(gf.adjust.numpy(), wf.adjust)


def test_optimizer_scores_the_intrabar_fills():
    from tradebot.optimize import GeneticOptimizer

    df = make_ohlc(400, seed=6)
    combo = check("close > sma(20)", "stop_pct(3) | take_profit(5)", "intrabar")
    opt = GeneticOptimizer(combo, df, backend="numpy", seed=1, train_fraction=1.0, min_trades=0, objective="return")
    fitness = opt.evaluate(opt.encode(combo.values)[None, :])
    want = backtest(df, combo, fee=opt.fee).metrics["total_return"]
    assert want != backtest(df, check("close > sma(20)", "stop_pct(3) | take_profit(5)"), fee=opt.fee).metrics[
        "total_return"]
    assert fitness[0] == pytest.approx(want, rel=1e-9, abs=1e-12)


def test_fills_are_saved_with_a_combo(tmp_path):
    path = tmp_path / "combos.toml"
    save_combo("ib_test", ALWAYS, "stop_pct(5)", path=path, fills="intrabar")
    assert 'fills = "intrabar"' in path.read_text()
    assert load_combos(path) == ["ib_test"]
    assert build_algorithm("ib_test").fills == "intrabar"
    with pytest.raises(ValueError, match="fills"):
        check(ALWAYS, "", "sometimes")
    with pytest.raises(ValueError, match="limit entry needs a condition"):
        check("limit_pct(1)")
    with pytest.raises(ValueError, match="entry rule"):
        check(ALWAYS, "limit_pct(1)")


def test_optimizer_random_windows_take_the_fills():
    from tradebot.optimize import GeneticOptimizer

    df = make_ohlc(600, seed=8)
    kw = dict(backend="numpy", seed=1, train_fraction=1.0, min_trades=0, samples=4, sample_length=120)
    close = GeneticOptimizer(check("close > sma(20)", "stop_pct(2) | take_profit(4)"), df, **kw)
    intrabar = GeneticOptimizer(check("close > sma(20)", "stop_pct(2) | take_profit(4)", "intrabar"), df, **kw)
    genome = close.encode(close.algorithm.values)[None, :]
    a, b = close.evaluate(genome), intrabar.evaluate(genome)
    assert np.isfinite(b).all() and not np.allclose(a, b)


def test_cli_fills(monkeypatch, tmp_path):
    from typer.testing import CliRunner

    import tradebot.cli.main as cli

    monkeypatch.setattr(cli, "load_data", lambda *a, **k: make_ohlc(400, seed=5))
    monkeypatch.setattr("tradebot.algorithms.combo.combos_path", lambda: tmp_path / "combos.toml")
    run = lambda *args: CliRunner().invoke(cli.app, list(args), env={"COLUMNS": "200"})  # noqa: E731
    res = run("backtest", "AAPL", "combo", "--entry", "rsi(14) < 40 & limit_pct(1)", "--exit",
              "take_profit(5, 0.5) | take_profit(10) | stop_pct(3)", "--fills", "intrabar", "--no-plot")
    assert res.exit_code == 0, res.output
    assert "fills: intrabar" in res.output
    assert run("backtest", "AAPL", "macd_cross", "--fills", "intrabar", "--no-plot").exit_code != 0
    opt = run("optimize", "AAPL", "combo", "--entry", "close > sma(20)", "--exit", "stop_pct(3)", "--fills",
              "intrabar", "--population", "8", "--generations", "1", "--backend", "numpy")
    assert opt.exit_code == 0, opt.output
    assert "--fills intrabar" in opt.output
    saved = run("combo", "save", "cli_ib", "--entry", ALWAYS, "--exit", "stop_pct(5)", "--fills", "intrabar")
    assert saved.exit_code == 0, saved.output
    assert "intrabar" in run("combo", "list").output
