# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Elliott wave detection, indicator, strategy and their freedom from look-ahead."""

import numpy as np
import pandas as pd
import pytest

from tradebot.algorithms import ALGORITHMS, build_algorithm
from tradebot.analysis import elliott as ew
from tradebot.backtest import backtest
from tradebot.core.registry import Param
from tradebot.core.series import OHLC
from tradebot.indicators import INDICATORS
from tradebot.indicators.elliott import ElliottWaves, detect

from .conftest import make_ohlc

# textbook impulse + correction: 0=100, 1=150, 2=119 (62 %), 3=200 (1.62 x w1), 4=169 (38 % of w3),
# 5=219 (= w1), A=190, B=205, C=160, then a new rise
WAYPOINTS = [120, 100, 150, 119, 200, 169, 219, 190, 205, 160, 230]
LEG = 30


def path(points=WAYPOINTS, leg=LEG, wiggle=0.0):
    c = [points[0]]
    for a, b in zip(points, points[1:]):
        c += list(np.linspace(a, b, leg + 1)[1:])
    c = np.array(c) + wiggle * np.sin(np.arange(len(c)) / 2)
    idx = pd.date_range("2024-01-01", periods=len(c), freq="D", tz="UTC")
    return pd.DataFrame({"open": np.r_[c[0], c[:-1]], "high": c + 0.3, "low": c - 0.3, "close": c}, index=idx)


def pivots_of(df, swing_atr=4.0):
    return detect(OHLC.from_frame(df), 14, swing_atr)[0]


# -- detection ---------------------------------------------------------------------------------------
def test_zigzag_finds_waypoints_and_confirms_later():
    piv = pivots_of(path())
    assert [p.index for p in piv] == [k * LEG for k in range(len(WAYPOINTS) - 1)]
    assert [round(p.price) for p in piv] == WAYPOINTS[:-1]
    assert all(p.confirmed > p.index for p in piv)  # never known at the extreme itself
    assert all(a.kind == -b.kind for a, b in zip(piv, piv[1:]))  # alternating highs and lows


def test_zigzag_sensitivity():
    noisy = path(wiggle=4.0)
    assert len(pivots_of(noisy, swing_atr=1.0)) > len(pivots_of(noisy, swing_atr=6.0))


def test_textbook_impulse_scores_high_and_bearish_mirror():
    assert ew.impulse_score([100, 150, 119, 200, 169, 219]) > 0.95
    assert ew.impulse_score([-p for p in (100, 150, 119, 200, 169, 219)], direction=-1) > 0.95
    assert ew.correction_score([219, 190, 205, 160]) > 0.8


@pytest.mark.parametrize("prices, rule", [
    ([100, 150, 95, 200, 169, 219], "wave 2 retraced beyond the start of wave 1"),
    ([100, 150, 119, 200, 140, 219], "wave 4 overlaps wave 1"),
    ([100, 150, 140, 160, 155, 260], "wave 3 is the shortest"),
    ([100, 150, 119, 140, 130, 160], "wave 3 does not exceed wave 1"),
])
def test_rule_violations_are_rejected(prices, rule):
    assert ew.impulse_score(prices) is None, rule


def test_invalid_correction_rejected():
    assert ew.correction_score([219, 190, 225, 160]) is None  # B beyond the impulse end
    assert ew.correction_score([219, 190, 205, 195]) is None  # C does not extend beyond A


def test_label_waves_textbook():
    labels = [(p.index // LEG, ew.LABELS[code]) for p, code in ew.label_waves(pivots_of(path()))]
    assert labels == [(1, "0"), (2, "1"), (3, "2"), (4, "3"), (5, "4"), (6, "5"), (7, "A"), (8, "B"), (9, "C")]


def test_wave_states_progression_is_causal():
    df = path()
    piv = pivots_of(df)
    ws = ew.wave_states(piv, len(df))
    changes = [int(ws.state[t]) for t in range(1, len(df)) if ws.state[t] != ws.state[t - 1]]
    assert changes == [3, 4, 5, 10, 11, 12, 13]
    by_confirm = {p.confirmed: p for p in piv}
    for t in range(1, len(df)):  # the state only changes on bars where a pivot gets confirmed
        if ws.state[t] != ws.state[t - 1]:
            assert t in by_confirm


# -- indicator ---------------------------------------------------------------------------------------
def test_indicator_outputs():
    assert "elliott" in INDICATORS
    df = path()
    ind = ElliottWaves()
    frame = ind(df)
    assert list(frame.columns) == ["elliott_zigzag", "elliott_wave", "elliott_state"]
    zz = frame["elliott_zigzag"]
    for p in pivots_of(df):
        assert zz.iloc[p.index] == pytest.approx(p.price)
    assert frame["elliott_wave"].dropna().map(ew.LABELS.get).tolist() == list("012345ABC")
    assert ind.is_line("elliott_zigzag") and not ind.is_line("elliott_wave") and not ind.is_line("elliott_state")


def test_indicator_state_has_no_lookahead():
    df = make_ohlc(800, seed=9)
    ind = ElliottWaves(swing_atr=3.0)
    full = ind(df)["elliott_state"].to_numpy()
    for k in range(150, 800, 73):
        assert np.array_equal(ind(df.iloc[:k])["elliott_state"].to_numpy(), full[:k])


# -- strategy ----------------------------------------------------------------------------------------
def trades(df, **params):
    res = backtest(df, build_algorithm("elliott_wave", params), fee=0.0)
    return [(round(df.close.loc[r.entry], 1), round(df.close.loc[r.exit], 1), r["return"]) for _, r in res.trades.iterrows()]


def test_strategy_trades_wave3_and_wave5_on_textbook():
    t = trades(path())
    assert len(t) == 2
    (e3, x3, r3), (e5, x5, r5) = t
    assert 150.3 < e3 < 155  # breakout above the wave-1 high
    assert x3 > 185 and r3 > 0.2  # out near the wave-3 top
    assert 200.3 < e5 < 205 and r5 > 0.05  # breakout above the wave-3 high, target ≈ wave 1


def test_strategy_parameters_shape_the_trades():
    assert len(trades(path(), trade_wave5=0)) == 1
    assert trades(path(), wave2_max=0.5)[0][0] > 200  # 62 % retrace outside the band -> only the wave-5 trade
    assert trades(path(), min_score=0.99) == []


def test_strategy_has_no_lookahead():
    df = make_ohlc(1200, seed=5)
    algo = build_algorithm("elliott_wave", {"swing_atr": 3.0})
    full = algo.run(df).to_numpy()
    for k in range(200, 1200, 97):
        assert np.array_equal(algo.run(df.iloc[:k]).to_numpy(), full[:k])


def test_strategy_is_registered_and_optimizable():
    from tradebot.optimize import GeneticOptimizer

    assert ALGORITHMS.get("elliott_wave").vectorized is False
    opt = GeneticOptimizer(build_algorithm("elliott_wave"), make_ohlc(600, seed=3), population=16, generations=2,
                           seed=0, min_trades=0)
    assert opt.backend.kind == "numpy"  # auto -> CPU for loop-based algorithms
    res = opt.run()
    assert set(res.best_params) == set(build_algorithm("elliott_wave").param_space())
    assert isinstance(res.best_params["trade_wave5"], int)


def test_param_integer_flag_is_keyword_only():
    assert Param(0.5, 0.0, 1.0, "a description").integer is False
    with pytest.raises(TypeError):
        Param(0.5, 0.0, 1.0, "d", True)


# -- rendering ---------------------------------------------------------------------------------------
def test_terminal_chart_draws_waves():
    from tradebot.cli.charts import render_chart

    out = render_chart(path(wiggle=1.0), "ew", [ElliottWaves()], build_algorithm("elliott_wave"), line=True,
                       width=140, height=30)
    assert "elliott_zigzag" in out and "elliott_state" not in out and "elliott_wave" not in out
