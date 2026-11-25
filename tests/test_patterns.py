# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Candlestick patterns, chart patterns and support/resistance (#164), on hand-built candles."""

import numpy as np
import pandas as pd
import pytest
from .conftest import make_ohlc

from tradebot.algorithms.combo import check
from tradebot.analysis import patterns as pt
from tradebot.core.series import OHLC
from tradebot.indicators import INDICATORS
from tradebot.scanner import scan

NAMES = ("candles", "chart_patterns", "support_resistance", "volume_levels")


def bars(rows):
    """Candles from (open, high, low, close) rows."""
    a = np.array(rows, float).reshape(-1, 4)
    return a[:, 0], a[:, 1], a[:, 2], a[:, 3]


def fall(n=8, start=100.0, step=1.0):
    """A steady fall of ``n`` small red candles, the context a hammer needs."""
    return [(start - i * step, start - i * step + 0.3, start - (i + 1) * step - 0.3, start - (i + 1) * step)
            for i in range(n)]


def rise(n=8, start=100.0, step=1.0):
    return [(start + i * step, start + (i + 1) * step + 0.3, start + i * step - 0.3, start + (i + 1) * step)
            for i in range(n)]


def hits(rows, name, **kw):
    return np.flatnonzero(pt.candlestick(*bars(rows), **kw)[name])


def path(points, seg=10, pad=0.3):
    """``(close, high, low)`` where the close runs straight between the waypoints, ``seg`` candles per leg."""
    legs = [np.linspace(a, b, seg, endpoint=False) for a, b in zip(points, points[1:])]
    close = np.concatenate([*legs, [points[-1]]])
    return close, close + pad, close - pad


# -- candlestick patterns -------------------------------------------------------------------------------
def test_doji():
    rows = [(100, 101, 99, 100.05), (100, 101, 99, 100.9)]
    assert list(hits(rows, "doji")) == [0]


def test_flat_candle_is_no_doji():
    assert hits([(100, 100, 100, 100)] * 3, "doji").size == 0


def test_hammer_needs_a_prior_fall():
    hammer = (92, 92.4, 88, 92.3)  # long lower shadow, tiny body at the top
    assert list(hits([*fall(), hammer], "hammer")) == [8]
    assert hits([*rise(), (108, 108.4, 104, 108.3)], "hammer").size == 0


def test_shooting_star_needs_a_prior_rise():
    star = (108, 112, 107.9, 108.3)
    assert list(hits([*rise(), star], "shooting_star")) == [8]
    assert hits([*fall(), (92, 96, 91.9, 92.3)], "shooting_star").size == 0


def test_engulfing():
    bull = [(10, 10.2, 9.0, 9.2), (9.0, 10.6, 8.9, 10.5)]
    bear = [(9, 10.2, 8.9, 10.0), (10.2, 10.3, 8.5, 8.8)]
    assert list(hits(bull, "bull_engulfing")) == [1] and hits(bull, "bear_engulfing").size == 0
    assert list(hits(bear, "bear_engulfing")) == [1] and hits(bear, "bull_engulfing").size == 0
    inside = [(10, 10.2, 9.0, 9.2), (9.3, 9.9, 9.2, 9.8)]  # a harami, not an engulfing
    assert hits(inside, "bull_engulfing").size == 0
    same_colour = [(9.2, 10.2, 9.0, 10.0), (10.0, 11.5, 9.9, 11.4)]
    assert hits(same_colour, "bull_engulfing").size == 0


def test_morning_and_evening_star():
    morning = [(20, 20.1, 14.9, 15), (14.5, 14.8, 14.2, 14.6), (14.8, 18.5, 14.7, 18.4)]
    evening = [(15, 20.1, 14.9, 20), (20.4, 20.8, 20.2, 20.5), (20.3, 20.4, 16.0, 16.2)]
    assert list(hits(morning, "morning_star")) == [2] and hits(morning, "evening_star").size == 0
    assert list(hits(evening, "evening_star")) == [2] and hits(evening, "morning_star").size == 0
    weak = [morning[0], morning[1], (14.8, 16.0, 14.7, 15.9)]  # closes below the middle of the first candle
    assert hits(weak, "morning_star").size == 0
    big_middle = [morning[0], (16.0, 18.0, 14.0, 17.0), morning[2]]
    assert hits(big_middle, "morning_star").size == 0


def test_candlestick_empty_and_short():
    for rows in ([], [(1, 2, 0.5, 1.5)]):
        out = pt.candlestick(*bars(rows))
        assert all(len(v) == len(rows) for v in out.values())
        assert not any(out[k].any() for k in pt.CANDLE_PATTERNS if k != "doji")


# -- swing points ---------------------------------------------------------------------------------------
def test_swing_points():
    _, high, low = path([10, 20, 10, 20], seg=10)
    got = [(p.index, p.high) for p in pt.swing_points(high, low, 3)]
    assert got == [(10, True), (20, False)]
    assert pt.swing_points([], [], 3) == []
    assert pt.swing_points([1, 2], [1, 2], 3) == []


# -- chart patterns ---------------------------------------------------------------------------------------
def found(points, name, strength=3, seg=10, **kw):
    c, h, lo = path(points, seg)
    return np.flatnonzero(pt.chart_patterns(h, lo, c, strength, **kw)[name])


def test_double_top_and_bottom_fire_when_the_second_swing_is_confirmed():
    top = [100, 120, 100, 120, 100]
    assert list(found(top, "double_top")) == [30 + 3]
    assert found(top, "double_bottom").size == 0
    assert list(found([120, 100, 120, 100, 120], "double_bottom")) == [30 + 3]


def test_double_top_rejected_when_tops_differ_or_dip_is_shallow():
    assert found([100, 120, 100, 135, 100], "double_top").size == 0
    assert found([100, 120, 118, 120, 100], "double_top").size == 0


def test_head_and_shoulders_and_inverse():
    hs = [100, 115, 105, 125, 105, 115, 100]
    assert list(found(hs, "head_shoulders")) == [50 + 3]
    assert list(found([240 - p for p in hs], "inverse_head_shoulders")) == [50 + 3]
    assert found([100, 115, 105, 116, 105, 115, 100], "head_shoulders").size == 0  # no head
    assert found(hs, "inverse_head_shoulders").size == 0


def test_triangles():
    asc = [105, 100, 120, 105, 120, 110, 119]
    desc = [220 - p for p in asc]
    sym = [105, 100, 130, 105, 125, 110, 120, 113]
    assert found(asc, "ascending_triangle").size and not found(asc, "descending_triangle").size
    assert found(desc, "descending_triangle").size and not found(desc, "ascending_triangle").size
    assert found(sym, "symmetrical_triangle").size and not found(sym, "ascending_triangle").size
    widening = [105, 100, 120, 95, 125, 90, 130]  # broadening, not a triangle
    for name in ("ascending_triangle", "descending_triangle", "symmetrical_triangle"):
        assert found(widening, name).size == 0


def test_flags():
    c = np.r_[np.full(10, 100.0), np.linspace(100, 120, 11)[1:], 119, 118, 118.5, 117.5, 118, 117, 118, 117.5]
    h, lo = c + 0.3, c - 0.3
    out = pt.chart_patterns(h, lo, c, 3)
    assert out["bull_flag"].sum() == 1 and not out["bear_flag"].any()
    m = pt.chart_patterns(240 - lo, 240 - h, 240 - c, 3)  # mirrored: a pole down
    assert m["bear_flag"].sum() == 1 and not m["bull_flag"].any()
    flat = np.full(40, 100.0)
    assert not any(v.any() for v in pt.chart_patterns(flat + 0.3, flat - 0.3, flat, 3).values())
    deep = c.copy()
    deep[-8:] = 105  # gives back most of the pole
    assert not pt.chart_patterns(deep + 0.3, deep - 0.3, deep, 3)["bull_flag"].any()


def test_chart_patterns_empty_and_short():
    for n in (0, 1, 5):
        z = np.full(n, 1.0)
        out = pt.chart_patterns(z, z, z)
        assert set(out) == set(pt.CHART_PATTERNS) and not any(v.any() for v in out.values())


# -- support and resistance --------------------------------------------------------------------------------
BOUNCES = [105, 100, 120, 100, 120, 100, 120, 110]


def test_support_resistance_from_repeated_touches():
    c, h, lo = path(BOUNCES)
    sup, res = pt.support_resistance(h, lo, c, strength=3)
    assert sup[-1] == pytest.approx(99.7) and res[-1] == pytest.approx(120.3)
    assert np.isnan(sup[:20]).all() and np.isnan(res[:20]).all()  # nothing known yet


def test_single_touch_needs_volume():
    c, h, lo = path([105, 100, 120, 110])
    _, res = pt.support_resistance(h, lo, c, strength=3)
    assert np.isnan(res).all()
    vol = np.full(len(c), 100.0)
    vol[10] = vol[20] = 1000  # the candles of the low and the high
    _, res = pt.support_resistance(h, lo, c, vol, strength=3)
    assert res[-1] == pytest.approx(120.3)
    _, res = pt.support_resistance(h, lo, c, np.full(len(c), np.nan), strength=3)  # no volume data
    assert np.isnan(res).all()


def test_support_resistance_empty_and_old_levels_expire():
    sup, res = pt.support_resistance([], [], [])
    assert sup.size == 0 and res.size == 0
    c, h, lo = path(BOUNCES)
    _, res = pt.support_resistance(h, lo, c, strength=3, lookback=15)
    assert np.isnan(res[-1])


# -- indicators --------------------------------------------------------------------------------------------
def frame(close, high, low, open_=None):
    idx = pd.date_range("2024-01-01", periods=len(close), freq="D", tz="UTC")
    return pd.DataFrame({"open": close if open_ is None else open_, "high": high, "low": low, "close": close,
                         "volume": 1.0}, index=idx)


def test_registered_with_alerts_and_labels():
    for name in NAMES:
        cls = INDICATORS.get(name)
        assert cls.alerts and set(cls.alerts) <= set(cls.outputs)
    assert INDICATORS.get("candles").labels["pattern"][2] == "\u25b2 Eng"
    assert list(INDICATORS.get("candles").alerts)[:2] == ["bullish", "bearish"]


def test_candles_indicator_marks_and_anchors():
    o, h, lo, c = bars([*fall(), (92, 92.4, 88, 92.3), (92.3, 93, 91.5, 92.5)])
    out = INDICATORS.get("candles")()(frame(c, h, lo, o))
    assert out["candles_pattern"].iloc[8] == 1 and out["candles_hammer"].iloc[8] == 1
    assert out["candles_bullish"].iloc[8] == 1 and out["candles_bearish"].iloc[8] == 0
    assert out["candles_anchor"].iloc[8] == 88  # bullish markers sit under the low
    assert np.isnan(out["candles_pattern"].iloc[9]) and np.isnan(out["candles_hammer"].iloc[0])


def test_chart_pattern_indicator():
    c, h, lo = path([100, 120, 100, 120, 100])
    out = INDICATORS.get("chart_patterns")(strength=3)(frame(c, h, lo))
    assert out["chart_patterns_double_top"].iloc[33] == 1 and out["chart_patterns_double_top"].sum() == 1
    assert out["chart_patterns_bearish"].iloc[33] == 1 and out["chart_patterns_pattern"].iloc[33] == -1
    assert out["chart_patterns_bullish"].dropna().sum() == 0


def test_support_resistance_indicator_breakouts():
    c, h, lo = path([*BOUNCES, 125])
    out = INDICATORS.get("support_resistance")(strength=3)(frame(c, h, lo))
    assert out["support_resistance_breakout_up"].sum() == 1
    assert out["support_resistance_near_support"].dropna().isin([0, 1]).all()


def test_values_never_depend_on_later_candles():
    df = make_ohlc(300, seed=3)
    for name, kw in (("candles", {}), ("chart_patterns", {"strength": 3}), ("support_resistance", {"strength": 3})):
        ind = INDICATORS.get(name)(**kw)
        full = ind(df)
        for k in (120, 200):
            pd.testing.assert_frame_equal(ind(df.iloc[:k]), full.iloc[:k], check_exact=False)


def test_tiny_input():
    for name in NAMES:
        for n in (0, 1, 3):
            assert len(INDICATORS.get(name)()(make_ohlc(5).iloc[:n])) == n


def test_compute_on_ohlc_defaults():
    o = OHLC.from_frame(make_ohlc(60))
    for name in NAMES:
        cls = INDICATORS.get(name)
        res = cls.compute(o, **{k: p.default for k, p in cls.params.items()})
        assert set(res) == set(cls.outputs) and all(len(v) == 60 for v in res.values())


# -- scanner and combinator ---------------------------------------------------------------------------------
def test_scanner_filters_on_pattern_alerts():
    c, h, lo = path([100, 120, 100, 120, 100])
    full = frame(c, h, lo)
    data = {"TOP": full.iloc[:34], "OTHER": full.iloc[:25]}
    r = scan(["TOP", "OTHER"], "chart_patterns", {"strength": 3}, fetch=lambda s, t: data[s],
             filters=[{"alert": "double_top", "state": True}])
    assert list(r.table.symbol) == ["TOP"]


def test_combinator_conditions():
    o = OHLC.from_frame(make_ohlc(200, seed=5))
    for rule in ("candles.bull_engulfing", "chart_patterns.double_bottom | candles.hammer",
                 "support_resistance.breakout_up & close > sma(20)", "volume_levels.breakout_up & rsi(14) > 50", "chart_patterns.bullish & !candles.doji"):
        combo = check(rule)
        sig = combo.signals(o, {k: p.default for k, p in combo.param_space().items()})
        assert np.asarray(sig).reshape(-1).size == 200


# -- robustness and speed ------------------------------------------------------------------------------------
def _check_full(df):
    for name in NAMES:
        out = INDICATORS.get(name)()(df)
        assert len(out) == len(df)
        assert not np.isinf(out.to_numpy(dtype=float)).any(), name


def test_robust_to_odd_data():
    base = make_ohlc(120, seed=2)
    gaps = base.copy()
    gaps.iloc[30:35, :4] = np.nan
    gaps.iloc[60, 3] = np.nan
    flat = base.copy()
    flat[["open", "high", "low", "close"]] = 100.0
    same_vol = base.assign(volume=5.0)
    no_vol = base.drop(columns="volume")
    zero_vol = base.assign(volume=0.0)
    dup = base.iloc[list(range(60)) + list(range(55, 120))]
    for df in (base, gaps, flat, same_vol, no_vol, zero_vol, dup, base.iloc[:2], base.iloc[:0]):
        _check_full(df)


def test_scales_linearly_enough():
    import time
    df = make_ohlc(20000, seed=4, freq="h")
    for name in NAMES:
        t = time.perf_counter()
        INDICATORS.get(name)()(df)
        assert time.perf_counter() - t < 15, name  # generous: ~1 s normally; catches a quadratic blow-up


# -- part 2: more candles, wedges, pennants, cup and handle, volume levels (#164) ---------------------------
def test_three_white_soldiers_and_black_crows():
    soldiers = [(100, 106.2, 99.8, 106), (104, 112.2, 103.8, 112), (109, 119.2, 108.8, 119)]
    assert list(hits(soldiers, "three_white_soldiers")) == [2]
    crows = [(119, 119.2, 108.8, 109), (115, 115.2, 106.8, 107), (111, 111.2, 101.8, 102)]
    assert list(hits(crows, "three_black_crows")) == [2]
    # an upper shadow as long as the body is not a soldier
    weak = [(100, 106.2, 99.8, 106), (104, 112.2, 103.8, 112), (109, 124, 108.8, 119)]
    assert not hits(weak, "three_white_soldiers").size


def test_harami_needs_the_trend_and_a_small_inside_body():
    rows = fall(8) + [(91, 91.2, 83.8, 84), (86, 88.5, 85, 87)]  # long red, then a small green inside
    assert list(hits(rows, "bull_harami")) == [9]
    assert not hits(rise(8) + [(108, 108.1, 106.9, 107), (107.3, 107.7, 107.2, 107.6)], "bull_harami").size  # no fall
    up = rise(8) + [(108, 115.2, 107.8, 115), (113, 114.5, 111, 112)]
    assert list(hits(up, "bear_harami")) == [9]


def test_piercing_and_dark_cloud():
    rows = fall(8) + [(91, 91.2, 83.8, 84), (82, 88.6, 81.5, 88)]  # opens under the low, closes past the middle
    assert list(hits(rows, "piercing")) == [9]
    assert not hits(fall(8) + [(91, 91.2, 83.8, 84), (82, 92.5, 81.5, 92)], "piercing").size  # closes above the open
    up = rise(8) + [(108, 115.2, 107.8, 115), (117, 117.5, 110.6, 111)]
    assert list(hits(up, "dark_cloud")) == [9]


def test_tweezer_bottom_and_top():
    rows = fall(8) + [(91, 91.2, 83.8, 84), (84.5, 90, 83.8, 89)]
    assert list(hits(rows, "tweezer_bottom")) == [9]
    up = rise(8) + [(108, 115.2, 107.8, 115), (114.5, 115.2, 109, 110)]
    assert list(hits(up, "tweezer_top")) == [9]


def test_new_candles_are_causal():
    df = make_ohlc(400, seed=11)
    a = pt.candlestick(df.open, df.high, df.low, df.close)
    b = pt.candlestick(df.open[:250], df.high[:250], df.low[:250], df.close[:250])
    for k in pt.CANDLE_PATTERNS:
        assert (a[k][:250] == b[k]).all(), k


def test_wedges():
    rising = [90, 110, 100, 116, 108, 121, 114, 120]  # both edges rise and converge
    assert found(rising, "rising_wedge").size
    assert not found(rising, "falling_wedge").size
    falling = [130, 110, 120, 103, 112, 98, 106]
    assert found(falling, "falling_wedge").size
    assert not found(falling, "rising_wedge").size
    # a wedge is marked once per formation, not at every further swing point
    assert len(found(rising + [118, 123, 117, 124], "rising_wedge")) == 1


def pole(up=True, n_flag=8, narrowing=False):
    base = np.linspace(100, 112, 11) if up else np.linspace(112, 100, 11)
    top = base[-1]
    flag = []
    for i in range(n_flag):
        swing = (1.5 * (0.5 if narrowing and i >= n_flag // 2 else 1)) * (1 if i % 2 else -1)
        flag.append(top + (-1 if up else 1) * 1.0 + swing * (0.4 if narrowing and i >= n_flag // 2 else 1) * 0.5)
    close = np.concatenate([base, flag])
    hi, lo = close + 0.4, close - 0.4
    if narrowing:  # the second half of the consolidation has a much smaller range
        half = n_flag // 2
        hi[-half:] = close[-half:] + 0.1
        lo[-half:] = close[-half:] - 0.1
    return close, hi, lo


def test_pennant_is_a_flag_that_narrows():
    c, h, lo = pole(True, narrowing=True)
    out = pt.chart_patterns(h, lo, c, 3)
    assert out["bull_pennant"].any() and not out["bull_flag"].any()
    c, h, lo = pole(True, narrowing=False)
    out = pt.chart_patterns(h, lo, c, 3)
    assert out["bull_flag"].any() or not out["bull_pennant"].any()


def cup(handle=3.0, depth=15.0, v_shape=False):
    t = np.linspace(-1, 1, 40)
    bowl = (np.abs(t) if v_shape else t * t)
    cup_c = 100 - depth + depth * bowl
    run_in = np.linspace(80, 100, 15, endpoint=False)
    handle_c = 100 - np.sin(np.linspace(0, np.pi, 12)) * handle
    after = np.linspace(100, 108, 10)
    close = np.concatenate([run_in, cup_c, handle_c, after])
    return close, close + 0.3, close - 0.3


def test_cup_and_handle():
    c, h, lo = cup()
    got = np.flatnonzero(pt.chart_patterns(h, lo, c, 3)["cup_handle"])
    assert got.size == 1
    # fires when the handle's low is confirmed, i.e. at least `strength` candles after it
    handle_low = 15 + 40 + 6
    assert got[0] >= handle_low
    c, h, lo = cup(v_shape=True)
    assert not pt.chart_patterns(h, lo, c, 3)["cup_handle"].any()  # a V is no cup
    c, h, lo = cup(handle=9.0)
    assert not pt.chart_patterns(h, lo, c, 3)["cup_handle"].any()  # the handle gave back too much


def test_chart_patterns_prefix_equals_full_run():
    df = make_ohlc(500, seed=21)
    full = pt.chart_patterns(df.high, df.low, df.close, 4)
    part = pt.chart_patterns(df.high[:320], df.low[:320], df.close[:320], 4)
    for k in pt.CHART_PATTERNS:
        assert (full[k][:320] == part[k]).all(), k


def test_volume_levels_find_the_busy_price_and_are_causal():
    n = 300
    rng = np.random.default_rng(5)
    close = 100 + np.cumsum(rng.normal(0, 0.6, n))
    vol = np.full(n, 100.0)
    close[100:140] = 100 + rng.normal(0, 0.15, 40)  # a long stop at 100
    vol[100:140] = 900.0
    close[140:] += 10
    h, lo = close + 0.3, close - 0.3
    sup, res, poc = pt.volume_levels(h, lo, close, vol, lookback=250, bins=40)
    assert abs(poc[200] - 100) < 3  # the point of control is the price where most was traded
    assert abs(sup[200] - 100) < 3 or abs(res[200] - 100) < 3
    s2, r2, p2 = pt.volume_levels(h[:220], lo[:220], close[:220], vol[:220], lookback=250, bins=40)
    for a, b in ((sup, s2), (res, r2), (poc, p2)):
        assert np.allclose(a[:220], b, equal_nan=True)
    s3 = pt.volume_levels(h, lo, close, None)
    assert all(np.isnan(x).all() for x in s3)
    s4 = pt.volume_levels(h, lo, close, np.full(n, np.nan))
    assert all(np.isnan(x).all() for x in s4)


def test_new_indicator_outputs():
    ind = INDICATORS.get("volume_levels")
    df = make_ohlc(300, seed=3)
    out = ind()(df)
    assert set(c[len("volume_levels_"):] for c in out) == set(ind.outputs)
    assert ind.zones and ind.uses_volume
    assert "volume_levels_support" not in [c for c in out if ind().is_line(c)]  # drawn as a band, not a line
    assert set(INDICATORS.get("candles").outputs) >= set(pt.CANDLE_PATTERNS)
    assert set(INDICATORS.get("chart_patterns").outputs) >= set(pt.CHART_PATTERNS)


def test_labels_carry_a_triangle_and_help_for_every_code():
    from tradebot.indicators import patterns as ip

    for labels, helps, codes in ((ip.CANDLE_LABELS, ip.CANDLE_HELP, ip.CANDLE_CODES),
                                 (ip.CHART_LABELS, ip.CHART_HELP, ip.CHART_CODES)):
        assert set(labels) == set(helps) == set(codes.values())
        for code, text in labels.items():
            assert text[0] in "▲▼◆" and (text[0] == "▲") == (0 < code < 10)
            assert (text[0] == "▼") == (code < 0)
