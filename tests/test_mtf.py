# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Multi-timeframe indicators and strategies (#161): resampling, no look-ahead, combos, scanner, evolve."""

import numpy as np
import pandas as pd
import pytest

from tradebot.algorithms.combo import Combo, check, combine, terms, tokenize
from tradebot.core import mtf
from tradebot.core.series import OHLC
from tradebot.core.timeframes import resample
from tradebot.indicators import INDICATORS

from .conftest import make_ohlc


def hourly(n=24 * 40, seed=0):
    return make_ohlc(n, seed=seed, freq="h")


# -- core ---------------------------------------------------------------------------------------------
def test_daily_close_is_seen_from_the_candle_that_closes_the_day():
    df = hourly(24 * 5)
    x = mtf.Higher(OHLC.from_frame(df), "1d").map(lambda d: d.close)
    daily = resample(df, "1d").close.to_numpy()
    assert np.isnan(x[:23]).all()  # the first day isn't over before its 23:00 candle closes
    assert x[23] == daily[0] and (x[24:47] == daily[0]).all() and x[47] == daily[1]
    assert x[-1] == daily[-1]


def test_aggregation_matches_pandas_resample():
    df = hourly(24 * 20)
    h = mtf.Higher(OHLC.from_frame(df), "4h").rows[0][1]
    ref = resample(df, "4h")
    for name in ("open", "high", "low", "close", "volume"):
        np.testing.assert_allclose(getattr(h, name), ref[name].to_numpy())


def test_a_session_that_ends_early_is_used_from_the_next_candle():
    """Stock hours 9-16: the day bucket ends at midnight, so the day counts from the next morning's candle."""
    days = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
    idx = pd.DatetimeIndex([d + pd.Timedelta(hours=h) for d in days for h in range(9, 17)])
    df = pd.DataFrame({c: np.arange(len(idx), dtype=float) + 1 for c in ("open", "high", "low", "close")}, index=idx)
    x = mtf.Higher(OHLC.from_frame(df), "1d").map(lambda d: d.close)
    assert np.isnan(x[:8]).all() and x[8] == 8.0 and (x[8:16] == 8.0).all() and x[16] == 16.0


def test_weeks_start_on_monday_and_months_on_the_first():
    start, _ = mtf.buckets(np.array([pd.Timestamp("2024-01-10", tz="UTC").timestamp()]), "1w")  # a Wednesday
    assert pd.Timestamp(start[0], unit="s").day_name() == "Monday"
    start, end = mtf.buckets(np.array([pd.Timestamp("2024-02-15 13:00", tz="UTC").timestamp()]), "1mo")
    assert pd.Timestamp(start[0], unit="s") == pd.Timestamp("2024-02-01")
    assert pd.Timestamp(end[0], unit="s") == pd.Timestamp("2024-03-01")


def test_monthly_on_daily_candles():
    df = make_ohlc(120)
    x = mtf.Higher(OHLC.from_frame(df), "1mo").map(lambda d: d.close)
    jan_31 = df.index.get_loc(pd.Timestamp("2020-01-31", tz="UTC"))
    assert np.isnan(x[:jan_31]).all() and x[jan_31] == df.close.iloc[jan_31] and x[jan_31 + 1] == df.close.iloc[jan_31]


def test_bad_timeframes_and_missing_timestamps_say_so():
    o = OHLC.from_frame(make_ohlc(50))
    with pytest.raises(ValueError, match="shorter than the candles"):
        mtf.Higher(o, "4h")
    with pytest.raises(ValueError, match="unknown timeframe"):
        mtf.Higher(o, "3d")
    with pytest.raises(ValueError, match="timestamps"):
        mtf.Higher(OHLC(o.open, o.high, o.low, o.close), "1w")


@pytest.mark.parametrize("name,params", [("sma", {"period": 5}), ("rsi", {"period": 3}), ("macd", {}),
                                         ("regime", {"period": 3, "vol_period": 2})])
def test_no_look_ahead_on_truncated_data(name, params):
    """A value never changes when later candles arrive, including the candles inside a running higher candle."""
    df = hourly(24 * 30, seed=4)
    cls = INDICATORS.get(name)
    params = INDICATORS.get(name)(**params).values  # every parameter, defaults filled in
    full = mtf.compute(cls, OHLC.from_frame(df), "1d", **params)
    for cut in (24 * 12 + 5, 24 * 20 - 1, 24 * 25 + 13):
        part = mtf.compute(cls, OHLC.from_frame(df.iloc[:cut]), "1d", **params)
        for k in part:
            np.testing.assert_allclose(full[k][:cut], part[k], equal_nan=True)


def test_leading_dimensions_and_batched_parameters():
    """Optimizer windows ([K, 1, T] candles, each with its own times) with batched parameters ([N, 1])."""
    df = hourly(24 * 20)
    o = OHLC.from_frame(df)
    windows = [slice(0, 240), slice(100, 340), slice(200, 440)]
    stacked = OHLC(*(np.stack([f[s] for s in windows])[:, None, :] for f in o.fields()))
    periods = np.array([[2], [3]])
    sma = INDICATORS.get("sma")
    out = mtf.compute(sma, stacked, "1d", period=periods)["value"]
    assert out.shape == (3, 2, 240)
    for k, s in enumerate(windows):
        for j, p in enumerate((2, 3)):
            one = mtf.compute(sma, OHLC(*(f[s] for f in o.fields())), "1d", period=p)["value"]
            np.testing.assert_allclose(out[k, j], one, equal_nan=True)


def test_closed_drops_a_running_candle():
    df = make_ohlc(10)
    last = df.index[-1].timestamp()
    assert len(mtf.closed(df, "1d", last + 3600)) == 9
    assert len(mtf.closed(df, "1d", last + 86_400)) == 10


# -- indicators and combos ----------------------------------------------------------------------------
def test_indicator_with_tf_label_and_frame():
    df = hourly(24 * 10)
    ind = INDICATORS.get("sma")(tf="1d", period=3)
    assert ind.label == "sma(3,tf=1d)" and INDICATORS.get("sma")(period=3).label == "sma(3)"
    frame = ind(df)
    ref = mtf.compute(INDICATORS.get("sma"), OHLC.from_frame(df), "1d", period=3)["value"]
    np.testing.assert_allclose(frame["sma_value"].to_numpy(), ref, equal_nan=True)
    with pytest.raises(ValueError, match="unknown timeframe"):
        INDICATORS.get("sma")(tf="2d")


def test_combo_rules_with_timeframes():
    c = check("close > sma(5, tf=1d) & rsi(14, tf=4h) < 70 & regime(3, tf=1w).bull", "macd_cross(tf=1d)")
    assert {"sma_1d.period", "rsi_4h.period", "regime_1w.period", "macd_cross_1d.fast"} <= set(c.param_space())
    assert c.timeframes == {"1d", "4h", "1w"}
    labels = [i.label for i in c.chart_indicators()]
    assert "sma(5,tf=1d)" in labels and "macd(12,26,9,tf=1d)" in labels
    # the same call on the candles' own timeframe is a different member
    both = check("sma(5) > sma(5, tf=1d)")
    assert {"sma.period", "sma_1d.period"} <= set(both.param_space())


def test_combo_condition_equals_hand_computed():
    df = hourly(24 * 30, seed=2)
    o = OHLC.from_frame(df)
    c = Combo(entry="close > sma(5, tf=1d)", exit="close < sma(5, tf=1d)")
    pos = np.asarray(c.signals(o, c.values))
    daily = mtf.compute(INDICATORS.get("sma"), o, "1d", period=5)["value"]
    above, below = df.close.to_numpy() > daily, df.close.to_numpy() < daily
    first = int(np.argmax(above))
    assert pos[:first].sum() == 0 and pos[first] == 1
    assert ((pos == 1) == np.where(np.isnan(daily), False, above | (~below & (pos == 1))))[first:].all()


@pytest.mark.parametrize("entry,exit", [("close > sma(5, tf=1d)", "close < ema(3, tf=1d)"),
                                        ("rsi(tf=1d)", "macd_cross(tf=4h)"),
                                        ("regime(3, tf=1d).bull & close > sma(10)", "")])
def test_combo_no_look_ahead(entry, exit):
    df = hourly(24 * 30, seed=6)
    c = Combo(entry=entry, exit=exit)
    full = np.asarray(c.signals(OHLC.from_frame(df), c.values))
    for cut in (24 * 15 + 7, 24 * 22 + 23):
        part = np.asarray(c.signals(OHLC.from_frame(df.iloc[:cut]), c.values))
        np.testing.assert_array_equal(full[:cut], part)


def test_batched_combo_matches_one_by_one():
    df = hourly(24 * 30, seed=3)
    o = OHLC.from_frame(df)
    c = Combo(entry="close > sma(5, tf=1d)", exit="close < sma(5, tf=1d)")
    p = dict(c.values, **{"sma_1d.period": np.array([[2], [4], [6]])})
    batched = np.asarray(c.signals(o, p))
    for i, period in enumerate((2, 4, 6)):
        np.testing.assert_array_equal(batched[i], c.signals(o, dict(c.values, **{"sma_1d.period": period})))


@pytest.mark.parametrize("entry,message", [("close(tf=1d) > 1", "one-candle average"),
                                            ("rsi_threshold & limit_pct(1, tf=1d)", "takes no tf"),
                                            ("sma(5, tf=3x) > close", "timeframe"),
                                            ("sma(5, tf=2d) > close", "unknown timeframe")])
def test_bad_timeframes_in_rules(entry, message):
    with pytest.raises(ValueError, match=message):
        check(entry)


def test_tokens_and_builder_round_trip():
    assert tokenize("sma(200, tf=1mo)") == ["sma", "(", "200", ",", "tf", "=", "1mo", ")"]
    assert tokenize("1e3 > 2") == ["1e3", ">", "2"]
    rule = combine(terms("close > sma(200, tf=1d) & rsi(tf=4h)")[0], "and")
    assert check(rule).timeframes == {"1d", "4h"}


def test_shorter_timeframe_than_the_candles_fails_clearly():
    c = Combo(entry="close > sma(5, tf=4h)", exit="")
    with pytest.raises(ValueError, match="shorter than the candles"):
        c.signals(OHLC.from_frame(make_ohlc(100)), c.values)


def test_optimizer_with_a_timeframe_combo():
    from tradebot.optimize import GeneticOptimizer

    c = Combo(entry="close > sma(3, tf=1d)", exit="close < sma(3, tf=1d)")
    assert c.warmup_candles(3600.0) == c.param_space()["sma_1d.period"].high * 24
    opt = GeneticOptimizer(c, {"X": hourly(24 * 60)}, backend="numpy", seed=0, min_trades=0, population=8,
                           generations=1)
    genomes = np.random.default_rng(0).random((6, len(opt.keys)))
    assert np.isfinite(opt.evaluate(genomes)).all()


# -- loadouts, scanner, CLI ---------------------------------------------------------------------------
def test_loadout_keeps_the_timeframe():
    from tradebot.loadouts import capture, restore

    data = capture([INDICATORS.get("sma")(tf="1w", period=10), INDICATORS.get("rsi")()], None, False)
    assert data["indicators"][0]["tf"] == "1w" and "tf" not in data["indicators"][1]
    inds = restore(data)[0]
    assert inds[0].tf == "1w" and inds[1].tf is None


def test_scanner_runs_the_indicator_on_closed_higher_candles():
    from tradebot.scanner import scan

    daily = make_ohlc(60)
    calls = []

    def fetch(sym, tf):
        calls.append(tf)
        return daily

    now = daily.index[-1].timestamp() + 3600  # the last day is still running
    r = scan(["X"], "sma", {"period": 3}, "1h", fetch=fetch, indicator_tf="1d", now=now)
    assert calls == ["1d"]
    assert r.table.value.iloc[0] == pytest.approx(daily.close.iloc[-4:-1].mean())
    with pytest.raises(ValueError, match="longer"):
        scan(["X"], "sma", {}, "1d", fetch=fetch, indicator_tf="4h")


def test_cli_indicator_spec_with_tf():
    from tradebot.cli.main import parse_indicator

    ind = parse_indicator("sma:period=50,tf=1d")
    assert ind.tf == "1d" and ind.values["period"] == 50


# -- evolve -------------------------------------------------------------------------------------------
def test_evolve_offers_higher_timeframe_indicators():
    from tradebot.evolve import runtime as rt
    from tradebot.evolve.primitives import PrimitiveSet
    from tradebot.evolve.program import Evaluator, Node

    ps = PrimitiveSet.default(["sma"], timeframes=("1d",))
    prim = ps["sma.value@1d:price"]
    assert prim.tf == "1d" and "tf=1d" in prim.text and 'tf="1d"' in prim.code
    assert "sma.value:price" in ps  # the candles' own stay
    df = hourly(24 * 20)
    ev = Evaluator(OHLC.from_frame(df))
    node = Node(prim, (Node(None, (), 5, prim.args[0]),))
    out = ev.value(node)
    np.testing.assert_allclose(out, rt.indicator("sma", OHLC.from_frame(df), tf="1d", period=5)["value"],
                               equal_nan=True)


def test_evolver_runs_with_a_higher_timeframe_and_exports_it():
    from tradebot.evolve import Evolver
    from tradebot.evolve.export import compile_signals, signals_source

    ev = Evolver(hourly(24 * 40, seed=8), population=12, generations=2, indicators=["sma", "rsi"],
                 timeframes=("1d",), seed=1, min_trades=0)
    assert any(p.tf == "1d" for p in ev.ps.primitives)
    res = ev.run()
    src = signals_source(res.program)
    assert callable(compile_signals(res.program)) and "def signals" in src
    with pytest.raises(ValueError, match="shorter than the candles"):
        Evolver(make_ohlc(300), population=8, generations=1, indicators=["sma"], timeframes=("4h",))
