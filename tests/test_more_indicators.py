# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""The extra built-in indicators (#162): ADX, Supertrend, SAR, Keltner, CCI, %R, ROC, momentum, pivots, volume profile,
anchored VWAP."""
import numpy as np
import pandas as pd
import pytest
from .conftest import make_ohlc

from tradebot.core.series import OHLC
from tradebot.evolve.primitives import causal_outputs
from tradebot.indicators import INDICATORS

NEW = ("adx", "supertrend", "psar", "keltner", "cci", "williams_r", "roc", "momentum", "pivots", "volume_profile",
       "avwap")


def out(name, df, **p):
    return INDICATORS.get(name)(**p)(df)


def close_eq(a, b):
    np.testing.assert_allclose(np.asarray(a, float), np.asarray(b, float), rtol=1e-9, atol=1e-9, equal_nan=True)


def test_registered_with_alerts():
    for name in NEW:
        cls = INDICATORS.get(name)
        assert cls.alerts, name
        assert set(cls.alerts) <= set(cls.outputs) and set(cls.alerts) <= set(cls.hidden)


def test_all_run_and_are_causal():
    df = make_ohlc(300)
    for name in NEW:
        cls = INDICATORS.get(name)
        full = out(name, df)
        part = out(name, df.iloc[:200])
        for col in full:
            close_eq(full[col].iloc[:200], part[col])  # no look-ahead
        assert set(causal_outputs(cls)) >= set(cls.plot_outputs()[:1]), name


def test_cci_williams_roc_momentum_match_pandas():
    df = make_ohlc()
    tp = (df.high + df.low + df.close) / 3
    mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    close_eq(out("cci", df)["cci_value"], (tp - tp.rolling(20).mean()) / (0.015 * mad))
    hh, ll = df.high.rolling(14).max(), df.low.rolling(14).min()
    close_eq(out("williams_r", df)["williams_r_value"], -100 * (hh - df.close) / (hh - ll))
    close_eq(out("roc", df, period=12)["roc_value"], (df.close / df.close.shift(12) - 1) * 100)
    close_eq(out("momentum", df, period=10)["momentum_value"], df.close - df.close.shift(10))


def test_keltner_matches_reference():
    df = make_ohlc()
    r = out("keltner", df)
    mid = df.close.ewm(span=20, adjust=False).mean()
    close_eq(r["keltner_middle"].iloc[19:], mid.iloc[19:])
    assert (r["keltner_upper"].dropna() > r["keltner_middle"].dropna()).all()


def test_adx_matches_wilder_reference():
    df = make_ohlc()
    n = 14
    up, dn = df.high.diff(), -df.low.diff()
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    pc = df.close.shift()
    tr = pd.concat([df.high - df.low, (df.high - pc).abs(), (df.low - pc).abs()], axis=1).max(axis=1)
    tr.iloc[0] = df.high.iloc[0] - df.low.iloc[0]
    w = lambda s: pd.Series(np.asarray(s, float), index=df.index).ewm(alpha=1 / n, adjust=False).mean()  # noqa: E731
    pdm[0] = mdm[0] = np.nan  # no previous bar
    atr = w(tr)
    pdi, mdi = 100 * w(pdm) / atr, 100 * w(mdm) / atr
    adx = (100 * (pdi - mdi).abs() / (pdi + mdi)).ewm(alpha=1 / n, adjust=False).mean()
    r = out("adx", df, period=n)
    close_eq(r["adx_plus_di"].iloc[60:], pdi.reindex(df.index).iloc[60:])
    close_eq(r["adx_adx"].iloc[100:], adx.reindex(df.index).iloc[100:])
    assert (r["adx_adx"].dropna().between(0, 100)).all()


def test_supertrend_follows_trend_and_flips():
    n = 120
    close = np.r_[np.linspace(100, 160, n // 2), np.linspace(160, 90, n // 2)]
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    df = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1.0}, index=idx)
    r = out("supertrend", df)
    assert r["supertrend_direction"].iloc[40] == 1 and r["supertrend_direction"].iloc[-1] == -1
    assert r["supertrend_up"].iloc[40] < close[40] and r["supertrend_down"].iloc[-1] > close[-1]
    assert r["supertrend_flip_down"].sum() == 1 and r["supertrend_flip_up"].iloc[20:].sum() == 0


def test_psar_trails_and_flips():
    n = 120
    close = np.r_[np.linspace(100, 160, n // 2), np.linspace(160, 90, n // 2)]
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    df = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1.0}, index=idx)
    r = out("psar", df)
    assert r["psar_trend"].iloc[30] == 1 and r["psar_trend"].iloc[-1] == -1
    assert (r["psar_up"].dropna() < df.low[r["psar_up"].notna()] + 1e-9).all()
    assert r["psar_flip_down"].sum() == 1


def test_pivots_use_previous_day():
    df = make_ohlc(72, freq="h")
    r = out("pivots", df, anchor=0)
    day = df.index.normalize()
    prev = df.groupby(day).agg(h=("high", "max"), l=("low", "min"), c=("close", "last")).shift()
    p = ((prev.h + prev.l + prev.c) / 3).reindex(day).values
    close_eq(r["pivots_p"], p)
    close_eq(r["pivots_r1"], 2 * p - prev.l.reindex(day).values)
    assert r["pivots_p"].iloc[:24].isna().all()


def test_pivots_nan_without_time():
    o = OHLC.from_frame(make_ohlc(50))
    o = OHLC(o.open, o.high, o.low, o.close, None, o.volume)
    assert np.isnan(INDICATORS.get("pivots").compute(o, anchor=0)["p"]).all()


def test_volume_profile_poc_is_busiest_price():
    n = 60
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    close = np.where(np.arange(n) % 2 == 0, 100.0, 110.0)
    close[::3] = 105.0
    vol = np.where(close == 105.0, 100.0, 1.0)
    df = pd.DataFrame({"open": close, "high": close + 0.1, "low": close - 0.1, "close": close, "volume": vol},
                      index=idx)
    r = out("volume_profile", df, period=30, bins=20)
    assert abs(r["volume_profile_poc"].iloc[-1] - 105) < 1
    assert r["volume_profile_val"].iloc[-1] <= r["volume_profile_poc"].iloc[-1] <= r["volume_profile_vah"].iloc[-1]
    assert r["volume_profile_poc"].iloc[:29].isna().all()


def test_anchored_vwap():
    df = make_ohlc(50)
    start = df.index[10]
    r = out("avwap", df, anchor=int(start.strftime("%Y%m%d")))["avwap_value"]
    sub = df.iloc[10:]
    tp = (sub.high + sub.low + sub.close) / 3
    close_eq(r.iloc[10:], (tp * sub.volume).cumsum() / sub.volume.cumsum())
    assert r.iloc[:10].isna().all()
    close_eq(out("avwap", df, anchor=0)["avwap_value"].iloc[0], (df.high + df.low + df.close).iloc[0] / 3)


def test_volume_indicators_nan_without_volume():
    df = make_ohlc(150).drop(columns="volume")
    for name in ("volume_profile", "avwap"):
        assert out(name, df)[f"{name}_{INDICATORS.get(name).outputs[0]}"].isna().all(), name


def test_cli_lists_new_indicators():
    from typer.testing import CliRunner

    from tradebot.cli import main as cli
    res = CliRunner().invoke(cli.app, ["list", "indicators"])
    assert res.exit_code == 0
    for name in NEW:
        assert name in res.output
