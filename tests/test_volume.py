# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Volume as a first-class series (#116)."""
import numpy as np
import pandas as pd

from tradebot.core.series import OHLC
from tradebot.core.timeframes import resample
from tradebot.evolve.primitives import causal_outputs
from tradebot.gui.chart import volume_buckets
from tradebot.indicators.base import INDICATORS


def run(cls, o):
    return cls.compute(o, **{k: v.default for k, v in cls.params.items()})


def frame(n=30, volume=1.0):
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    close = 100 + np.arange(n, dtype=float)
    d = {"open": close - 0.5, "high": close + 1, "low": close - 1, "close": close}
    if volume is not None:
        d["volume"] = volume
    return pd.DataFrame(d, index=idx)


def test_missing_or_zero_volume_is_nan():
    for df in (frame(volume=None), frame(volume=0.0)):
        o = OHLC.from_frame(df)
        assert o.volume.shape == o.close.shape and np.isnan(o.volume).all()
    assert len(OHLC.from_frame(frame()).fields()) == 6
    assert np.isnan(OHLC(*[np.ones(3)] * 4).volume).all()  # positional construction still works


def test_volume_reaches_indicators_and_nan_without_data():
    o = OHLC.from_frame(frame(volume=np.arange(1, 31, dtype=float)))
    assert np.allclose(run(INDICATORS.get("volume"), o)["volume"], o.volume)
    nan = OHLC.from_frame(frame(volume=None))
    for name in ("obv", "mfi", "vwap", "volume_ratio"):
        cls = INDICATORS.get(name)
        assert cls.uses_volume
        out = run(cls, nan)
        assert all(np.isnan(np.asarray(v, float)).all() for k, v in out.items() if k in cls.outputs), name


def test_resampling_sums_volume():
    r = resample(frame(24, volume=2.0), "1d")
    assert r["volume"].iloc[0] == 48.0


def test_merged_bars_sum_volume():
    x, o, c = np.arange(6.0), np.arange(6.0), np.arange(6.0) + 1
    v = np.array([1, 2, 3, 4, np.nan, 6.0])
    _, _, vm, _ = volume_buckets(x, o, v, c, 0, 6, 3)
    assert list(vm) == [6.0, 10.0]


def test_volume_indicators_are_causal():
    for name in ("volume", "obv", "vwap", "mfi", "volume_ratio"):
        cls = INDICATORS.get(name)
        assert set(causal_outputs(cls)) >= set(cls.outputs[:1]), name


def test_volume_unit():
    assert INDICATORS.get("volume").units["volume"] == "volume"
    assert set(INDICATORS.get("volume_ratio").units.values()) == {"ratio"}


def test_screener_volume_ratio():
    from tradebot.screener import screen_row
    vol = np.r_[np.full(29, 10.0), 40.0]
    row = screen_row("X", frame(30, volume=vol))
    assert row["avg_volume"] == vol[-20:].mean() and np.isclose(row["volume_ratio"], 40 / vol[-20:].mean())
    assert np.isnan(screen_row("X", frame(30, volume=None))["volume_ratio"])


def test_cli_chart_volume_flag():
    from tradebot.cli.charts import render_chart
    df = frame(60, volume=np.arange(1, 61, dtype=float))
    with_vol = render_chart(df, "t", window=60, volume=True)
    without = render_chart(df, "t", window=60, volume=False)
    assert "volume" in with_vol and "volume" not in without
    assert "volume" not in render_chart(frame(60, volume=None), "t", window=60, volume=True)


def test_optimizer_windows_carry_volume():
    from tradebot.algorithms import build_algorithm
    from tradebot.optimize import GeneticOptimizer
    from .conftest import make_ohlc
    df = make_ohlc(400, seed=1)
    df["volume"] = np.arange(len(df), dtype=float) + 1
    opt = GeneticOptimizer(build_algorithm("rsi_threshold"), df, backend="numpy", seed=0, samples=3,
                           sample_length=100, train_fraction=1.0, min_trades=0)
    assert opt.resample()
    w = opt._windows.ohlc
    assert w.volume.shape == w.close.shape and np.isfinite(w.volume).all()
    assert all(np.isfinite(r.volume).all() for r in opt._windows.ohlc_np)
