# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Scanner core (#109) with a fake ``fetch``."""

import threading

import numpy as np
import pandas as pd
import pytest

from tradebot import scanner
from tradebot.core.timeframes import resample
from tradebot.scanner import MAX_SYMBOLS, ScanBusy, scan


def candles(close, n=None):
    close = np.asarray(close, float)
    idx = pd.date_range("2024-01-01", periods=len(close), freq="D", tz="UTC")
    return pd.DataFrame({"open": close, "high": close, "low": close, "close": close}, index=idx)


UP = np.r_[np.full(30, 10.0), np.linspace(10, 20, 10)]  # fast crosses above slow on the way up
DOWN = np.r_[np.full(30, 10.0), np.linspace(10, 2, 10)]
FLAT = np.full(40, 5.0)
DATA = {"UP": UP, "DOWN": DOWN, "FLAT": FLAT}


def fetch(sym, timeframe):
    if sym == "NONE":
        raise RuntimeError("no data")
    return candles(DATA[sym])


def test_rows_have_last_bar_of_every_output_and_alerts_as_bool():
    r = scan(["UP", "FLAT"], "ma_cross", {"fast": 3, "slow": 10}, fetch=fetch)
    assert list(r.table.columns) == ["symbol", "fast", "slow", "cross_up", "cross_down"]
    assert r.table.cross_up.dtype == bool and r.scanned == 2 and r.matched == 2


def test_value_and_alert_filters_are_anded():
    params = {"fast": 3, "slow": 10}
    gt = scan(["UP", "DOWN", "FLAT"], "ma_cross", params, fetch=fetch,
              filters=[{"output": "fast", "op": ">", "value": "slow"}])
    assert list(gt.table.symbol) == ["UP"]
    num = scan(["UP", "DOWN", "FLAT"], "ma_cross", params, fetch=fetch,
               filters=[{"output": "fast", "op": "<", "value": 6.0}])
    assert list(num.table.symbol) == ["DOWN", "FLAT"]
    both = scan(["UP", "DOWN"], "ma_cross", params, fetch=fetch, filters=[
        {"output": "fast", "op": ">", "value": "slow"}, {"alert": "cross_down", "state": False}])
    assert list(both.table.symbol) == ["UP"]


def test_alert_state_matches_only_the_crossing_bar():
    cross = np.r_[np.full(30, 10.0), 20.0]  # the last bar is the crossing bar
    late = np.r_[cross, 21.0]  # one bar later the cross is over
    data = {"A": cross, "B": late}
    r = scan(["A", "B"], "ma_cross", {"fast": 2, "slow": 5}, fetch=lambda s, t: candles(data[s]),
             filters=[{"alert": "cross_up", "state": True}])
    assert list(r.table.symbol) == ["A"]


def test_warmup_rows_never_match_values_and_alerts_are_false():
    r = scan(["UP"], "ma_cross", {"fast": 3, "slow": 10}, fetch=lambda s, t: candles(UP[:5]),
             filters=[{"output": "slow", "op": ">", "value": 0}])  # slow is still NaN
    assert r.scanned == 1 and r.matched == 0
    off = scan(["UP"], "ma_cross", {"fast": 3, "slow": 10}, fetch=lambda s, t: candles(UP[:5]),
               filters=[{"alert": "cross_up", "state": False}])
    assert off.matched == 1


def test_skipped_symbols_get_a_reason():
    r = scan(["UP", "NONE", "EMPTY"], "ma_cross", fetch=lambda s, t: (candles([]) if s == "EMPTY" else fetch(s, t)))
    assert r.scanned == 1 and r.skipped == {"NONE": "no data", "EMPTY": "no data"}


def test_only_the_last_500_bars_are_used():
    seen = {}
    big = candles(np.arange(1.0, 1201.0))

    def f(sym, tf):
        return big

    from tradebot.indicators.base import INDICATORS

    cls = INDICATORS.get("sma")
    orig = cls.compute.__func__

    def spy(c, o, **kw):
        seen["n"] = len(o)
        return orig(c, o, **kw)

    cls.compute = classmethod(spy)
    try:
        scan(["X"], "sma", fetch=f)
    finally:
        cls.compute = classmethod(orig)
    assert seen["n"] == scanner.BARS == 500


@pytest.mark.parametrize("kwargs", [
    {"timeframe": "3h"},
    {"indicator": "elliott"},
    {"indicator": "nope"},
    {"params": {"bogus": 1}},
    {"filters": [{"output": "close", "op": ">", "value": 1}]},
    {"filters": [{"output": "fast", "op": "~", "value": 1}]},
    {"filters": [{"output": "fast", "op": ">", "value": "other"}]},
    {"filters": [{"alert": "bogus"}]},
])
def test_invalid_input_fails_before_fetching(kwargs):
    calls = []
    with pytest.raises((ValueError, KeyError)):
        scan(["UP"], fetch=lambda s, t: calls.append(s), **{"indicator": "ma_cross", **kwargs})
    assert not calls


def test_universe_cap():
    with pytest.raises(ValueError, match=str(MAX_SYMBOLS)):
        scan([f"S{i}" for i in range(MAX_SYMBOLS + 1)], fetch=fetch)
    assert scan([f"S{i}" for i in range(MAX_SYMBOLS)], fetch=lambda s, t: candles(UP)).scanned == MAX_SYMBOLS


def test_only_one_scan_at_a_time():
    started, release = threading.Event(), threading.Event()

    def slow(sym, tf):
        started.set()
        release.wait(5)
        return candles(UP)

    box = {}
    t = threading.Thread(target=lambda: box.update(r=scan(["UP"], fetch=slow)))
    t.start()
    assert started.wait(5)
    with pytest.raises(ScanBusy):
        scan(["UP"], fetch=fetch)
    release.set()
    t.join()
    assert box["r"].scanned == 1
    assert scan(["UP"], fetch=fetch).scanned == 1  # the lock is free again


def test_progress_and_cancel():
    ticks = []
    scan(["UP", "DOWN", "FLAT"], fetch=fetch, progress=lambda d, n: ticks.append((d, n)))
    assert sorted(ticks) == [(1, 3), (2, 3), (3, 3)]
    done = []
    r = scan([f"S{i}" for i in range(50)], fetch=lambda s, t: (done.append(s), candles(UP))[1], workers=1,
             cancel=lambda: len(done) >= 5)
    assert r.cancelled and r.scanned == 5 == len(done)


def test_new_timeframes_and_monthly_resampling():
    from tradebot.core.timeframes import TIMEFRAMES

    assert {"2h", "1mo"} <= set(TIMEFRAMES) and set(scanner.TIMEFRAMES) <= set(TIMEFRAMES)
    idx = pd.date_range("2024-01-15", "2024-04-10", freq="D", tz="UTC")
    df = pd.DataFrame({"open": 1.0, "high": np.arange(len(idx)) + 2.0, "low": 0.5, "close": np.arange(len(idx)) + 1.0},
                      index=idx)
    m = resample(df, "1mo")
    assert list(m.index.strftime("%Y-%m-%d")) == ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"]
    assert m.close.iloc[1] == df.close[df.index < "2024-03-01"].iloc[-1]  # calendar months, not 30-day blocks
    hourly = pd.DataFrame({"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5},
                          index=pd.date_range("2024-01-01", periods=6, freq="h", tz="UTC"))
    assert len(resample(hourly, "2h")) == 3


def test_cli_scan(monkeypatch):
    from typer.testing import CliRunner

    from tradebot.cli.main import app

    monkeypatch.setattr(scanner, "default_fetch", fetch)
    res = CliRunner().invoke(app, ["scan", "UP", "DOWN", "-p", "fast=3", "-p", "slow=10", "--where", "fast > slow",
                                   "--alert", "!cross_down"])
    assert res.exit_code == 0, res.output
    assert "UP" in res.output and "DOWN" not in res.output.split("scanned")[0] and "scanned 2 · 1 match" in res.output
    bad = CliRunner().invoke(app, ["scan", "UP", "--where", "nonsense"])
    assert bad.exit_code != 0
