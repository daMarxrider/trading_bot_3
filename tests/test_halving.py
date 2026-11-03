# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Bitcoin halving schedule, forecast, indicator and strategy (#99)."""

import numpy as np
import pandas as pd
import pytest

from tradebot.algorithms import build_algorithm
from tradebot.algorithms.pool import RULES
from tradebot.analysis import halving as hv
from tradebot.core.series import OHLC
from tradebot.indicators import INDICATORS

from .conftest import make_ohlc

DAY = hv.DAY


def btc_like(n: int = 4500, start: str = "2014-01-01") -> pd.DataFrame:
    df = make_ohlc(n, seed=3)
    df.index = pd.date_range(start, periods=n, freq="D", tz="UTC")
    return df


def at(date: str) -> float:
    return pd.Timestamp(date, tz="UTC").timestamp()


# -- schedule and forecast ----------------------------------------------------------------------------
def test_schedule_follows_the_protocol():
    assert hv.NOMINAL_DAYS == pytest.approx(1458.33, abs=0.01)
    assert [h for h, _ in hv.HALVINGS] == [hv.INTERVAL * k for k in range(1, len(hv.HALVINGS) + 1)]
    assert list(hv.STARTS) == sorted(hv.STARTS) and hv.STARTS[0] == hv.GENESIS
    assert pd.Timestamp(hv.HALVINGS[-1][1], unit="s") == pd.Timestamp("2024-04-20 00:09:27")
    # every halving so far came early, as blocks were faster than 600 s while the hashrate grew
    assert all(s < hv.TARGET_SPACING for s in hv.spacings(hv.STARTS))


def test_the_chosen_estimator_is_the_best_on_the_past_halvings():
    assert hv.best_estimator() == hv.ESTIMATOR
    errors = {name: hv.mean_abs_error(name) for name in hv.ESTIMATORS}
    assert errors[hv.ESTIMATOR] < errors["nominal"]  # better than the plain 4-year rule


def test_backtest_predicts_each_halving_from_the_earlier_ones_only():
    nominal = hv.backtest("nominal")
    assert [f.height for f in nominal] == [h for h, _ in hv.HALVINGS[1:]]
    # the 4-year rule predicted 2016 from 2012: 1458.33 days, the halving came after 1319.1
    assert nominal[0].error_days == pytest.approx(hv.NOMINAL_DAYS - (hv.STARTS[2] - hv.STARTS[1]) / DAY)
    mean = hv.backtest("mean")[0]  # only the genesis epoch was known before the 2016 halving
    assert mean.predicted == pytest.approx(hv.STARTS[1] + (hv.STARTS[1] - hv.STARTS[0]))


def test_forecast_from_the_chain_tip():
    height, ts = 840_000 + 100_000, hv.HALVINGS[-1][1] + 100_000 * 590
    target, when, pace = hv.next_from_tip(height, ts)
    assert target == 1_050_000 and pace == pytest.approx(590)
    assert when == pytest.approx(ts + 110_000 * 590)
    assert hv.next_from_tip(840_000, hv.HALVINGS[-1][1])[:2] == (1_050_000, hv.HALVINGS[-1][1] + 210_000 * 600)


# -- the cycle clock ----------------------------------------------------------------------------------
def test_cycle_counts_days_since_the_latest_halving():
    t = np.array([hv.GENESIS - 1, hv.STARTS[2] - DAY, hv.STARTS[2], hv.STARTS[2] + 10 * DAY, np.nan])
    c = hv.cycle(t)
    assert np.isnan(c["days"][0]) and np.isnan(c["days"][-1])
    assert c["days"][1] == pytest.approx((hv.STARTS[2] - hv.STARTS[1]) / DAY - 1)
    assert c["days"][2] == 0 and c["days"][3] == pytest.approx(10)
    # the next halving as predicted right after 2016, from 2012 and earlier only
    nxt = hv.predict(hv.STARTS[:3])[0]
    assert c["days_to_next"][3] == pytest.approx((nxt - t[3]) / DAY)
    assert c["phase"][3] == pytest.approx(10 * DAY / (nxt - hv.STARTS[2]))


def test_cycle_goes_on_with_forecast_halvings_after_the_known_ones():
    nxt = hv.predict()[0]
    c = hv.cycle(np.array([nxt - DAY, nxt + DAY]))
    assert c["days"][0] > 1300 and c["days"][1] == pytest.approx(1)


def test_cycle_on_torch_matches_numpy():
    torch = pytest.importorskip("torch")
    t = np.linspace(at("2011-01-01"), at("2030-01-01"), 500)[None, None, :]
    a, b = hv.cycle(t), hv.cycle(torch.tensor(t, dtype=torch.float64))
    for k in a:
        np.testing.assert_allclose(b[k].numpy(), a[k])


# -- indicator ----------------------------------------------------------------------------------------
def test_ohlc_carries_the_timestamps():
    df = btc_like(10)
    o = OHLC.from_frame(df)
    assert o.time[0] == at("2014-01-01") and np.all(np.diff(o.time) == DAY)
    assert OHLC.from_frame(df.reset_index(drop=True)).time is None


def test_indicator_on_candles():
    ind = INDICATORS.get("halving")()
    out = ind(btc_like())
    assert list(out.columns) == ["halving_days", "halving_days_to_next", "halving_phase"]
    assert out.loc["2016-07-10", "halving_days"] == pytest.approx(1 - (hv.STARTS[2] % DAY) / DAY)
    assert out["halving_days"].max() < 1500 and not out.isna().any().any()
    assert not ind.is_line("halving_phase")
    no_time = INDICATORS.get("halving").compute(OHLC(*OHLC.from_frame(btc_like(5)).fields()[:4]))
    assert all(np.isnan(v).all() for v in no_time.values())


def test_evolve_can_use_the_halving_clock():
    from tradebot.evolve.primitives import indicator_outputs

    found = {(cls.name, out): unit for cls, out, unit in indicator_outputs(["halving"])}
    assert found == {("halving", "days"): "halving", ("halving", "days_to_next"): "halving"}


# -- strategy -----------------------------------------------------------------------------------------
def test_strategy_spans_the_halving_with_the_defaults():
    df = btc_like()
    pos = build_algorithm("halving_cycle").run(df)
    days = INDICATORS.get("halving")()(df)["halving_days"]
    assert ((pos == 1) == ((days >= 850) | (days < 530))).all()
    assert pos.loc["2016-07-09"] == 1 and pos.loc["2017-06-01"] == 1 and pos.loc["2018-06-01"] == 0


def test_strategy_window_within_one_cycle_and_batched_params():
    df = btc_like()
    o = OHLC.from_frame(df)
    algo = build_algorithm("halving_cycle")
    days = INDICATORS.get("halving").compute(o)["days"]
    pos = algo.signals(o, {"buy_day": np.array([[100.0], [850.0]]), "sell_day": np.array([[400.0], [530.0]])})
    assert pos.shape == (2, len(df))
    np.testing.assert_array_equal(pos[0], ((days >= 100) & (days < 400)).astype(float))
    np.testing.assert_array_equal(pos[1], algo.signals(o, algo.values))


def test_optimizer_random_windows_keep_the_dates():
    from tradebot.optimize.genetic import GeneticOptimizer

    data = {"BTC": btc_like(3000)}
    opt = GeneticOptimizer(build_algorithm("halving_cycle"), data, backend="numpy", seed=0, samples=3,
                           sample_length=400, population=8, generations=2, min_trades=0)
    opt.resample()
    for w in opt._windows.ohlc_np:  # each window's clock matches its own dates
        assert w.time is not None and np.all(np.diff(w.time) == DAY)
    assert opt._windows.ohlc.time.shape == opt._windows.ohlc.close.shape
    assert np.isfinite(opt.run().best_fitness)


def test_pool_rule():
    df = btc_like()
    o = OHLC.from_frame(df)
    rule = RULES["halving"]
    days = INDICATORS.get("halving").compute(o)["days"]
    allowed = rule.condition(o, {k: v.default for k, v in rule.params.items()})
    np.testing.assert_array_equal(allowed, (days >= 850) | (days < 530))


def test_cli_shows_halvings_and_forecasts(monkeypatch):
    from typer.testing import CliRunner

    from tradebot.cli import main as cli

    monkeypatch.setattr(hv, "fetch_tip", lambda: (974_325, 1_793_667_617))
    res = CliRunner().invoke(cli.app, ["halving", "--live"], env={"COLUMNS": "160"})
    assert res.exit_code == 0, res.output
    assert "2024-04-20" in res.output and "mean ✓" in res.output and "2028-04-08" in res.output
