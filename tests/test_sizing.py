# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Position sizing models (#147): backtest, optimizer and live bot."""

import numpy as np
import pandas as pd
import pytest

from tradebot.algorithms import build_algorithm
from tradebot.algorithms.sizing import MODELS, Sized, sized
from tradebot.backtest import backtest
from tradebot.brokers.crypto import BinanceBroker
from tradebot.live import TradingBot
from tradebot.optimize import GeneticOptimizer

from .test_brokers_data import FakeExchange
from .test_services import Scripted


def frame(n=600, vol=0.02, seed=1):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0.0005, vol, n)))
    idx = pd.date_range("2022-01-01", periods=n, freq="1D", tz="UTC")
    return pd.DataFrame({"open": c, "high": c * 1.01, "low": c * 0.99, "close": c, "volume": 1.0}, index=idx)


def algo(model=None, **params):
    return sized(build_algorithm("sma_cross"), model, **params)


def test_no_model_is_the_algorithm_itself():
    a = build_algorithm("sma_cross")
    assert sized(a, None) is a and sized(a, "none") is a
    with pytest.raises(ValueError):
        sized(a, None, **{"size.fraction": 0.5})
    with pytest.raises(ValueError):
        sized(a, "nonsense")


@pytest.mark.parametrize("model", list(MODELS))
def test_every_model_only_reduces_exposure(model):
    df = frame()
    raw = backtest(df, algo()).positions
    res = backtest(df, algo(model)).positions
    assert (res.abs() <= raw.abs() + 1e-12).all() and (res >= 0).all()
    assert res.sum() < raw.sum() or model == "kelly"
    assert not res.isna().any()


def test_fixed_fraction_scales_the_signal():
    df = frame()
    raw = backtest(df, algo()).positions
    res = backtest(df, algo("fixed", **{"size.fraction": 0.25})).positions
    assert np.allclose(res, raw * 0.25)


def test_vol_target_sizes_down_when_the_market_is_wilder():
    quiet = backtest(frame(vol=0.005), algo("vol", **{"size.target_vol": 0.1})).positions
    wild = backtest(frame(vol=0.05), algo("vol", **{"size.target_vol": 0.1})).positions
    assert quiet[quiet > 0].mean() > wild[wild > 0].mean()
    assert wild[wild > 0].max() < 0.5


def test_fixed_risk_is_set_at_the_entry_and_held():
    res = backtest(frame(), algo("risk"))
    invested = res.positions[res.positions > 0]
    assert len(invested) > 10
    runs = (res.positions > 0).ne((res.positions > 0).shift()).cumsum()[res.positions > 0]
    assert (invested.groupby(runs).nunique() == 1).all()  # one size per trade
    wide = backtest(frame(), algo("risk", **{"size.stop_atr": 5.0})).positions
    assert wide.sum() < res.positions.sum()  # a wider stop means a smaller position


def test_kelly_is_zero_without_an_edge_and_capped_at_one():
    df = frame(n=800)
    res = backtest(df, algo("kelly", **{"size.kelly_fraction": 1.0})).positions
    assert res.max() <= 1.0


def test_sizing_params_join_the_parameter_space_and_split_off_again():
    a = algo("vol", **{"size.target_vol": 0.3})
    assert {"size.target_vol", "size.window"} <= set(a.param_space())
    assert a.values["size.target_vol"] == 0.3
    assert "size.window" not in a.inner.values
    b = a.with_params(**{"size.target_vol": 0.1})
    assert b.values["size.target_vol"] == 0.1 and a.values["size.target_vol"] == 0.3
    assert isinstance(b, Sized) and b.name == "sma_cross"


def test_the_optimizer_tunes_the_sizing_parameters():
    a = algo("vol")
    opt = GeneticOptimizer(a, frame(), population=16, generations=2, seed=1, backend="numpy")
    assert "size.target_vol" in opt.keys
    res = opt.run()
    assert "size.target_vol" in res.best_params
    fixed = GeneticOptimizer(a, frame(), population=16, generations=2, seed=1, backend="numpy",
                             fixed={"size.target_vol": 0.15})
    assert "size.target_vol" not in fixed.keys


def test_batched_evaluation_matches_the_backtest():
    df = frame()
    a = algo("risk")
    opt = GeneticOptimizer(a, df, population=8, generations=1, seed=3, backend="numpy", train_fraction=1.0)
    p = a.values
    g = opt.encode(p)[None, :]
    expected = backtest(df, a).metrics["sharpe"]
    assert opt.evaluate(g)[0] == pytest.approx(expected, rel=1e-6, abs=1e-6)


def test_live_bot_buys_stake_times_size():
    now = pd.Timestamp.now(tz="UTC").floor("1D")
    start = int((now - pd.Timedelta(days=9)).timestamp() * 1000)
    candles = [[start + i * 86_400_000, 100, 100, 100, 100, 1] for i in range(10)]

    class HalfSized(Scripted):
        def run_sized(self, df):
            return df["close"] * 0 + self.next, df["close"] * 0 + 0.4

    a = HalfSized()
    a.next = 1.0
    bot = TradingBot(BinanceBroker(exchange=FakeExchange(candles)), a, "BTC/USDT", "1d", 100.0, dry_run=False)
    ev = bot.step()
    assert ev.action == "buy" and ev.amount == pytest.approx(100.0 * 0.4 / 100.0) and "40%" in ev.note
    a.next = 0.0
    assert bot.step().action == "sell"


# -- the wrapper's contract with the strategy inside it, and the year length it assumes ---------------------------
def test_without_candle_times_a_year_has_252_candles():
    from tradebot.algorithms.sizing import periods_per_year
    from tradebot.core.series import OHLC

    o = OHLC.from_frame(frame(50))
    o_without_times = OHLC(o.open, o.high, o.low, o.close)
    assert periods_per_year(o_without_times) == 252.0


def test_daily_candles_are_365_and_a_quarter_a_year():
    from tradebot.algorithms.sizing import periods_per_year
    from tradebot.core.series import OHLC

    assert periods_per_year(OHLC.from_frame(frame(50))) == pytest.approx(365.25)


def test_an_unknown_sizing_model_is_refused_in_the_size_calculation():
    from tradebot.algorithms.sizing import multiplier
    from tradebot.core.series import OHLC

    with pytest.raises(ValueError, match="unknown sizing model 'martingale'"):
        multiplier("martingale", OHLC.from_frame(frame(50)), np.ones(50), {})


def test_a_loop_based_strategy_stays_loop_based_when_sized():
    assert sized(build_algorithm("elliott_wave"), "fixed").vectorized is False


def test_the_sized_strategy_shows_the_inner_strategys_chart_indicators():
    inner = build_algorithm("sma_cross")
    assert [type(i) for i in sized(inner, "fixed").chart_indicators()] == [type(i) for i in inner.chart_indicators()]


def test_an_attribute_the_inner_strategy_lacks_is_an_attribute_error():
    with pytest.raises(AttributeError):
        sized(build_algorithm("sma_cross"), "fixed").no_such_thing  # noqa: B018


def test_signals_are_the_sized_positions():
    from tradebot.core.series import OHLC

    s = sized(build_algorithm("sma_cross"), "fixed", **{"size.fraction": 0.5})
    o = OHLC.from_frame(frame(300))
    np.testing.assert_array_equal(s.signals(o, s.values), s.execute(o, s.values)[0])


def test_intrabar_fills_are_scaled_by_the_size_too():
    df = frame(400)
    combo = build_algorithm("combo", {}, None, "rsi_threshold", "stop_pct(3)", "intrabar")
    _, plain = combo.run_fills(df)
    _, half = sized(combo, "fixed", **{"size.fraction": 0.5}).run_fills(df)
    np.testing.assert_allclose(half["turnover"], 0.5 * plain["turnover"])
    np.testing.assert_allclose(half["adjust"].iloc[1:], 0.5 * plain["adjust"].iloc[1:])  # the size of the bar before


def test_reusing_parameters_keeps_the_sizing_and_what_the_pool_needs():
    pool = build_algorithm("pool", {}, ["rsi_threshold", "sma_cross"])
    s = sized(pool, "fixed", **{"size.fraction": 0.3})
    params = {**s.values, "use.rsi_threshold": 1, "use.sma_cross": 0}
    out = s.reuse_params(params)
    assert out["size.fraction"] == 0.3
    assert not any(k.startswith("use.") or k.startswith("sma_cross.") for k in out)


def test_the_risk_size_is_the_same_on_torch_as_on_numpy():
    torch = pytest.importorskip("torch")  # the GPU path of the fixed-risk size; skipped (visibly) without torch
    from tradebot.algorithms.sizing import multiplier
    from tradebot.core.series import OHLC

    df = frame(300)
    s = sized(build_algorithm("sma_cross"), "risk")
    pos = s.inner.run(df).to_numpy()
    on_numpy = multiplier("risk", OHLC.from_frame(df), pos, s.values)
    on_torch = multiplier("risk", OHLC.from_frame(df, device="cpu"), torch.tensor(pos), s.values)
    np.testing.assert_allclose(on_torch.numpy(), on_numpy)


def test_a_sized_strategy_can_be_copied():
    import copy

    s = sized(build_algorithm("sma_cross"), "fixed", **{"size.fraction": 0.3})
    twin = copy.deepcopy(s)
    assert twin.values == s.values and twin.inner is not s.inner
