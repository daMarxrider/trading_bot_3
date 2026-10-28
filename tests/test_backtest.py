# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Backtest math on hand-checkable series."""

import numpy as np
import pandas as pd
import pytest

from tradebot.algorithms import ALGORITHMS, BaseAlgorithm, build_algorithm
from tradebot.backtest import backtest, batch_fitness, simulate
from tradebot.backtest.metrics import infer_periods_per_year, max_drawdown


def frame(closes, freq="D"):
    c = np.asarray(closes, float)
    idx = pd.date_range("2024-01-01", periods=len(c), freq=freq, tz="UTC")
    return pd.DataFrame({"open": c, "high": c, "low": c, "close": c}, index=idx)


class Fixed(BaseAlgorithm):
    """Test helper: replays a fixed position vector."""

    name = "_fixed_test"

    def __init__(self, pos):
        self.pos = np.asarray(pos, float)
        super().__init__()

    def signals(self, o, p):
        return self.pos


def test_simulate_no_lookahead_and_fees():
    close = np.array([100.0, 110.0, 121.0, 121.0])
    pos = np.array([1.0, 1.0, 0.0, 0.0])
    r = simulate(close, pos, fee=0.01)
    # bar0: enter (fee only); bar1: +10%; bar2: +10% then exit fee; bar3: flat
    np.testing.assert_allclose(r, [-0.01, 0.10, 0.10 - 0.01, 0.0])


def test_buy_and_hold_equity():
    df = frame([100, 110, 99, 120])
    res = backtest(df, Fixed([1, 1, 1, 1]), fee=0.0, initial_capital=1000)
    assert res.equity.iloc[-1] == pytest.approx(1200)
    assert res.metrics["total_return"] == pytest.approx(0.2)
    assert res.metrics["max_drawdown"] == pytest.approx(0.1)
    assert res.metrics["trades"] == 1


def test_trade_stats():
    # long 100->110 (win), flat, long 110->99 (loss)
    df = frame([100, 110, 110, 110, 99, 99])
    res = backtest(df, Fixed([1, 0, 0, 1, 0, 0]), fee=0.0)
    t = res.trades
    assert list(t["return"].round(6)) == [0.1, -0.1]
    assert res.metrics["win_rate"] == 0.5
    assert res.metrics["profit_factor"] == pytest.approx(1.0)


def test_cagr_uses_observed_bar_frequency():
    idx = pd.bdate_range("2023-01-02", periods=253, tz="UTC")  # ~1 trading year
    c = np.linspace(100, 150, 253)
    df = pd.DataFrame({"open": c, "high": c, "low": c, "close": c}, index=idx)
    assert 250 < infer_periods_per_year(idx) < 262
    res = backtest(df, Fixed(np.ones(253)), fee=0.0)
    years = (idx[-1] - idx[0]).days / 365.25
    assert res.metrics["cagr"] == pytest.approx(1.5 ** (1 / years) - 1)


def test_max_drawdown():
    assert max_drawdown(np.array([1, 2, 1, 3, 1.5])) == pytest.approx(0.5)


def test_batch_fitness_matches_single(ohlc):
    algo = build_algorithm("rsi_threshold")
    single = backtest(ohlc, algo, fee=0.001, periods_per_year=365)
    r = simulate(ohlc["close"].to_numpy(), algo.run(ohlc).to_numpy(), 0.001)
    assert batch_fitness(r[None, :], 365)[0] == pytest.approx(single.metrics["sharpe"])


@pytest.mark.parametrize("name", ["rsi_threshold", "macd_cross", "ichimoku_cloud", "sma_cross", "bollinger_reversion", "vote"])
def test_builtin_algorithms_run(ohlc, name):
    res = backtest(ohlc, build_algorithm(name))
    assert res.positions.between(0, 1).all()
    assert np.isfinite(res.equity).all()
    assert "_fixed_test" in ALGORITHMS


def test_range_trader():
    df = frame([10, 9, 11, 12, 8, 7])
    pos = build_algorithm("range", {"buy_price": 9, "sell_price": 12, "stop_price": 7.5}).run(df)
    assert list(pos) == [0, 1, 1, 0, 1, 0]


def test_vote_ensemble_param_space():
    v = build_algorithm("vote", members=["rsi_threshold", "sma_cross"])
    assert "rsi_threshold.lower" in v.param_space()
    v2 = v.with_params(**{"sma_cross.fast": 5})
    assert v2.values["sma_cross.fast"] == 5 and v.values["sma_cross.fast"] == 20
