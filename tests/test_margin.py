# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Shorts, leverage, liquidation, borrow fees and funding (#149)."""

import numpy as np
import pandas as pd
import pytest

from tradebot.algorithms import build_algorithm
from tradebot.algorithms.direction import Directed, directed
from tradebot.algorithms.sizing import sized
from tradebot.backtest import backtest, simulate
from tradebot.backtest.engine import from_positions
from tradebot.backtest.margin import Margin
from tradebot.data.funding import fetch_history, per_candle, perpetual
from tradebot.optimize import GeneticOptimizer

from .conftest import make_ohlc


def frame(close, low=None, high=None):
    close = np.asarray(close, dtype=float)
    idx = pd.date_range("2024-01-01", periods=len(close), freq="1D", tz="UTC")
    return pd.DataFrame({"open": close, "high": close if high is None else high,
                         "low": close if low is None else low, "close": close}, index=idx)


def run(close, pos, margin=None, low=None, high=None, fee=0.0, funding=None):
    df = frame(close, low, high)
    return from_positions(df, pd.Series(pos, index=df.index, dtype=float), "t", fee=fee, margin=margin,
                          periods_per_year=365.0, funding=funding)


def test_long_without_leverage_is_unchanged(ohlc):
    a = build_algorithm("sma_cross")
    plain = backtest(ohlc, a)
    same = backtest(ohlc, a, margin=Margin())
    assert same.liquidations is None and "liquidations" not in same.metrics
    pd.testing.assert_series_equal(plain.returns, same.returns)


def test_short_gains_when_the_price_falls_and_holds_fixed_units():
    res = run([100, 100, 90, 80], [0, -1, -1, -1])
    # fixed units: short 1x from 100 to 80 earns 20 %, not 1 - 0.9 * (1 - 1/9)
    assert res.equity.iloc[-1] / res.equity.iloc[0] == pytest.approx(1.2)
    assert res.metrics["liquidations"] == 0
    assert res.trades["side"].tolist() == ["short"]


def test_leverage_multiplies_the_move_on_the_margin():
    res = run([100, 100, 110], [0, 1, 1], Margin(leverage=3))
    assert res.equity.iloc[-1] / res.equity.iloc[0] == pytest.approx(1.3)


def test_leveraged_long_is_liquidated_at_the_low_and_loses_the_margin():
    # 10x: liquidation about 9.5 % below the entry; the low touches it although the close recovers
    close = [100, 100, 95, 99, 120]
    low = [100, 100, 89, 98, 119]
    res = run(close, [0, 1, 1, 1, 1], Margin(leverage=10, maintenance=0.005), low=low)
    assert res.metrics["liquidations"] == 1
    assert bool(res.liquidations.iloc[2])
    assert res.equity.iloc[-1] == pytest.approx(0.0)  # the whole account was the margin
    # half the account as margin: only that half is lost, the rest stays flat afterwards
    half = run(close, [0, 0.5, 0.5, 0.5, 0.5], Margin(leverage=10), low=low)
    assert half.equity.iloc[-1] / half.equity.iloc[0] == pytest.approx(0.5)


def test_no_liquidation_just_above_the_liquidation_price():
    res = run([100, 100, 95], [0, 1, 1], Margin(leverage=10), low=[100, 100, 91])
    assert res.metrics["liquidations"] == 0
    assert res.equity.iloc[-1] / res.equity.iloc[0] == pytest.approx(0.5)


def test_unleveraged_short_is_liquidated_when_the_price_doubles():
    res = run([100, 100, 150, 210, 100], [0, -1, -1, -1, -1], high=[100, 100, 160, 215, 210])
    assert res.metrics["liquidations"] == 1 and bool(res.liquidations.iloc[3])
    assert res.equity.iloc[-1] == pytest.approx(0.0)
    unleveraged_long = run([100, 100, 1, 50], [0, 1, 1, 1], low=[100, 100, 0.5, 1])
    assert "liquidations" not in unleveraged_long.metrics  # long without leverage: the plain model, never liquidated
    assert unleveraged_long.equity.iloc[-1] > 0


def test_new_entry_after_a_liquidation_trades_again():
    close = [100, 100, 80, 80, 80, 88]
    pos = [0, 0.5, 0.5, 0, 0.5, 0.5]
    res = run(close, pos, Margin(leverage=5))
    assert res.metrics["liquidations"] == 1
    # half lost, then 0.5 of the remaining 0.5 at 5x gains 10 % * 5 = 50 % on that margin
    assert res.equity.iloc[-1] / res.equity.iloc[0] == pytest.approx(0.5 * (1 + 0.5 * 0.5))


def test_fees_scale_with_the_notional():
    res = run([100, 100, 100], [0, 1, 0], Margin(leverage=4), fee=0.001)
    assert res.equity.iloc[-1] / res.equity.iloc[0] == pytest.approx((1 - 0.004) ** 2)


def test_borrow_fee_and_funding():
    flat = [100.0] * 366
    short = [0.0] + [-1.0] * 365
    res = run(flat, short, Margin(borrow=0.0365))
    assert res.equity.iloc[-1] / res.equity.iloc[0] == pytest.approx((1 - 0.0001) ** 364)  # held from the 2nd candle
    # funding: longs pay a positive rate, shorts receive it; 3 settlements a day
    long = [0.0] + [1.0] * 365
    paid = run(flat, long, Margin(funding=0.0001))
    got = run(flat, short, Margin(funding=0.0001))
    assert paid.equity.iloc[-1] / paid.equity.iloc[0] == pytest.approx((1 - 0.0003) ** 364, rel=1e-3)
    assert got.equity.iloc[-1] > got.equity.iloc[0]
    hist = run(flat, long, funding=np.full(366, 0.001))
    assert hist.equity.iloc[-1] / hist.equity.iloc[0] == pytest.approx((1 - 0.001) ** 364)


def test_margin_validation():
    with pytest.raises(ValueError):
        Margin(leverage=0.5)
    with pytest.raises(ValueError):
        Margin(leverage=10, maintenance=0.2)
    with pytest.raises(ValueError):
        Margin(borrow=-1)


def test_simulate_batched_matches_single_and_torch():
    df = make_ohlc(300)
    close = df["close"].to_numpy()
    rng = np.random.default_rng(0)
    pos = np.sign(rng.normal(size=(4, 300))) * (rng.random((4, 300)) > 0.3)
    m = Margin(leverage=3, borrow=0.02, funding=0.0001)
    kw = dict(margin=m, low=df["low"].to_numpy(), high=df["high"].to_numpy(), periods_per_year=365.0)
    batched = simulate(close, pos, 0.001, 0.0, **kw)
    for i in range(4):
        np.testing.assert_allclose(batched[i], simulate(close, pos[i], 0.001, 0.0, **kw))
    torch = pytest.importorskip("torch")
    t = simulate(torch.tensor(close), torch.tensor(pos), 0.001, 0.0, margin=m, low=torch.tensor(kw["low"]),
                 high=torch.tensor(kw["high"]), periods_per_year=365.0)
    np.testing.assert_allclose(t.numpy(), batched, atol=1e-10)


@pytest.mark.parametrize("name", ["rsi_threshold", "macd_cross", "sma_cross", "ichimoku_cloud",
                                  "bollinger_reversion", "halving_cycle", "vote"])
def test_short_variants_of_the_built_ins(name, ohlc):
    a = build_algorithm(name)
    long = a.run(ohlc)
    both = directed(a, "long_short").run(ohlc)
    short = directed(a, "short").run(ohlc)
    assert directed(a, "long") is a
    assert short.max() <= 0 and both.min() >= -1
    assert ((both > 0) == (long > 0)).all()  # the long side is unchanged
    assert (short < 0).equals(both < 0)
    if name not in ("halving_cycle", "vote"):  # no exit inside this short sample
        assert (both < 0).any()


def test_mirror_rules():
    df = make_ohlc(500, seed=3)
    from tradebot.core.series import OHLC
    from tradebot.indicators.builtin import Bollinger

    boll = directed(build_algorithm("bollinger_reversion"), "long_short").run(df)
    upper = np.asarray(Bollinger.compute(OHLC.from_frame(df), period=20, width=2.0)["upper"])
    entries = ((boll < 0) & (boll.shift(1, fill_value=0) >= 0)).to_numpy()
    assert entries.any() and (df["close"].to_numpy()[entries] > upper[entries]).all()
    ich = directed(build_algorithm("ichimoku_cloud"), "long_short").run(df)
    assert (ich == 0).sum() > 100 and (ich < 0).any()  # flat inside the cloud


def test_rsi_long_short_reverses():
    a = build_algorithm("rsi_threshold")
    pos = directed(a, "long_short").run(make_ohlc(800, seed=5))
    first = pos[pos != 0].index[0]
    held = pos[first:]
    assert set(np.unique(held)) <= {-1.0, 1.0}  # always in the market once it started


def test_short_with_intrabar_fills_is_rejected():
    combo = build_algorithm("combo", entry="sma_cross", exit="stop_pct(5)", fills="intrabar")
    with pytest.raises(ValueError):
        directed(combo, "short")


def test_combo_and_sizing_on_top(ohlc):
    combo = build_algorithm("combo", entry="macd_cross & close>sma(50)")
    pos = directed(combo, "long_short").run(ohlc)
    assert (pos < 0).any()
    s = sized(directed(build_algorithm("sma_cross"), "long_short"), "fixed", **{"size.fraction": 0.5})
    assert set(np.unique(s.run(ohlc))) <= {-0.5, 0.0, 0.5}
    res = backtest(ohlc, s, margin=Margin(leverage=2))
    assert "liquidations" in res.metrics


def test_with_params_keeps_the_direction(ohlc):
    a = Directed(build_algorithm("sma_cross"), "short")
    b = a.with_params(fast=10)
    assert isinstance(b, Directed) and b.direction == "short" and b.values["fast"] == 10
    assert b.run(ohlc).max() <= 0


def test_optimizer_scores_shorts_and_leverage_with_liquidation():
    df = make_ohlc(500, seed=2)
    a = directed(build_algorithm("sma_cross"), "long_short")
    opt = GeneticOptimizer(a, df, population=8, generations=2, seed=1, backend="numpy",
                           margin=Margin(leverage=5), min_trades=1)
    res = opt.run()
    assert "liquidations" in res.train.metrics
    # the optimizer's score agrees with the backtest of the winner on the training candles
    tuned = a.with_params(**res.best_params)
    again = backtest(opt.data[0].train_df, tuned, opt.fee, periods_per_year=opt.data[0].ppy, margin=Margin(leverage=5))
    np.testing.assert_allclose(again.returns.to_numpy(), res.train.returns.to_numpy())
    windows = GeneticOptimizer(a, make_ohlc(800, seed=2), population=8, generations=1, seed=1, backend="numpy", samples=3,
                               sample_length=80, margin=Margin(leverage=5, borrow=0.02), min_trades=1).run()
    assert np.isfinite(windows.best_fitness)


def test_funding_history_and_per_candle():
    class Ex:
        has = {"fetchFundingRateHistory": True}

        def __init__(self):
            start = pd.Timestamp("2024-01-01", tz="UTC")
            self.rows = [{"timestamp": int((start + pd.Timedelta(hours=8 * i)).timestamp() * 1000),
                          "fundingRate": 0.0001 * (i % 3 + 1)} for i in range(30)]
            self.calls = []

        def fetch_funding_rate_history(self, symbol, since=None, limit=None):
            self.calls.append(symbol)
            return [r for r in self.rows if r["timestamp"] >= since][:4]

    ex = Ex()
    rates = fetch_history("BTC/USDT", pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-01-11", tz="UTC"),
                          exchange=ex)
    assert ex.calls[0] == "BTC/USDT:USDT" and len(rates) == 30
    idx = pd.date_range("2024-01-01", periods=10, freq="1D", tz="UTC")
    per = per_candle(rates, idx)
    assert per == pytest.approx(np.full(10, 0.0006))
    assert perpetual("ETH/USDT:USDT") == "ETH/USDT:USDT" and perpetual("AAPL") == "AAPL"


def test_cli_backtest_and_optimize_with_shorts_and_leverage(monkeypatch):
    from typer.testing import CliRunner

    from tradebot.cli import main as cli

    df = make_ohlc(400, seed=4)
    monkeypatch.setattr(cli, "load_data", lambda *a, **k: df)
    run_cli = lambda *args: CliRunner().invoke(cli.app, list(args), env={"COLUMNS": "160"})  # noqa: E731
    res = run_cli("backtest", "BTC/USDT", "sma_cross", "--no-plot", "--direction", "long_short", "--leverage", "3",
                  "--borrow-fee", "0.02", "--funding", "0.0001")
    assert res.exit_code == 0, res.output
    assert "long/short" in res.output and "liquidations" in res.output
    res = run_cli("backtest", "BTC/USDT", "sma_cross", "--no-plot", "--leverage", "50")
    assert res.exit_code == 0, res.output
    assert "liquidated" in res.output
    assert run_cli("backtest", "X", "sma_cross", "--no-plot", "--direction", "sideways").exit_code != 0
    assert run_cli("backtest", "X", "sma_cross", "--no-plot", "--leverage", "10",
                   "--maintenance-margin", "0.5").exit_code != 0
    assert run_cli("backtest", "X,Y", "sma_cross", "--portfolio", "--no-plot", "--leverage", "2").exit_code != 0
    res = run_cli("optimize", "X", "sma_cross", "--population", "8", "--generations", "1", "--backend", "numpy",
                  "--min-trades", "0", "--direction", "short", "--leverage", "2")
    assert res.exit_code == 0, res.output
    assert run_cli("optimize", "X", "sma_cross", "--funding", "history").exit_code != 0


def test_cli_funding_history(monkeypatch):
    from typer.testing import CliRunner

    import tradebot.data.funding as funding
    from tradebot.cli import main as cli

    df = make_ohlc(100, seed=4)
    monkeypatch.setattr(cli, "load_data", lambda *a, **k: df)
    rates = pd.Series(0.01, index=pd.date_range(df.index[0], periods=300, freq="8h"))
    monkeypatch.setattr(funding, "fetch_history", lambda *a, **k: rates)
    res = CliRunner().invoke(cli.app, ["backtest", "BTC/USDT", "sma_cross", "--no-plot", "--funding", "history"],
                             env={"COLUMNS": "160"})
    assert res.exit_code == 0, res.output

    def offline(*a, **k):
        raise ConnectionError("offline")

    monkeypatch.setattr(funding, "fetch_history", offline)
    res = CliRunner().invoke(cli.app, ["backtest", "BTC/USDT", "sma_cross", "--no-plot", "--funding", "history"])
    assert res.exit_code != 0 and "funding" in res.output
