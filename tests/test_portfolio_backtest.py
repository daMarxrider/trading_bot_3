# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Portfolio backtest: one shared account across several symbols (#148)."""

import numpy as np
import pandas as pd
import pytest

from tests.conftest import make_ohlc
from tradebot.algorithms import build_algorithm
from tradebot.backtest.baselines import buy_and_hold
from tradebot.backtest.portfolio import align, portfolio_backtest, rebalance_days, run_portfolio


def _closes(n=120, seeds=(1, 2, 3), vols=None):
    frames = {}
    for i, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        vol = vols[i] if vols else 0.02
        frames[f"S{i}"] = 100 * np.exp(np.cumsum(rng.normal(0, vol, n)))
    return pd.DataFrame(frames, index=pd.date_range("2021-01-01", periods=n, freq="D", tz="UTC"))


def test_one_symbol_always_in_is_buy_and_hold(ohlc):
    close = ohlc[["close"]].rename(columns={"close": "X"})
    res, w, contrib = run_portfolio(close, pd.DataFrame(1.0, index=close.index, columns=["X"]), fee=0.001)
    bh = buy_and_hold(ohlc, fee=0.001)
    assert np.allclose(res.equity.to_numpy(), bh.equity.to_numpy())
    assert (w["X"] == 1.0).all() and len(res.trades) == 1
    assert contrib["X"] == pytest.approx(res.equity.iloc[-1] - 10_000)


def test_contributions_add_up_and_drift_without_rebalancing():
    close = _closes()
    sig = pd.DataFrame(1.0, index=close.index, columns=close.columns)
    res, w, contrib = run_portfolio(close, sig, fee=0.002)
    assert contrib.sum() == pytest.approx(res.equity.iloc[-1] - 10_000)
    assert np.allclose(w.iloc[0], 1 / 3)
    assert not np.allclose(w.iloc[-1], 1 / 3)  # drifted with the prices
    assert w.abs().sum(axis=1).max() <= 1 + 1e-9
    _, wm, _ = run_portfolio(close, sig, rebalance="month")
    first = rebalance_days(close.index, "month")
    assert np.allclose(wm[first], 1 / 3)  # set back on the first candle of each month


def test_slots_and_strongest_signal_first():
    close = _closes()
    sig = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    sig.loc[close.index[10]:, "S0"] = 0.5
    sig.loc[close.index[10]:, "S2"] = 1.0
    _, w, _ = run_portfolio(close, sig, max_positions=1, fee=0)
    assert (w.iloc[:10] == 0).all().all()
    assert w["S2"].iloc[10] == pytest.approx(1.0) and (w["S0"] == 0).all()  # the stronger one gets the slot
    _, w, _ = run_portfolio(close, sig, max_positions=2, fee=0)  # each slot is half the account
    assert w["S2"].iloc[10] == pytest.approx(0.5) and w["S0"].iloc[10] == pytest.approx(0.25)
    _, w, _ = run_portfolio(close, sig, max_positions=2, allocation="signal", fee=0)
    assert w["S2"].iloc[10] == pytest.approx(2 / 3) and w["S0"].iloc[10] == pytest.approx(1 / 3)


def test_held_position_keeps_its_slot():
    close = _closes()
    sig = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    sig.loc[close.index[5]:, "S1"] = 1.0
    sig.loc[close.index[8]:, "S0"] = 1.0  # earlier in the list, but the slot is taken
    _, w, _ = run_portfolio(close, sig, max_positions=1)
    assert (w["S0"] == 0).all() and (w["S1"].iloc[5:] > 0).all()


def test_volatility_allocation_favours_the_calm_symbol():
    close = _closes(n=200, seeds=(1, 2), vols=[0.01, 0.04])
    sig = pd.DataFrame(1.0, index=close.index, columns=close.columns)
    _, w, _ = run_portfolio(close, sig, allocation="volatility", rebalance="month")
    month = rebalance_days(close.index, "month")
    later = w[month].iloc[-1]
    assert later["S0"] > 2 * later["S1"] and later.sum() == pytest.approx(1.0)


def test_no_leverage_when_entries_meet_drifted_positions():
    close = _closes(n=300, seeds=(4, 5, 6, 7))
    rng = np.random.default_rng(0)
    sig = pd.DataFrame((rng.random(close.shape) > 0.5).astype(float), index=close.index, columns=close.columns)
    _, w, _ = run_portfolio(close, sig, max_positions=2)
    assert w.abs().sum(axis=1).max() <= 1 + 1e-9
    assert ((w.abs() > 1e-9).sum(axis=1) <= 2).all()


def test_no_look_ahead():
    close = _closes()
    sig = pd.DataFrame(1.0, index=close.index, columns=close.columns)
    sig.iloc[::7, 0] = 0.0
    a, wa, _ = run_portfolio(close, sig, max_positions=2, allocation="volatility", rebalance="week")
    later = close.copy()
    later.iloc[60:] *= 1.5
    b, wb, _ = run_portfolio(later, sig, max_positions=2, allocation="volatility", rebalance="week")
    assert np.allclose(a.returns.iloc[:60], b.returns.iloc[:60])
    assert np.allclose(wa.iloc[:60], wb.iloc[:60])


def test_align_carries_a_stock_over_the_weekend():
    days = pd.date_range("2024-01-01", periods=14, freq="D", tz="UTC")
    crypto = pd.DataFrame({"close": np.arange(14.0) + 1}, index=days)
    stock = pd.DataFrame({"close": np.arange(14.0) + 100}, index=days)[days.day_of_week < 5]
    close, sig = align({"BTC": crypto, "AAPL": stock},
                       {"BTC": pd.Series(1.0, index=crypto.index), "AAPL": pd.Series(1.0, index=stock.index)})
    assert len(close) == 12  # ends on the last stock candle (Friday the 12th)
    sat = pd.Timestamp("2024-01-06", tz="UTC")
    assert close.loc[sat, "AAPL"] == stock["close"].iloc[4] and sig.loc[sat, "AAPL"] == 1.0


def test_portfolio_backtest_with_a_strategy():
    frames = {"A": make_ohlc(400, seed=1), "B": make_ohlc(350, seed=2).iloc[20:], "C": make_ohlc(400, seed=3)}
    algo = build_algorithm("sma_cross", {})
    pf = portfolio_backtest(frames, algo, max_positions=2, allocation="volatility", rebalance="month")
    res = pf.result
    assert res.equity.index[0] == frames["B"].index[0]
    assert set(res.trades["symbol"]) <= set(frames) and {"entry", "exit", "side", "return"} <= set(res.trades)
    cols = pf.columns()
    assert "vs buy & hold" in cols["portfolio"] and "xirr" in cols["buy & hold basket"]
    assert pf.contribution.sum() == pytest.approx(res.equity.iloc[-1] - 10_000)
    per = pf.per_symbol()
    assert list(per.index) == ["A", "B", "C"] and per["trades"].sum() == len(res.trades)
    with pytest.raises(ValueError):
        portfolio_backtest(frames, algo, allocation="random")


def test_cli_portfolio(monkeypatch, tmp_path):
    from typer.testing import CliRunner

    import tradebot.cli.main as cli

    data = {"A": make_ohlc(300, seed=1), "B": make_ohlc(300, seed=2)}
    monkeypatch.setattr(cli, "load_data", lambda s, *a, **k: data[s])
    out = tmp_path / "trades.csv"
    res = CliRunner().invoke(cli.app, ["backtest", "A,B", "sma_cross", "--max-positions", "1", "--no-plot",
                                       "--allocation", "volatility", "--rebalance", "month",
                                       "--trades-csv", str(out)], catch_exceptions=False)
    assert res.exit_code == 0, res.output
    assert "buy & hold basket" in res.output and "per symbol" in res.output
    assert pd.read_csv(out).columns[0] == "symbol"
    bad = CliRunner().invoke(cli.app, ["backtest", "A,B", "sma_cross", "--portfolio", "--rebalance", "daily"])
    assert bad.exit_code != 0
