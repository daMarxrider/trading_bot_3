# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Slippage next to the fee in the backtester, optimizer, baselines and settings (#146)."""

import numpy as np
import pytest

from tradebot.algorithms import build_algorithm
from tradebot.backtest import backtest, baselines, bps, simulate
from tradebot.backtest.baselines import buy_and_hold, monthly_investing
from tradebot.backtest.slippage import asset_class, configured_bps, resolve_bps, suggested_bps
from tradebot.core import ops
from tradebot.glossary import explain
from tradebot.optimize import GeneticOptimizer

from .conftest import make_ohlc


def test_simulate_charges_slippage_like_the_fee():
    close = np.array([100.0, 110.0, 121.0, 121.0])
    pos = np.array([0.0, 1.0, 1.0, 0.0])
    base = simulate(close, pos, 0.0, 0.0)
    both = simulate(close, pos, 0.001, 0.0005)
    assert (base - both).sum() == pytest.approx(2 * 0.0015)  # one round trip
    np.testing.assert_allclose(simulate(close, pos, 0.0015), both)  # only the sum matters


def test_zero_slippage_changes_nothing():
    df = make_ohlc(300, seed=1)
    algo = build_algorithm("rsi_threshold")
    a, b = backtest(df, algo), backtest(df, algo, slippage=0.0)
    np.testing.assert_array_equal(a.returns, b.returns)


def test_slippage_lowers_returns_and_baselines():
    df = make_ohlc(300, seed=1)
    algo = build_algorithm("rsi_threshold")
    free, costly = backtest(df, algo), backtest(df, algo, slippage=bps(20))
    assert costly.metrics["total_return"] < free.metrics["total_return"]
    assert buy_and_hold(df, slippage=bps(20)).equity.iloc[-1] < buy_and_hold(df).equity.iloc[-1]
    assert monthly_investing(df, slippage=bps(20)).returns.sum() < monthly_investing(df).returns.sum()
    bl = baselines(df, costly, slippage=bps(20))
    assert bl.results["buy & hold"].equity.iloc[-1] == pytest.approx(buy_and_hold(df, slippage=bps(20)).equity.iloc[-1])


def backends():
    out = ["numpy"]
    if ops.torch is not None:
        out.append("cpu")
    return out


@pytest.mark.parametrize("backend", backends())
def test_batched_fitness_with_slippage_matches_numpy(backend):
    df = make_ohlc(500, seed=3)
    algo = build_algorithm("rsi_threshold")
    kw = dict(seed=1, train_fraction=1.0, min_trades=0, slippage=bps(10))
    ref = GeneticOptimizer(algo, df, backend="numpy", **kw)
    opt = GeneticOptimizer(algo, df, backend=backend, **kw)
    genomes = np.random.default_rng(0).random((64, 3))
    np.testing.assert_allclose(opt.evaluate(genomes), ref.evaluate(genomes), rtol=1e-7, atol=1e-9)
    free = GeneticOptimizer(algo, df, backend="numpy", seed=1, train_fraction=1.0, min_trades=0)
    assert not np.allclose(ref.evaluate(genomes), free.evaluate(genomes))


@pytest.mark.skipif(ops.torch is None, reason="torch not installed")
def test_simulate_torch_matches_numpy():
    torch = ops.torch
    rng = np.random.default_rng(0)
    close = 100 + rng.normal(size=200).cumsum() * 0.1
    pos = rng.integers(0, 2, size=(8, 200)).astype(float)
    ref = simulate(close, pos, 0.001, 0.0007)
    got = simulate(torch.as_tensor(close), torch.as_tensor(pos), 0.001, 0.0007)
    np.testing.assert_allclose(got.numpy(), ref, atol=1e-12)


def test_optimizer_result_uses_slippage():
    df = make_ohlc(400, seed=2)
    algo = build_algorithm("rsi_threshold")
    kw = dict(backend="numpy", seed=1, population=8, generations=2, min_trades=0)
    free = GeneticOptimizer(algo, df, **kw).run()
    costly = GeneticOptimizer(algo, df, slippage=bps(50), **kw).run()
    assert costly.train_baselines.results["buy & hold"].equity.iloc[-1] < free.train_baselines.results["buy & hold"].equity.iloc[-1]


def test_evolver_takes_slippage():
    from tradebot.evolve import Evolver

    ev = Evolver(make_ohlc(300, seed=1), population=8, generations=1, slippage=bps(5), seed=1)
    assert ev.slippage == bps(5) and ev.settings["slippage"] == bps(5)


def test_asset_class_defaults_and_config():
    assert asset_class("BTC/USDT") == "crypto" and asset_class("AAPL") == "stock" and asset_class("EURUSD=X") == "fx"
    assert suggested_bps("BTC/USDT") > suggested_bps("AAPL") > 0
    assert configured_bps("AAPL", {}) == 0.0
    assert configured_bps("AAPL", {"backtest": {"slippage_bps": 3}}) == 3.0
    table = {"backtest": {"slippage_bps": {"crypto": 5, "stock": 2}}}
    assert configured_bps("ETH/USDT", table) == 5.0 and configured_bps("AAPL", table) == 2.0 and configured_bps("EURUSD=X", table) == 0.0


def test_resolve_bps(monkeypatch):
    monkeypatch.setenv("TRADEBOT_CONFIG", "/nonexistent/config.toml")
    assert resolve_bps(None, "AAPL") == 0.0
    assert resolve_bps("auto", "BTC/USDT") == suggested_bps("BTC/USDT")
    assert resolve_bps("7.5") == 7.5
    with pytest.raises(ValueError):
        resolve_bps("-1")


def test_glossary_explains_slippage():
    assert "basis points" in explain("slippage")


def test_cli_backtest_passes_slippage(monkeypatch):
    from typer.testing import CliRunner

    from tradebot.cli import main as cli

    df = make_ohlc(300, seed=1)
    monkeypatch.setattr(cli, "load_data", lambda *a, **k: df)
    monkeypatch.setenv("TRADEBOT_CONFIG", "/nonexistent/config.toml")
    out = {}
    for label, args in (("free", []), ("slip", ["--slippage", "100"])):
        res = CliRunner().invoke(cli.app, ["backtest", "AAPL", "rsi_threshold", "--no-plot", *args])
        assert res.exit_code == 0, res.output
        out[label] = res.output
    assert out["free"] != out["slip"]
    bad = CliRunner().invoke(cli.app, ["backtest", "AAPL", "rsi_threshold", "--no-plot", "--slippage", "-3"])
    assert bad.exit_code != 0
