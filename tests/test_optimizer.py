# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
import numpy as np
import pytest

from tradebot.algorithms import build_algorithm
from tradebot.core import ops
from tradebot.optimize import GeneticOptimizer, select_backend

from .conftest import make_ohlc


def backends():
    out = ["numpy"]
    if ops.torch is not None:
        out.append("cpu")
        if ops.torch.cuda.is_available():
            out.append("gpu")
    return out


@pytest.mark.parametrize("backend", backends())
def test_batched_fitness_matches_across_backends(backend):
    df = make_ohlc(500, seed=3)
    algo = build_algorithm("rsi_threshold")
    ref = GeneticOptimizer(algo, df, backend="numpy", seed=1, train_fraction=1.0, min_trades=0)
    opt = GeneticOptimizer(algo, df, backend=backend, seed=1, train_fraction=1.0, min_trades=0)
    genomes = np.random.default_rng(0).random((64, 3))
    np.testing.assert_allclose(opt.evaluate(genomes), ref.evaluate(genomes), rtol=1e-7, atol=1e-9)


@pytest.mark.parametrize("backend", backends())
def test_optimizer_improves_and_reports_oos(backend):
    df = make_ohlc(600, seed=7)
    algo = build_algorithm("sma_cross")
    res = GeneticOptimizer(algo, df, backend=backend, population=48, generations=6, seed=0).run()
    assert res.history[-1] >= res.history[0]
    assert res.history == sorted(res.history)  # elitism: never gets worse
    assert res.test is not None and len(res.test.equity) == 600 - 420
    assert 2 <= res.best_params["fast"] <= 100


def test_ensemble_is_optimizable():
    df = make_ohlc(400, seed=2)
    algo = build_algorithm("vote", members=["rsi_threshold", "macd_cross"])
    res = GeneticOptimizer(algo, df, backend="numpy", population=16, generations=2, seed=0, min_trades=0).run()
    assert "rsi_threshold.lower" in res.best_params


def test_backend_selection():
    assert select_backend("numpy").kind == "numpy"
    b = select_backend("auto")
    assert b.kind in {"cuda", "rocm", "cpu", "numpy"}


# -- PSO and GA/PSO comparison ------------------------------------------------------------------------
from tradebot.optimize import OPTIMIZERS, ParticleSwarmOptimizer, compare, optimize, run_comparison  # noqa: E402
from tradebot.optimize.base import OptimizationResult  # noqa: E402


@pytest.mark.parametrize("backend", backends())
def test_pso_improves_monotonically_and_reports_oos(backend):
    df = make_ohlc(600, seed=7)
    res = ParticleSwarmOptimizer(build_algorithm("rsi_threshold"), df, backend=backend,
                                 population=48, generations=6, seed=0).run()
    assert res.method == "pso"
    assert res.history == sorted(res.history)  # the swarm's best never gets worse
    assert res.test is not None and len(res.test.equity) == 600 - 420
    space = build_algorithm("rsi_threshold").param_space()
    for k, v in res.best_params.items():
        assert space[k].low <= v <= space[k].high
    assert isinstance(res.best_params["period"], int)


@pytest.mark.parametrize("method", ["ga", "pso"])
def test_search_finds_known_optimum(method):
    """Replace the backtest fitness by a known function to check the search loop itself."""
    target = np.array([0.2, 0.7, 0.4])

    class Toy(OPTIMIZERS[method]):
        def evaluate(self, genomes):
            self.evaluations += len(genomes)
            return -((genomes - target) ** 2).sum(axis=1)

    opt = Toy(build_algorithm("rsi_threshold"), make_ohlc(300), population=40, generations=40, seed=3, backend="numpy")
    res = opt.run()
    found = opt.encode(res.best_params)
    # period is an integer -> its encoded position is quantized to 1/48
    np.testing.assert_allclose(found, target, atol=0.03)


def test_pso_respects_fixed_params_and_seed():
    df = make_ohlc(400, seed=1)
    a = optimize(build_algorithm("rsi_threshold"), df, method="pso", population=16, generations=3, seed=5,
                 backend="numpy", fixed={"period": 10})
    b = optimize(build_algorithm("rsi_threshold"), df, method="pso", population=16, generations=3, seed=5,
                 backend="numpy", fixed={"period": 10})
    assert a.best_params["period"] == 10 and a.best_params == b.best_params
    with pytest.raises(ValueError):
        optimize(build_algorithm("rsi_threshold"), df, method="annealing")


def _fake(method, params, fit, oos_sharpe):
    class R:
        metrics = {"sharpe": oos_sharpe, "total_return": 0.0, "cagr": 0.0, "max_drawdown": 0.1}

    return OptimizationResult(params, fit, [fit], None, R(), R(), 100, method)


SPACE = build_algorithm("rsi_threshold").param_space()  # period 2..50, lower 5..50, upper 50..95


@pytest.mark.parametrize("pso_params, pso_fit, verdict", [
    ({"period": 15, "lower": 31.0, "upper": 66.0}, 1.02, "consistent"),  # tiny diffs
    ({"period": 40, "lower": 10.0, "upper": 90.0}, 0.98, "fragile"),  # far apart, same score
    ({"period": 40, "lower": 10.0, "upper": 90.0}, 0.40, "divergent"),  # far apart, clearly worse
])
def test_comparison_verdicts(pso_params, pso_fit, verdict):
    ga = _fake("ga", {"period": 14, "lower": 30.0, "upper": 65.0}, 1.0, 0.5)
    cmp = compare(ga, _fake("pso", pso_params, pso_fit, 0.5), SPACE)
    assert cmp.verdict == verdict
    assert cmp.param_diff["period"] == pytest.approx(abs(pso_params["period"] - 14) / 48)
    assert 0 <= cmp.agreement <= 1
    assert cmp.explanation()


def test_comparison_flags_overfitting():
    ga = _fake("ga", {"period": 14, "lower": 30.0, "upper": 65.0}, 1.5, -0.2)  # wins in-sample, loses OOS
    pso = _fake("pso", {"period": 14, "lower": 31.0, "upper": 65.0}, 1.4, 0.6)
    cmp = compare(ga, pso, SPACE)
    assert cmp.better is ga and cmp.overfit_warning and "overfitting" in cmp.explanation()
    honest = _fake("pso", {"period": 14, "lower": 31.0, "upper": 65.0}, 1.4, -0.5)  # loses both ways
    assert not compare(ga, honest, SPACE).overfit_warning


def test_run_comparison_same_budget_and_split():
    df = make_ohlc(500, seed=4)
    phases = []
    cmp = run_comparison(build_algorithm("sma_cross"), df, on_phase=phases.append, population=24, generations=4,
                         seed=2, backend="numpy", min_trades=0)
    assert phases == ["ga", "pso"]
    assert cmp.ga.method == "ga" and cmp.pso.method == "pso"
    assert cmp.ga.test.equity.index.equals(cmp.pso.test.equity.index)  # identical out-of-sample window
    assert set(cmp.param_diff) == {"fast", "slow"}


def test_cli_optimize_both(monkeypatch):
    from typer.testing import CliRunner

    from tradebot.cli import main as cli

    monkeypatch.setattr(cli, "load_data", lambda *a, **k: make_ohlc(600, seed=11))
    res = CliRunner().invoke(cli.app, ["optimize", "X", "sma_cross", "--method", "both", "--population", "16",
                                       "--generations", "3", "--backend", "numpy", "--min-trades", "0"],
                             env={"COLUMNS": "140"})
    assert res.exit_code == 0, res.output
    assert "GA vs PSO: parameters" in res.output and "verdict:" in res.output and "agreement" in res.output
    bad = CliRunner().invoke(cli.app, ["optimize", "X", "sma_cross", "--method", "nope"])
    assert bad.exit_code != 0
