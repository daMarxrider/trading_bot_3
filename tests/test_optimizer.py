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
