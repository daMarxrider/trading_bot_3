# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""The glossary explains every metric, objective, method and built-in strategy setting (#70), so a new one
can't ship without a plain-language text."""

import inspect

import pytest

from tradebot.algorithms import ALGORITHMS, build_algorithm
from tradebot.backtest import backtest
from tradebot.glossary import METHODS, METRICS, OBJECTIVES, STRATEGIES, TERMS, explain
from tradebot.optimize import OPTIMIZERS
from tradebot.optimize.base import PopulationOptimizer

from .conftest import make_ohlc


def test_every_metric_is_explained():
    metrics = backtest(make_ohlc(300), build_algorithm("sma_cross")).metrics
    assert set(metrics) == set(METRICS)


@pytest.mark.parametrize("objective", ["sharpe", "calmar", "return"])
def test_every_objective_is_explained_and_valid(objective):
    from tradebot.backtest.engine import batch_fitness

    batch_fitness(make_ohlc(50)["close"].pct_change().fillna(0).to_numpy()[None], 365, objective)  # accepted
    assert OBJECTIVES[objective]


def test_every_method_is_explained():
    assert set(METHODS) == {*OPTIMIZERS, "both"}


def test_every_builtin_strategy_and_its_settings_are_explained():
    builtin = [name for name in ALGORITHMS.names() if ALGORITHMS.get(name).__module__.startswith("tradebot.")]
    assert set(builtin) <= set(STRATEGIES)
    for name in builtin:
        for key, spec in build_algorithm(name).param_space().items():
            assert spec.description, f"{name}.{key} has no description"


def test_in_sample_text_matches_the_split():
    default = inspect.signature(PopulationOptimizer.__init__).parameters["train_fraction"].default
    assert f"{default:.0%}" in TERMS["in-sample"] and f"{1 - default:.0%}" in TERMS["out-of-sample"]


def test_explain_looks_everywhere_and_texts_stay_short():
    assert explain("sharpe") == METRICS["sharpe"] and explain("calmar") == OBJECTIVES["calmar"]
    assert explain("fee") == TERMS["fee"] and explain("nonsense") == ""
    for table in (METRICS, OBJECTIVES, METHODS, STRATEGIES, TERMS):
        for key, text in table.items():
            assert 0 < len(text) <= 330, key  # 1–3 short sentences
