# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Evolve whole strategies from scratch with genetic programming (#83).

* :mod:`~tradebot.evolve.primitives` - the typed building blocks (prices, indicators, arithmetic, logic)
* :mod:`~tradebot.evolve.program` - expression trees, simplification, evaluation
* :mod:`~tradebot.evolve.search` - the search with its train / validation / test split
* :mod:`~tradebot.evolve.export` - the result as a ``BaseAlgorithm`` plugin file
"""

from tradebot.evolve.export import plugin_source, valid_name
from tradebot.evolve.autoconfig import AutoConfig, auto_configure
from tradebot.evolve.preset import best_result_settings
from tradebot.evolve.primitives import PrimitiveSet
from tradebot.evolve.program import Evaluator, Node, Program
from tradebot.evolve.search import EvolutionResult, Evolver, PermutationTest, evolve, shuffle_candles

__all__ = ["AutoConfig", "auto_configure", "best_result_settings", "EvolutionResult", "Evaluator", "Evolver", "Node", "PermutationTest", "PrimitiveSet", "Program",
           "evolve", "plugin_source", "shuffle_candles", "valid_name"]
