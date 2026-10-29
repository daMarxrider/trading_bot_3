# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
from tradebot.algorithms import elliott, pool  # noqa: F401  (register the Elliott wave strategy and the pool)
from tradebot.algorithms.base import ALGORITHMS, BaseAlgorithm
from tradebot.algorithms.builtin import VotingEnsemble, build_algorithm


def algorithm_names() -> list[str]:
    """Registered algorithm names for menus: single strategies first, then the ensembles (pool, vote)."""
    return sorted(ALGORITHMS.names(), key=lambda name: issubclass(ALGORITHMS.get(name), VotingEnsemble))


__all__ = ["ALGORITHMS", "BaseAlgorithm", "algorithm_names", "build_algorithm"]
