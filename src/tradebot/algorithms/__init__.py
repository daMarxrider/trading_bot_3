# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
from tradebot.algorithms import combo, elliott, pool  # noqa: F401  (register the Elliott wave strategy, pool, combo)
from tradebot.algorithms.base import ALGORITHMS, BaseAlgorithm
from tradebot.algorithms.builtin import VotingEnsemble, build_algorithm


def algorithm_names() -> list[str]:
    """Registered algorithm names for menus: single strategies first, then the ensembles (pool, vote), then
    the combos (``combo`` and the saved ones)."""
    def rank(name):
        cls = ALGORITHMS.get(name)
        return 2 if issubclass(cls, combo.Combo) else 1 if issubclass(cls, VotingEnsemble) else 0

    return sorted(ALGORITHMS.names(), key=rank)


__all__ = ["ALGORITHMS", "BaseAlgorithm", "algorithm_names", "build_algorithm"]
