# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""The "Best result" preset of the Evolve page and ``tradebot evolve --best-result`` (#101).

The normal defaults are chosen so a run finishes quickly. This preset ignores runtime: it returns the settings
that should give the best *out-of-sample* result, scaled to the data at hand. Maxing everything out would be
wrong (deeper rules and weaker parsimony overfit), so every value has a reason, listed in :data:`WHY`.
"""

from __future__ import annotations

#: Why each value is what it is (also shown in the glossary).
WHY = {
    "population / generations": "Pure search effort: more never hurts the out-of-sample score, so they are high "
                                "(and higher from scratch, whose space is far larger).",
    "hall_of_fame": "A long list gives the validation part more candidates to pick from.",
    "max_depth / max_size": "Not maxed: complexity overfits. Scaled to the history, short histories get small rules "
                            "(from scratch the floor is what an RSI needs, depth 8).",
    "parsimony": "Twice the default: shorter rules win ties, which is what survives on unseen data.",
    "aggregate": "mean-std: a rule has to work consistently on every segment and symbol, not on average.",
    "segments / min_trades": "Grow with the number of candles, so each stretch holds enough trades to judge a rule.",
    "permutations": "20 shuffled-price runs: the smallest count that can show p <= 0.05 with room to spare.",
    "folds": "Walk-forward folds from the history length; too short a history gets none.",
}


def best_result_settings(candles: int, symbols: int = 1, *, from_scratch: bool = False) -> dict:
    """Evolver settings (plus ``permutations`` and ``folds`` for :func:`tradebot.evolve.evolve`) for the best
    out-of-sample result, for ``symbols`` symbols with ``candles`` candles each. The explicit arguments of
    the Evolver still override these (``{**best_result_settings(...), **explicit}``)."""
    n = max(int(candles), 1)
    train = int(n * 0.6)  # validation and test keep 20 % each
    tier = 0 if n < 800 else 1 if n < 3000 else 2
    if from_scratch:
        depth, size = ((10, 70), (12, 100), (14, 130))[tier]
        population, generations = 2000, 250
    else:
        depth, size = ((5, 30), (6, 40), (7, 55))[tier]
        population, generations = 1000, 120
    return dict(
        population=population, generations=generations, max_depth=depth, max_size=size,
        islands=4, tournament=5, crossover=0.6, mutation=0.35, elite=0.02, hall_of_fame=100,
        parsimony=0.004, aggregate="mean-std" if symbols > 1 or n >= 800 else "mean", state=True,
        validation=0.2, test=0.2,
        segments=min(6, max(3, train // 300)),
        min_trades=min(20, max(4, train // 100)),
        permutations=20,
        folds=0 if n < 600 else min(5, max(3, n // 1000)),
    )
