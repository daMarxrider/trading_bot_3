# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Auto-configure (#101): let a search choose the Evolver settings for the data at hand.

An outer successive-halving search over the settings. Every candidate runs a (shortened) evolve on the data
*without its newest* ``test`` *part*; the candidates are ranked by :func:`robustness`, never by the training
score, and the weaker half is dropped before the survivors get a longer run. The test part is not seen by any
of this: it is touched exactly once, by the final run with the chosen settings. Otherwise the overfitting would
only move up one level, into the settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import reduce
from typing import Callable, Mapping

import numpy as np
import pandas as pd

from tradebot.backtest.baselines import BUY_HOLD
from tradebot.evolve.preset import LEVELS, best_result_settings
from tradebot.evolve.search import SIGNIFICANT, EvolutionResult, Progress, evolve

GAP_WEIGHT = 0.5  #: how much of the training-over-validation gap is subtracted
DEFLATE_WEIGHT = 1.0  #: score lost when the deflated Sharpe ratio is 1.0 below :data:`SIGNIFICANT`
CONSISTENCY_WEIGHT = 0.5  #: score lost when the candidate beats buy & hold on none of the symbols
MIN_TRADES = 3  #: validation trades below this make a candidate worthless
FAILED = -1e9
#: the keys the search varies; the rest comes from the preset
SEARCHED = ("max_depth", "max_size", "parsimony", "segments", "min_trades", "aggregate", "tournament", "mutation")
GUARDS = ("permutations", "folds")
#: per level: candidates, rounds, and the checks every candidate gets in the search (shuffled-price runs, folds).
#: "good" only uses what a single run yields (gap, consistency across symbols, deflated Sharpe).
EFFORT = {"good": dict(candidates=8, rounds=3, permutations=0, folds=0),
          "best": dict(candidates=16, rounds=4, permutations=9, folds=3)}


def robustness(result: EvolutionResult) -> float:
    """Out-of-sample quality of one candidate, never its training score. The mean of

    * the validation score minus ``GAP_WEIGHT`` times how far training lies above it (overfitting gap),
    * with walk-forward folds: their mean and their worst unseen score,
    * with shuffled-price runs: how far the validation score lies above the best shuffled run (the margin
      over the permutation test, negative if noise did as well),

    minus penalties for a deflated Sharpe ratio below :data:`SIGNIFICANT` (corrected for the number of
    programs tried) and, with several symbols, for the symbols it does not beat buy & hold on (consistency,
    like min-symbols-beating of #84). Too few validation trades or no validation part: :data:`FAILED`."""
    val, train = result.aggregated("validation"), result.aggregated("train")
    if val is None or train is None or result.trades("validation") < MIN_TRADES or not np.isfinite(val):
        return FAILED
    parts = [val - GAP_WEIGHT * max(0.0, train - val)]
    if (wf := result.walk_forward) is not None:
        scores = [x if x is not None and np.isfinite(x) else 0.0 for x in wf.scores()]
        if scores:
            parts += [float(np.mean(scores)), float(min(scores))]
    if (pt := result.permutation) is not None and pt.shuffled:
        parts.append(val - float(max(pt.shuffled)))
    score = float(np.mean(parts))
    if (d := result.deflated) is not None and np.isfinite(d.probability) and not d.significant:
        score -= DEFLATE_WEIGHT * (SIGNIFICANT - d.probability)
    if result.multi and len(result.symbols) > 1:
        score -= CONSISTENCY_WEIGHT * (1 - result.beats("validation")[BUY_HOLD] / len(result.symbols))
    return score


def hold_out(df: pd.DataFrame | Mapping[str, pd.DataFrame], test: float):
    """The data without its newest ``test`` fraction (by date, over all symbols together)."""
    frames = {"": df} if isinstance(df, pd.DataFrame) else dict(df)
    timeline = reduce(lambda a, b: a.union(b), (f.index for f in frames.values()))
    cut = timeline[len(timeline) - int(round(len(timeline) * test))] if test else None
    cropped = {s: (f if cut is None else f[f.index < cut]) for s, f in frames.items()}
    return cropped[""] if isinstance(df, pd.DataFrame) else cropped


@dataclass
class Trial:
    settings: dict
    round: int
    generations: int
    train: float | None
    validation: float | None
    score: float
    folds: list[float] = field(default_factory=list)  #: walk-forward: each fold's score on its unseen stretch
    margin: float | None = None  #: validation score minus the best shuffled-price run (negative: noise did as well)
    deflated: float | None = None  #: deflated Sharpe ratio of the candidate


@dataclass
class AutoConfig:
    settings: dict  #: for :func:`tradebot.evolve.evolve`, final effort, safeguards included
    trials: list[Trial] = field(default_factory=list)  #: every candidate run, all rounds
    test: float = 0.2  #: the part nothing here has seen
    level: str = "best"

    def report(self) -> list[str]:
        """Plain-language lines: the chosen values, and the scores of the last round's candidates."""
        last = max((t.round for t in self.trials), default=0)
        out = ["chosen: " + ", ".join(f"{k} {v:g}" if isinstance(v, (int, float)) else f"{k} {v}"
                                      for k, v in self.settings.items() if k in SEARCHED + GUARDS)]
        for t in sorted((t for t in self.trials if t.round == last), key=lambda t: -t.score):
            fmt = lambda v: "–" if v is None else f"{v:.3f}"  # noqa: E731
            checks = "".join(
                [f" · folds {', '.join(f'{x:.2f}' for x in t.folds)}" if t.folds else "",
                 f" · margin over shuffled {t.margin:+.2f}" if t.margin is not None else "",
                 f" · deflated {t.deflated:.2f}" if t.deflated is not None and np.isfinite(t.deflated) else ""])
            out.append(f"candidate: train {fmt(t.train)} · validation {fmt(t.validation)}{checks} · "
                       f"robust {t.score:.3f} · "
                       + ", ".join(f"{k} {t.settings[k]:g}" if isinstance(t.settings[k], (int, float))
                                   else f"{k} {t.settings[k]}" for k in SEARCHED))
        out.append(f"The newest {self.test:.0%} of the data was kept out of this search.")
        return out


def _sample(rng: np.random.Generator, base: dict) -> dict:
    """One candidate around the preset ``base``: complexity limits, parsimony, segments, minimum trades,
    aggregation and selection pressure."""
    out = dict(base)
    out["max_depth"] = int(np.clip(base["max_depth"] + rng.integers(-2, 3), 3, 16))
    out["max_size"] = int(np.clip(base["max_size"] * rng.uniform(0.6, 1.5), 10, 160))
    out["parsimony"] = float(np.exp(rng.uniform(np.log(0.001), np.log(0.02))))
    out["segments"] = int(np.clip(base["segments"] + rng.integers(-1, 2), 2, 8))
    out["min_trades"] = int(np.clip(base["min_trades"] * rng.uniform(0.5, 2.0), 2, 40))
    out["aggregate"] = str(rng.choice(["mean", "min", "mean-std", "median"]))
    out["tournament"] = int(rng.integers(3, 8))
    out["mutation"] = float(rng.uniform(0.2, 0.4))
    out["crossover"] = min(base["crossover"], 1 - out["mutation"])
    return out


def auto_configure(df: pd.DataFrame | Mapping[str, pd.DataFrame], *, from_scratch: bool = False,
                   objective: str = "sharpe", level: str = "best", candidates: int | None = None,
                   rounds: int | None = None, population: int | None = None, generations: int | None = None,
                   permutations: int | None = None, folds: int | None = None, seed: int | None = None,
                   progress: Progress | None = None, on_trial: Callable[[int, int, str], None] | None = None,
                   **fixed) -> AutoConfig:
    """Choose the Evolver settings for ``df`` (one frame or ``{symbol: frame}``).

    ``candidates`` settings (the preset among them) run ``rounds`` times, the better half surviving each round
    and getting twice the generations; the last round runs at ``generations`` (default: the level's preset).
    Every candidate is judged by :func:`robustness`. ``level`` "good" (:data:`EFFORT`) judges by one run each
    (gap, consistency across symbols, deflated Sharpe); "best" also gives every candidate ``permutations``
    shuffled-price runs and ``folds`` walk-forward folds, which multiplies the run time. ``fixed`` are Evolver
    settings the search must not change (fee, align, state, dca_amount, ...). ``on_trial(done, total, label)``
    reports each candidate run; ``progress`` is the Evolver's own per-generation callback (raise from it to
    stop). The returned ``settings`` run on the *whole* data, test part included, in
    :func:`tradebot.evolve.evolve`."""
    if level not in LEVELS:
        raise ValueError(f"unknown level {level!r}; choose {', '.join(LEVELS)}")
    effort = EFFORT[level]
    candidates = effort["candidates"] if candidates is None else candidates
    rounds = effort["rounds"] if rounds is None else rounds
    permutations = effort["permutations"] if permutations is None else permutations
    folds = effort["folds"] if folds is None else folds
    seed = int(np.random.SeedSequence().generate_state(1)[0]) if seed is None else seed
    rng = np.random.default_rng(seed)
    n_symbols = 1 if isinstance(df, pd.DataFrame) else len(df)
    frames = [df] if isinstance(df, pd.DataFrame) else list(df.values())
    candles = min(len(f) for f in frames)
    base = best_result_settings(candles, n_symbols, from_scratch=from_scratch, level=level)
    population = population or base["population"]
    generations = generations or base["generations"]
    test = fixed.pop("test", base["test"])
    search_data = hold_out(df, test)
    pool = [dict(base)] + [_sample(rng, base) for _ in range(max(candidates, 1) - 1)]
    total = sum(max(1, len(pool) >> r) for r in range(rounds))
    trials: list[Trial] = []
    done = 0
    for r in range(rounds):
        gens = max(2, generations // 2 ** (rounds - 1 - r))
        scored = []
        for s in pool:
            kwargs = {k: v for k, v in s.items() if k not in GUARDS}
            kwargs.update(fixed, objective=objective, from_scratch=from_scratch, population=population,
                          generations=gens, test=0.0, validation=0.25, deflate=True,
                          seed=int(rng.integers(1 << 31)))
            here = done
            phase = (lambda label, here=here, r=r: on_trial(here, total, f"round {r + 1}/{rounds}, {label}")
                     ) if on_trial else None
            try:
                result = evolve(search_data, permutations=permutations, folds=folds, progress=progress,
                                on_phase=phase, **kwargs)
            except ValueError:  # e.g. too little data for this candidate's segments or folds
                trials.append(Trial(s, r, gens, None, None, FAILED))
                scored.append((FAILED, len(scored), s))
                done += 1
                continue
            done += 1
            wf = result.walk_forward.scores() if result.walk_forward else []
            trials.append(t := Trial(
                s, r, gens, result.aggregated("train"), result.aggregated("validation"), robustness(result),
                folds=[x for x in wf if x is not None],
                margin=(result.permutation.score - max(result.permutation.shuffled)) if result.permutation
                and result.permutation.shuffled else None,
                deflated=result.deflated.probability if result.deflated else None))
            scored.append((t.score, len(scored), s))
        scored.sort(key=lambda x: (-x[0], x[1]))
        pool = [s for _, _, s in scored[:max(1, len(scored) // 2)]]
    if all(t.score == FAILED for t in trials):
        raise ValueError("no setting gave a usable result (too few trades or too little data); "
                         "use more candles or a smaller validation/test part")
    best = {**pool[0], "population": population, "generations": generations, **fixed}
    best["test"] = test
    return AutoConfig(best, trials, test, level)
