# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""The search (#83): genetic programming of whole strategies, with the overfitting safeguards it needs.

A population of random programs (ramped half-and-half) evolves by tournament selection, subtree
crossover (also swapping whole entry or exit rules between parents), and mutation: a new random subtree,
a primitive swapped for one of the same signature (``>`` / ``<``, ``sma`` / ``ema``, ``and`` / ``or``),
a constant nudged, or a subtree hoisted up in place of its parent (shrinks bloated trees). The best few
survive unchanged (elitism). Every new tree is simplified and limited in depth and size, and the
fitness pays ``parsimony`` per node, so of two equally good rules the shorter one wins.

The search space is huge, so the data is split by date (all symbols at the same dates)::

    |---------- train (segments 1..k) ----------|-- validation --|-- test --|

* **Fitness** is computed on the training part only: the objective on each of ``segments`` consecutive
  stretches of each symbol, combined by ``aggregate``. A rule has to work in every stretch and on
  every symbol, not just on average over one series. Fewer than ``min_trades`` round trips on a
  symbol's training part rejects the program.
* **Selection on validation**: the best programs ever seen (the hall of fame) are scored on the
  validation part, and the best there is the result. The validation part never drives evolution.
* **Test**: the result is scored once on the test part, which nothing in the search has seen, next to
  the #19 baselines.
* Optional **permutation test**: the same search on prices with shuffled candles (same moves, random
  order, so no pattern is left to find). If the evolver scores as well on noise, the result is noise.

**From scratch** (#100, ``from_scratch=True``): rules are built from candles and low-level maths only
(:meth:`PrimitiveSet.from_scratch`), with higher depth and size limits and a growing library of evolved
building blocks (:mod:`tradebot.evolve.modules`). The bigger search space fits noise more easily, so the
mode turns on more safeguards by default:

* **behavioural deduplication**: programs that trade exactly alike in training count as one (the
  duplicates make room for new ideas, and the hall of fame holds different strategies),
* the **deflated Sharpe ratio** of the result: its training Sharpe ratio against the best one luck
  reaches over the number of distinct strategies tried,
* the **permutation test** and **walk-forward folds** (:func:`evolve`): the whole search again on
  rolling windows, each result scored on the stretch after its window.

Indicators are computed over the whole history and only *scored* per part: all primitives are causal,
so this equals computing them on each part with a warm-up, and a position open at a boundary carries
over as it would in live trading.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from functools import reduce
from typing import Callable, Mapping

import numpy as np
import pandas as pd

from tradebot.backtest import BacktestResult, Baselines, baselines, batch_fitness, simulate
from tradebot.backtest.baselines import BUY_HOLD, MONTHLY
from tradebot.backtest.engine import from_positions
from tradebot.backtest.metrics import deflated_sharpe, infer_periods_per_year
from tradebot.core.series import OHLC
from tradebot.evolve.export import load_plugin, plugin_source
from tradebot.evolve.modules import Library, rewrite
from tradebot.evolve.primitives import BOOL, Primitive, PrimitiveSet, uses_volume
from tradebot.evolve.program import (Evaluator, Node, Program, constants, log_scale, quantize, random_program,
                                     random_tree, simplify_tree)
from tradebot.optimize.base import AGGREGATES, carried_by_few, objective_value, select_frames

log = logging.getLogger(__name__)

REJECTED = -1e9
LIMITS = (6, 40)  #: default max depth and max size
SCRATCH_LIMITS = (12, 100)  #: from scratch: an RSI alone is depth 8 and ~25 nodes
SCRATCH_GUARDS = {"permutations": 5, "folds": 3}  #: from scratch: shuffled-price runs and walk-forward folds
SCRATCH_ISLANDS = 4  #: from scratch: separate populations (see :class:`Evolver`)
SIGNIFICANT = 0.95  #: the deflated Sharpe ratio a result should reach
#: ``progress(generation, best_fitness, best_program)``
Progress = Callable[[int, float, Program], None]


@dataclass
class Candidate:
    program: Program
    fitness: float  #: on the training part, after the parsimony penalty
    generation: int
    validation: float | None = None


@dataclass
class SymbolReport:
    """The result on one symbol: a backtest per part, and the baselines on validation and test."""

    symbol: str
    train: BacktestResult
    validation: BacktestResult | None
    test: BacktestResult | None
    validation_baselines: Baselines | None
    test_baselines: Baselines | None


@dataclass
class PermutationTest:
    """The same search on shuffled candles: how often it found something as good on noise."""

    score: float  #: the real result's out-of-sample score
    shuffled: list[float]  #: the shuffled runs' out-of-sample scores
    part: str = "test"

    @property
    def p(self) -> float:
        """Share of runs (counting the real one) at least as good as the real result: small = unlikely to be luck."""
        return (1 + sum(s >= self.score for s in self.shuffled)) / (1 + len(self.shuffled))


@dataclass
class Deflated:
    """The result's training Sharpe ratio, corrected for how many distinct strategies the search tried."""

    probability: float  #: that the Sharpe ratio beats what luck reaches over ``trials`` tries (NaN: undefined)
    sharpe: float  #: per candle, on the training part (all symbols' returns in a row)
    luck: float  #: the best per-candle Sharpe ratio expected from ``trials`` strategies without an edge
    trials: int  #: distinct strategies tried (by their training positions)

    @property
    def significant(self) -> bool:
        return bool(self.probability >= SIGNIFICANT)


@dataclass
class WalkForward:
    """The whole search once per fold: evolved on a rolling window, scored on the stretch after it."""

    folds: list["EvolutionResult"]

    def scores(self) -> list[float | None]:
        """Each fold's objective on its unseen stretch, combined over the symbols."""
        return [f.aggregated("test") for f in self.folds]

    def losing(self) -> int:
        return sum(s is not None and s < 0 for s in self.scores())

    def beating(self) -> int:
        """Folds in which the result beat buy & hold on at least half the symbols."""
        return sum(f.beats("test")[BUY_HOLD] >= len(f.symbols) / 2 for f in self.folds)

    def warnings(self, objective: str) -> list[str]:
        n = len(self.folds)
        out = []
        if (k := self.losing()) > n / 2 - 1e-9 and k:
            out.append(f"Walk-forward: evolved afresh on {n} rolling windows, the result had a negative {objective} on "
                       f"the unseen stretch after its window in {k} of {n} folds.")
        if (b := self.beating()) < n / 2:
            out.append(f"Walk-forward: the fold results beat buy & hold in only {b} of {n} folds.")
        return out


@dataclass
class EvolutionResult:
    program: Program
    fitness: float
    symbols: list[SymbolReport]
    history: list[dict]  #: per generation: best fitness, its validation score, mean size, distinct programs
    hall_of_fame: list[Candidate]
    objective: str = "sharpe"
    aggregate: str = "mean"
    evaluations: int = 0
    seconds: float = 0.0
    seed: int | None = None
    generation: int = 0  #: the generation the result first appeared in
    validation_start: pd.Timestamp | None = None
    test_start: pd.Timestamp | None = None
    skipped: dict[str, str] = field(default_factory=dict)
    permutation: PermutationTest | None = None
    from_scratch: bool = False
    modules: list[Primitive] = field(default_factory=list)  #: the run's library of evolved building blocks
    deflated: Deflated | None = None
    walk_forward: WalkForward | None = None
    no_volume: list[str] = field(default_factory=list)  #: symbols without volume data, if the program uses volume

    @property
    def multi(self) -> bool:
        return len(self.symbols) > 1 or bool(self.symbols and self.symbols[0].symbol)

    def objectives(self, part: str = "test") -> dict[str, float]:
        """The objective per symbol on ``part`` ("train", "validation" or "test")."""
        return {r.symbol: objective_value(getattr(r, part).metrics, self.objective)
                for r in self.symbols if getattr(r, part) is not None}

    def aggregated(self, part: str = "test") -> float | None:
        values = np.array(list(self.objectives(part).values()), dtype=float)
        return float(AGGREGATES[self.aggregate](values[:, None])[0]) if len(values) else None

    def beats(self, part: str = "test") -> dict[str, int]:
        """Baseline -> on how many symbols the strategy's XIRR beat it on ``part``."""
        out = {BUY_HOLD: 0, MONTHLY: 0}
        for r in self.symbols:
            if (bl := getattr(r, f"{part}_baselines")) is not None:
                for name in bl.results:
                    out[name] += name not in bl.not_beaten
        return out

    def trades(self, part: str = "test") -> int:
        return sum(len(getattr(r, part).trades) for r in self.symbols if getattr(r, part) is not None)

    def rows(self) -> list[dict]:
        """One row per symbol: objective per part, then XIRR, lead per baseline and trades on the unseen part
        (:attr:`unseen`; None for a symbol without one)."""
        out = []
        part = self.unseen
        for r in self.symbols:
            res, bl = (getattr(r, part), getattr(r, f"{part}_baselines")) if part else (None, None)
            out.append({"symbol": r.symbol, **{k: objective_value(x.metrics, self.objective) if x is not None else None
                                               for k, x in (("train", r.train), ("validation", r.validation),
                                                            ("test", r.test))},
                        "xirr": bl.strategy_xirr if bl else None,
                        **{f"vs {k}": bl.excess[k] if bl else None for k in (BUY_HOLD, MONTHLY)},
                        "trades": len(res.trades) if res is not None else None})
        return out

    @property
    def unseen(self) -> str | None:
        """The part the honest score comes from: "test", else "validation", else None."""
        return "test" if self.test_start is not None else "validation" if self.validation_start is not None else None

    def warnings(self) -> list[str]:
        """Plain-language flags for the reports."""
        out = []
        part = self.unseen
        if part is None:
            return ["There is no validation or test part: every score is in-sample, which says nothing about the "
                    "future for a rule found in a search this large."]
        n, label = len(self.symbols), "test" if part == "test" else "validation"
        beats = self.beats(part)[BUY_HOLD]
        if beats < n / 2 or not beats:
            out.append(f"On the {label} part, the evolved strategy beats buy & hold on only {beats} of {n} "
                       f"symbol{'s' if n > 1 else ''}.")
        train, unseen = self.aggregated("train"), self.aggregated(part)
        if train is not None and unseen is not None and train > 0 and unseen < train / 2:
            out.append(f"It is much weaker on unseen data ({self.objective} {train:.2f} in training, {unseen:.2f} "
                       f"on the {label} part): probably fitted to noise in the training data.")
        if (trades := self.trades(part)) < 3:
            out.append(f"Only {trades} trade{'s' if trades != 1 else ''} on the {label} part; the score says little.")
        if self.multi and carried_by_few(self.objectives(part).values()):
            out.append("The result is carried by one or two symbols (median far below the mean).")
        if self.permutation is not None and self.permutation.p > 0.05:
            pt = self.permutation
            k = sum(s >= pt.score for s in pt.shuffled)
            if k:
                out.append(f"It doesn't beat shuffled prices: there (no patterns to find) the evolver scored as well "
                           f"in {k} of {len(pt.shuffled)} runs (p = {pt.p:.2f}). No evidence of a real edge.")
            else:
                out.append(f"It beat all {len(pt.shuffled)} shuffled-price runs, but so few runs can't show p below "
                           f"{pt.p:.2f}; use 19 or more for p ≤ 0.05.")
        if (d := self.deflated) is not None and not d.significant:
            out.append(f"After allowing for the {d.trials} different strategies the search tried, the training Sharpe "
                       f"ratio is not significant (deflated Sharpe ratio {d.probability:.2f}, needs {SIGNIFICANT}): "
                       "luck alone reaches that high.")
        if self.walk_forward is not None:
            out += self.walk_forward.warnings(self.objective)
        if self.no_volume:
            out.append(f"The rules use volume, which {', '.join(s or 'the symbol' for s in self.no_volume)} "
                       "has no data for: there they never fire (no trades).")
        return out

    def notes(self) -> list[str]:
        """How it was evolved, for the plugin's docstring."""
        out = [f"Evolved on {', '.join(r.symbol for r in self.symbols if r.symbol) or 'one symbol'} "
               f"({self.symbols[0].train.equity.index[0]:%Y-%m-%d} to "
               f"{(self.symbols[0].test or self.symbols[0].validation or self.symbols[0].train).equity.index[-1]:%Y-%m-%d}), "
               f"objective {self.objective}, seed {self.seed}."]
        for part in ("train", "validation", "test"):
            if (v := self.aggregated(part)) is not None:
                out.append(f"{part}: {self.objective} {v:.3f}")
        if (d := self.deflated) is not None:
            out.append(f"deflated Sharpe ratio {d.probability:.3f} over {d.trials} distinct strategies")
        if self.walk_forward is not None:
            out.append("walk-forward (unseen stretch per fold): "
                       + ", ".join("–" if v is None else f"{v:.3f}" for v in self.walk_forward.scores()))
        out += [f"Warning: {w}" for w in self.warnings()]
        return out

    def plugin_source(self, name: str) -> str:
        return plugin_source(self.program, name, self.notes())

    def algorithm(self, name: str = "evolved", register: bool = True):
        """The result as an algorithm instance; its class is registered under ``name`` like a loaded plugin,
        unless ``register`` is False."""
        return load_plugin(self.plugin_source(name), f"tradebot_evolved_{name}", register)()


@dataclass
class _Symbol:
    symbol: str
    df: pd.DataFrame
    close: np.ndarray
    ppy: float
    evaluator: Evaluator
    segments: list[tuple[int, int]]  #: the training part's stretches (candle positions)
    validation: tuple[int, int] | None
    test: tuple[int, int] | None

    @property
    def train_end(self) -> int:
        return self.segments[-1][1]


class Evolver:
    """Genetic programming of entry/exit rules on one or several symbols (``{symbol: frame}``)."""

    def __init__(
        self,
        df: pd.DataFrame | Mapping[str, pd.DataFrame],
        *,
        population: int = 300,
        generations: int = 40,
        max_depth: int | None = None,
        max_size: int | None = None,
        objective: str = "sharpe",
        aggregate: str = "mean",
        fee: float = 0.001,
        slippage: float = 0.0,
        validation: float = 0.2,
        test: float = 0.2,
        segments: int = 3,
        min_trades: int = 4,
        parsimony: float = 0.002,
        tournament: int = 4,
        crossover: float = 0.6,
        mutation: float = 0.35,
        elite: float = 0.02,
        hall_of_fame: int = 30,
        state: bool = True,
        indicators: list[str] | None = None,
        seed: int | None = None,
        align: bool = False,
        min_history: float = 0.5,
        dca_amount: float | None = None,
        primitives: PrimitiveSet | None = None,
        from_scratch: bool = False,
        library: bool | None = None,
        dedupe: bool | None = None,
        deflate: bool | None = None,
        library_every: int = 5,
        tune: int | None = None,
        islands: int | None = None,
        migrate_every: int = 10,
        migrants: int = 2,
    ):
        """``max_depth``/``max_size`` default to :data:`LIMITS`, from scratch to :data:`SCRATCH_LIMITS`.
        ``library`` (evolved building blocks every ``library_every`` generations), ``dedupe`` (programs
        trading alike count once) and ``deflate`` (the deflated Sharpe ratio of the result) default to
        ``from_scratch``. ``tune`` (default 0, from scratch 3): the best this many programs of every generation
        get their constants hill-climbed (a memetic step: GP finds the shape, local search the numbers).
        ``islands`` (default 1, from scratch :data:`SCRATCH_ISLANDS`): the population is split into that many
        islands that evolve apart; every ``migrate_every`` generations each sends its best ``migrants`` to the
        next one (a ring). Separate islands keep several shapes alive long enough for their constants to be
        tuned (the best ``tune`` programs are per island)."""
        if from_scratch and indicators is not None:
            raise ValueError("from scratch uses no ready-made indicators; leave out the indicator choice")
        limits = SCRATCH_LIMITS if from_scratch else LIMITS
        max_depth = limits[0] if max_depth is None else max_depth
        max_size = limits[1] if max_size is None else max_size
        if aggregate not in AGGREGATES:
            raise ValueError(f"unknown aggregate {aggregate!r}; choose {', '.join(AGGREGATES)}")
        if objective not in ("sharpe", "return", "calmar"):
            raise ValueError(f"unknown objective {objective!r} (sharpe, return, calmar)")
        if not (0 <= validation < 1 and 0 <= test < 1 and validation + test <= 0.8):
            raise ValueError("validation and test are fractions of the data and may add up to at most 0.8")
        if population < 4 or generations < 1 or max_depth < 2 or segments < 1:
            raise ValueError("need population >= 4, generations >= 1, max_depth >= 2 and segments >= 1")
        if crossover < 0 or mutation < 0 or crossover + mutation > 1:
            raise ValueError("crossover and mutation are probabilities that may add up to at most 1")
        self.population, self.generations = population, generations
        self.max_depth, self.max_size = max_depth, max(max_size, 4)
        self.objective, self.aggregate, self.fee, self.slippage = objective, aggregate, fee, slippage
        self.min_trades, self.parsimony = min_trades, parsimony
        self.tournament, self.p_crossover, self.p_mutation = max(1, tournament), crossover, mutation
        self.elite = elite
        self.hall_size = max(1, hall_of_fame)
        self.dca_amount = dca_amount
        self.seed = int(np.random.SeedSequence().generate_state(1)[0]) if seed is None else seed
        self.rng = np.random.default_rng(self.seed)
        self.from_scratch = from_scratch
        self.base_ps = primitives or (PrimitiveSet.from_scratch(state) if from_scratch
                                      else PrimitiveSet.default(indicators, state))
        self.ps = self.base_ps  #: grows with the library's modules
        library, dedupe, deflate = (from_scratch if x is None else x for x in (library, dedupe, deflate))
        self.library = Library() if library else None
        self.library_every = max(1, library_every)
        self.dedupe, self.deflate = dedupe, deflate
        self.tune = (3 if from_scratch else 0) if tune is None else max(0, tune)
        islands = (SCRATCH_ISLANDS if from_scratch else 1) if islands is None else islands
        self.islands = max(1, min(islands, population // 4))
        self.migrate_every, self.migrants = max(1, migrate_every), max(0, migrants)
        self.settings = dict(population=population, generations=generations, max_depth=max_depth, max_size=max_size,
                             objective=objective, aggregate=aggregate, fee=fee, slippage=slippage, validation=validation, test=test,
                             segments=segments, min_trades=min_trades, parsimony=parsimony, tournament=tournament,
                             crossover=crossover, mutation=mutation, elite=elite, hall_of_fame=hall_of_fame,
                             dca_amount=dca_amount, primitives=self.base_ps, from_scratch=from_scratch,
                             library=library, dedupe=dedupe, deflate=deflate, library_every=library_every,
                             tune=self.tune, islands=self.islands, migrate_every=migrate_every, migrants=migrants)

        self.multi = not isinstance(df, pd.DataFrame)
        frames, self.skipped = select_frames(df if self.multi else {"": df}, align, min_history)
        self.frames = frames
        timeline = reduce(lambda a, b: a.union(b), (f.index for f in frames.values()))
        n = len(timeline)
        test_at = n - int(round(n * test)) if test else n
        val_at = test_at - int(round(n * validation)) if validation else test_at
        self.test_start = timeline[test_at] if test_at < n else None
        self.validation_start = timeline[val_at] if val_at < test_at else None
        cuts = [timeline[int(round(val_at * i / segments))] for i in range(1, segments)]
        cache = max(300, 6000 // len(frames))
        self.data: list[_Symbol] = []
        for symbol, f in frames.items():
            pos = lambda ts, f=f: int(f.index.searchsorted(ts))  # noqa: E731
            v = pos(self.validation_start) if self.validation_start is not None else None
            t = pos(self.test_start) if self.test_start is not None else len(f)
            train_end = v if v is not None else t
            bounds = [0, *(pos(c) for c in cuts), train_end]
            segs = [(a, b) for a, b in zip(bounds, bounds[1:]) if b - a >= 2]
            if train_end < 2 * segments + 10 or not segs:
                raise ValueError(f"{symbol or 'the symbol'} has only {train_end} training candles; "
                                 "use more data, fewer segments or smaller validation/test parts")
            self.data.append(_Symbol(symbol, f, f["close"].to_numpy(dtype=float), infer_periods_per_year(f.index),
                                     Evaluator(OHLC.from_frame(f), cache),
                                     segs, (v, t) if v is not None and t - v >= 2 else None,
                                     (t, len(f)) if len(f) - t >= 2 else None))
        self.evaluations = 0
        self._fitness: dict[str, float] = {}
        self._validation: dict[str, float] = {}
        self._signature: dict[str, bytes] = {}  #: program -> fingerprint of its training positions
        self._sharpe: dict[bytes, float] = {}  #: per distinct behaviour: training Sharpe ratio per candle

    # -- scoring --------------------------------------------------------------------------------------
    def _objective(self, d: _Symbol, pos: np.ndarray, a: int, b: int) -> float:
        value = float(batch_fitness(simulate(d.close[a:b], pos[a:b], self.fee, self.slippage), d.ppy, self.objective))
        return value if np.isfinite(value) else REJECTED

    def _positions(self, d: _Symbol, program: Program) -> np.ndarray | None:
        try:
            return d.evaluator.positions(program)
        except Exception:  # e.g. a plugin indicator failing on some parameters
            log.debug("evolve: %s failed on %s", program, d.symbol, exc_info=True)
            return None

    def fitness(self, program: Program) -> float:
        """Objective per (symbol, training segment), combined by ``aggregate``, minus ``parsimony`` per node."""
        if (hit := self._fitness.get(program.key)) is not None:
            return hit
        scores = []
        trains = []
        for d in self.data:
            pos = self._positions(d, program)
            if pos is None:
                scores = None
                break
            train = pos[:d.train_end]
            trains.append(train)
            trades = (np.abs(np.diff(train, prepend=0.0)) > 0).sum() / 2
            if trades < self.min_trades:
                scores += [REJECTED] * len(d.segments)
            else:
                scores += [self._objective(d, pos, a, b) for a, b in d.segments]
        value = REJECTED if scores is None else float(AGGREGATES[self.aggregate](np.array(scores)[:, None])[0])
        if scores is not None and (self.dedupe or self.deflate):
            sig = hashlib.blake2b(b"".join(np.asarray(t, dtype=np.int8).tobytes() for t in trains),
                                  digest_size=16).digest()
            self._signature[program.key] = sig
            if self.deflate and value > REJECTED / 2 and sig not in self._sharpe:
                r = self._train_returns(program)
                self._sharpe[sig] = float(r.mean() / r.std()) if r.std() > 0 else float("nan")
        value = max(value, REJECTED) - self.parsimony * program.size
        self._fitness[program.key] = value
        self.evaluations += 1
        return value

    def _train_returns(self, program: Program) -> np.ndarray:
        """The training part's returns per candle, all symbols in a row."""
        return np.concatenate([simulate(d.close[:d.train_end], d.evaluator.positions(program)[:d.train_end], self.fee, self.slippage)
                               for d in self.data])

    def deflated(self, program: Program) -> Deflated | None:
        """The deflated Sharpe ratio of ``program`` over the distinct strategies tried so far (needs ``deflate``)."""
        if not self.deflate or not self._sharpe:
            return None
        r = self._train_returns(program)
        trials = np.array(list(self._sharpe.values()))
        probability, luck = deflated_sharpe(r, trials, len(trials))
        sr = float(r.mean() / r.std()) if r.std() > 0 else float("nan")
        return Deflated(probability, sr, luck, len(trials))

    def _distinct(self, pop: list[Program], fit: np.ndarray) -> np.ndarray:
        """With ``dedupe``: a program trading exactly like a fitter one (or an equally fit, earlier one) is
        set below every real score, so it counts once and its place goes to a new idea."""
        if not self.dedupe:
            return fit
        fit = fit.copy()
        first: set[bytes] = set()
        for i in np.argsort(-fit, kind="stable"):
            sig = self._signature.get(pop[i].key)
            if sig is None:
                continue
            if sig in first:
                fit[i] = 2 * REJECTED
            else:
                first.add(sig)
        return fit

    def validation(self, program: Program) -> float | None:
        """The objective on the validation part, combined over the symbols; None without one."""
        if not any(d.validation for d in self.data):
            return None
        if (hit := self._validation.get(program.key)) is None:
            scores = []
            for d in self.data:
                if d.validation is None:
                    continue
                pos = self._positions(d, program)
                scores.append(REJECTED if pos is None else self._objective(d, pos, *d.validation))
            hit = float(AGGREGATES[self.aggregate](np.array(scores)[:, None])[0])
            self._validation[program.key] = hit
        return hit

    # -- variation ------------------------------------------------------------------------------------
    def _fits(self, program: Program) -> bool:
        return program.depth <= self.max_depth and program.entry.size <= self.max_size and program.exit.size <= self.max_size

    def _new(self, depth: int, full: bool) -> Program:
        for _ in range(20):
            p = random_program(self.ps, depth, self.rng, full)
            if self._fits(p):
                return p
        return random_program(self.ps, 2, self.rng)

    def initial_population(self, size: int | None = None) -> list[Program]:
        """Ramped half-and-half: depths 2..max_depth (at most 6, deeper starts are bloat), half "full" and half
        "grow" trees, no duplicates; ``size`` defaults to the population."""
        size = size or self.population
        out: dict[str, Program] = {}
        depths = range(2, min(self.max_depth, 6) + 1)
        for i in range(size * 20):
            p = self._new(depths[i % len(depths)], full=bool((i // len(depths)) % 2))
            out.setdefault(p.key, p)
            if len(out) == size:
                break
        return list(out.values())

    def _pick(self, pop: list[Program], fit: np.ndarray) -> Program:
        idx = self.rng.integers(0, len(pop), self.tournament)
        return pop[int(idx[np.argmax(fit[idx])])]

    def _node(self, tree: Node, internal: float = 0.9):
        """A random ``(path, node)``: a primitive with probability ``internal`` (if any), else a constant."""
        nodes = list(tree.nodes())
        prims = [x for x in nodes if not x[1].is_const]
        consts = [x for x in nodes if x[1].is_const]
        pool = prims if (not consts or self.rng.random() < internal) else consts
        return pool[int(self.rng.integers(len(pool)))]

    def _with(self, program: Program, rule: str, tree: Node) -> Program:
        tree = simplify_tree(tree, self.ps)
        if self.library is not None:  # parts matching a module are written as the module
            tree = rewrite(tree, self.library.by_template)
        return Program(tree, program.exit) if rule == "entry" else Program(program.entry, tree)

    def crossover(self, a: Program, b: Program) -> Program:
        """A subtree of ``a`` replaced by a subtree of ``b`` of the same unit (or a whole rule taken from ``b``)."""
        if self.rng.random() < 0.1:
            return Program(a.entry, b.exit) if self.rng.random() < 0.5 else Program(b.entry, a.exit)
        for _ in range(5):
            rule = "entry" if self.rng.random() < 0.5 else "exit"
            ta, tb = getattr(a, rule), getattr(b, rule)
            path, node = self._node(ta)
            same = [n for _, n in tb.nodes() if n.is_const == node.is_const
                    and (n.spec == node.spec if node.is_const else n.unit == node.unit)]
            if not same:
                continue
            child = self._with(a, rule, ta.replace(path, same[int(self.rng.integers(len(same)))]))
            if self._fits(child):
                return child
        return a

    def mutate(self, program: Program) -> Program:
        """One random change: new subtree, swapped primitive, nudged constant or hoisted subtree."""
        for _ in range(5):
            rule = "entry" if self.rng.random() < 0.5 else "exit"
            tree = getattr(program, rule)
            kind = self.rng.choice(["subtree", "point", "constant", "hoist"], p=[0.35, 0.2, 0.3, 0.15])
            new = None
            if kind == "constant" and (consts := [x for x in tree.nodes() if x[1].is_const]):
                path, node = consts[int(self.rng.integers(len(consts)))]
                new = tree.replace(path, self._nudge(node))
            elif kind == "point":
                options = [(path, n, alts) for path, n in tree.nodes() if not n.is_const
                           and (alts := self.ps.same_signature(n.prim))]
                if options:
                    path, node, alts = options[int(self.rng.integers(len(options)))]
                    new = tree.replace(path, Node(alts[int(self.rng.integers(len(alts)))], node.args))
            elif kind == "hoist":
                options = [(path, sub) for path, n in tree.nodes() if not n.is_const
                           for sub_path, sub in n.nodes() if sub_path and not sub.is_const and sub.unit == n.unit]
                if options:
                    path, sub = options[int(self.rng.integers(len(options)))]
                    new = tree.replace(path, sub)
            if new is None:  # subtree, or the chosen kind didn't apply
                path, node = self._node(tree, internal=1.0)
                room = self.max_depth - len(path)
                try:
                    sub = random_tree(self.ps, node.unit, int(self.rng.integers(1, max(2, min(room, 4)) + 1)), self.rng,
                                      state=rule == "exit" and self.ps.state)
                except ValueError:  # the unit can't be built in that little room
                    continue
                new = tree.replace(path, sub)
            child = self._with(program, rule, new)
            if self._fits(child):
                return child
        return program

    def hill_climb(self, program: Program, fitness: float, tries: int = 8) -> tuple[Program, float]:
        """Up to ``tries`` constants (in random order), each moved one step and a bigger step either way;
        every improvement is kept."""
        for _ in range(tries):
            consts = constants(program)
            if not consts:
                break
            rule, path, node = consts[int(self.rng.integers(len(consts)))]
            s = node.spec
            if s.integer:
                steps = {node.value - 1, node.value + 1, int(round(node.value * 0.8)), int(round(node.value * 1.25))}
            else:
                step = (s.high - s.low) / 200
                steps = {node.value + k * step for k in (-5, -1, 1, 5)}
            for value in sorted(steps):
                tree = getattr(program, rule).replace(path, Node(None, value=quantize(s, value), spec=s))
                child = self._with(program, rule, tree)
                if child.key != program.key and self._fits(child) and (f := self.fitness(child)) > fitness:
                    program, fitness = child, f
                    break
        return program, fitness

    def _nudge(self, node: Node) -> Node:
        s = node.spec
        span = s.high - s.low
        if self.ps.scratch and log_scale(s):  # by a factor around 1 (14 -> 11 or 18), at least one step
            value = int(round(node.value * np.exp(self.rng.normal(0, 0.3))))
            value = value if value != node.value else node.value + int(self.rng.choice([-1, 1]))
        elif s.integer:
            step = int(round(self.rng.normal(0, max(1.0, 0.1 * span))))
            value = node.value + (step or int(self.rng.choice([-1, 1])))
        else:
            value = node.value + self.rng.normal(0, 0.1 * span)
        return Node(None, value=quantize(s, value), spec=s)

    def breed(self, pop: list[Program], fit: np.ndarray, n: int, seen: set[str]) -> list[Program]:
        children = []
        while len(children) < n:
            r = self.rng.random()
            parent = self._pick(pop, fit)
            if r < self.p_crossover:
                child = self.crossover(parent, self._pick(pop, fit))
            elif r < self.p_crossover + self.p_mutation:
                child = self.mutate(parent)
            else:
                child = parent
            for _ in range(3):  # keep the population diverse: duplicates are mutated again
                if child.key not in seen:
                    break
                child = self.mutate(child)
            seen.add(child.key)
            children.append(child)
        return children

    # -- the run --------------------------------------------------------------------------------------
    def run(self, progress: Progress | None = None) -> EvolutionResult:
        started = time.monotonic()
        size = max(4, self.population // self.islands)
        islands = []
        for _ in range(self.islands):
            pop = self.initial_population(size)
            islands.append((pop, np.array([self.fitness(p) for p in pop])))
        hall: dict = {}
        history = []
        for gen in range(self.generations):
            if self.library is not None and gen and gen % self.library_every == 0:
                islands = self._grow_library(islands)
            if self.islands > 1 and gen and gen % self.migrate_every == 0:
                islands = self._migrate(islands)
            islands = [self._rank(pop, fit) for pop, fit in islands]
            for pop, fit in islands:
                for p, f in zip(pop[: self.hall_size], fit[: self.hall_size]):
                    if f <= REJECTED / 2:
                        continue
                    # with dedupe, one entry per behaviour: the fittest (smallest) program trading that way
                    k = self._signature.get(p.key, p.key) if self.dedupe else p.key
                    if k not in hall or (self.dedupe and f > hall[k].fitness):
                        hall[k] = Candidate(p, float(f), gen)
            hall = dict(sorted(hall.items(), key=lambda kv: -kv[1].fitness)[: self.hall_size])
            top_pop, top_fit = max(islands, key=lambda isl: isl[1][0])
            everyone = [p for pop, _ in islands for p in pop]
            history.append({"generation": gen, "best": float(top_fit[0]), "validation": self.validation(top_pop[0]),
                            "size": float(np.mean([p.size for p in everyone])),
                            "distinct": len({p.key for p in everyone}),
                            **({"modules": len(self.library.modules)} if self.library is not None else {})})
            if progress:
                progress(gen, float(top_fit[0]), top_pop[0])
            if gen == self.generations - 1:
                break
            islands = [self._next(pop, fit) for pop, fit in islands]
        if hall:
            best = self._select(list(hall.values()))
        else:
            top_pop, top_fit = max(islands, key=lambda isl: isl[1][0])
            best = self._select([Candidate(top_pop[0], float(top_fit[0]), self.generations - 1)])
        return EvolutionResult(best.program, best.fitness, [self._report(d, best.program) for d in self.data], history,
                               sorted(hall.values(), key=lambda c: -c.fitness), self.objective, self.aggregate,
                               self.evaluations, time.monotonic() - started, self.seed, best.generation,
                               self.validation_start, self.test_start, self.skipped, from_scratch=self.from_scratch,
                               modules=self.library.modules if self.library is not None else [],
                               deflated=self.deflated(best.program), no_volume=self._no_volume(best.program))

    def _no_volume(self, program: Program) -> list[str]:
        """The symbols without volume data, if ``program`` reads the volume (#116)."""
        if not any(uses_volume(t) for t in (program.entry, program.exit)):
            return []
        return [d.symbol for d in self.data if not d.evaluator.ohlc.has_volume]

    def _rank(self, pop: list[Program], fit: np.ndarray) -> tuple[list[Program], np.ndarray]:
        """One island sorted best first, after deduplication and the memetic step."""
        fit = self._distinct(pop, fit)
        order = np.argsort(-fit, kind="stable")
        pop, fit = [pop[i] for i in order], fit[order]
        if self.tune:
            for i in range(min(self.tune, len(pop))):
                if fit[i] > REJECTED / 2:
                    pop[i], fit[i] = self.hill_climb(pop[i], float(fit[i]))
            order = np.argsort(-fit, kind="stable")
            pop, fit = [pop[i] for i in order], fit[order]
        return pop, fit

    def _next(self, pop: list[Program], fit: np.ndarray) -> tuple[list[Program], np.ndarray]:
        """The next generation of one (ranked) island: its elite plus bred children."""
        n_elite = max(1, int(round(len(pop) * self.elite)))
        elite = pop[:n_elite]
        children = self.breed(pop, fit, len(pop) - n_elite, {p.key for p in elite})
        return elite + children, np.concatenate([fit[:n_elite], [self.fitness(c) for c in children]])

    def _migrate(self, islands: list) -> list:
        """Ring migration: each island's best few replace the worst of the next island."""
        out = []
        for i, (pop, fit) in enumerate(islands):
            src_pop, src_fit = islands[i - 1]
            k = min(self.migrants, len(pop) // 2)
            order = np.argsort(-fit, kind="stable")
            pop, fit = [pop[j] for j in order], fit[order]
            best = np.argsort(-src_fit, kind="stable")[:k]
            out.append((pop[: len(pop) - k] + [src_pop[j] for j in best],
                        np.concatenate([fit[: len(pop) - k], src_fit[best]])))
        return out

    def _grow_library(self, islands: list) -> list:
        """Promote the subtrees shared by the best fifth of every island to modules and rewrite everyone."""
        good = [p for pop, fit in islands for p, f in list(zip(pop, fit))[: max(10, len(pop) // 5)] if f > REJECTED / 2]
        self.ps, new = self.library.grow(good, self.ps)
        if not new:
            return islands
        log.debug("evolve: new modules %s", ", ".join(m.name for m in new))
        out = []
        for pop, _ in islands:
            pop = [self.library.rewrite(p) for p in pop]
            out.append((pop, np.array([self.fitness(p) for p in pop])))
        return out

    def _select(self, hall: list[Candidate]) -> Candidate:
        """The hall-of-fame program that does best on validation (the training fitness breaks ties)."""
        for c in hall:
            c.validation = self.validation(c.program)
        if hall[0].validation is None:
            return max(hall, key=lambda c: c.fitness)
        return max(hall, key=lambda c: (c.validation, c.fitness))

    def _report(self, d: _Symbol, program: Program) -> SymbolReport:
        pos = pd.Series(d.evaluator.positions(program), index=d.df.index, name="position")
        name = "evolved"

        def part(bounds):
            if bounds is None:
                return None, None
            a, b = bounds
            df = d.df.iloc[a:b]
            res = from_positions(df, pos.iloc[a:b], name, {}, self.fee, periods_per_year=d.ppy, slippage=self.slippage)
            return res, baselines(df, res, self.fee, amount=self.dca_amount, periods_per_year=d.ppy,
                                    slippage=self.slippage)

        train, _ = part((0, d.train_end))
        validation, vb = part(d.validation)
        test, tb = part(d.test)
        return SymbolReport(d.symbol, train, validation, test, vb, tb)


def shuffle_candles(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """The same candles in random order: every move is kept (open vs. previous close, high/low/close vs. open),
    so volatility and the overall drift stay, but no pattern in time is left."""
    o, h, lo, c = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    prev = np.r_[o[0], c[:-1]]
    moves = np.stack([o / prev, h / o, lo / o, c / o], axis=1)[1:]
    order = rng.permutation(len(moves))
    moves = moves[order]
    opens, closes = [o[0]], [c[0]]
    highs, lows = [h[0]], [lo[0]]
    for gap, hi, low, close in moves:
        start = closes[-1] * gap
        opens.append(start)
        highs.append(start * hi)
        lows.append(start * low)
        closes.append(start * close)
    out = pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes}, index=df.index)
    if "volume" in df.columns:  # each candle keeps its volume (#116)
        v = df["volume"].to_numpy(dtype=float)
        out["volume"] = np.r_[v[:1], v[1:][order]]
    return out


def evolve(df: pd.DataFrame | Mapping[str, pd.DataFrame], *, permutations: int | None = None, folds: int | None = None,
           progress: Progress | None = None, on_phase: Callable[[str], None] | None = None, **kwargs) -> EvolutionResult:
    """Run :class:`Evolver` (``kwargs`` are its settings); with ``permutations``, also run the same search that
    many times on shuffled candles and attach the :class:`PermutationTest`; with ``folds``, run it once per
    walk-forward fold and attach the :class:`WalkForward`. Both default to 0, from scratch to
    :data:`SCRATCH_GUARDS`. ``on_phase(label)`` is called before each run ("evolving", "shuffled 1/5",
    "fold 2/3", ...)."""
    guards = SCRATCH_GUARDS if kwargs.get("from_scratch") else {}
    permutations = guards.get("permutations", 0) if permutations is None else permutations
    folds = guards.get("folds", 0) if folds is None else folds
    if on_phase:
        on_phase("evolving")
    evolver = Evolver(df, **kwargs)
    result = evolver.run(progress)
    part = result.unseen
    if permutations and part is not None:
        rng = np.random.default_rng(evolver.seed + 1)
        shuffled = []
        for i in range(permutations):
            if on_phase:
                on_phase(f"shuffled {i + 1}/{permutations}")
            frames = {s: shuffle_candles(f, rng) for s, f in evolver.frames.items()}
            settings = {**evolver.settings, "seed": int(rng.integers(1 << 31))}
            run = Evolver(frames if evolver.multi else frames[""], **settings).run(progress)
            shuffled.append(run.aggregated(part))
        result.permutation = PermutationTest(result.aggregated(part), shuffled, part)
    if folds:
        result.walk_forward = _walk_forward(evolver, folds, progress, on_phase)
    return result


def _walk_forward(evolver: Evolver, folds: int, progress: Progress | None,
                  on_phase: Callable[[str], None] | None, train_fraction: float = 0.7) -> WalkForward:
    """The search once per fold of :func:`tradebot.optimize.walkforward.plan` (rolling windows over all the data):
    each fold evolves on its window (with its own validation part) and is scored on the stretch after it."""
    from tradebot.optimize.walkforward import plan

    timeline = reduce(lambda a, b: a.union(b), (f.index for f in evolver.frames.values()))
    rng = np.random.default_rng(evolver.seed + 2)
    out = []
    for i, (start, split, end) in enumerate(plan(len(timeline), folds, train_fraction=train_fraction)):
        label = f"fold {i + 1}/{folds}"
        if on_phase:
            on_phase(label)
        lo, hi = timeline[start], timeline[end] if end < len(timeline) else None
        cut = {s: f[f.index.searchsorted(lo):len(f) if hi is None else f.index.searchsorted(hi)]
               for s, f in evolver.frames.items()}
        test = (end - split) / (end - start)
        validation = min(evolver.settings["validation"], 0.8 - test)
        settings = {**evolver.settings, "seed": int(rng.integers(1 << 31)), "test": test, "validation": validation}
        try:
            out.append(Evolver(cut if evolver.multi else cut[""], **settings).run(progress))
        except ValueError as e:
            raise ValueError(f"walk-forward {label}: {e}") from None
    return WalkForward(out)
