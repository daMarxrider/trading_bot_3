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

Indicators are computed over the whole history and only *scored* per part: all primitives are causal,
so this equals computing them on each part with a warm-up, and a position open at a boundary carries
over as it would in live trading.
"""

from __future__ import annotations

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
from tradebot.backtest.metrics import infer_periods_per_year
from tradebot.core.series import OHLC
from tradebot.evolve.export import load_plugin, plugin_source
from tradebot.evolve.primitives import BOOL, PrimitiveSet
from tradebot.evolve.program import Evaluator, Node, Program, quantize, random_program, random_tree, simplify_tree
from tradebot.optimize.base import AGGREGATES, carried_by_few, objective_value, select_frames

log = logging.getLogger(__name__)

REJECTED = -1e9
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
            out.append(f"On shuffled prices (no patterns to find) the evolver scored as well in {k} of "
                       f"{len(pt.shuffled)} runs (p = {pt.p:.2f}): no evidence of a real edge.")
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
        max_depth: int = 6,
        max_size: int = 40,
        objective: str = "sharpe",
        aggregate: str = "mean",
        fee: float = 0.001,
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
    ):
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
        self.objective, self.aggregate, self.fee = objective, aggregate, fee
        self.min_trades, self.parsimony = min_trades, parsimony
        self.tournament, self.p_crossover, self.p_mutation = max(1, tournament), crossover, mutation
        self.n_elite = max(1, int(round(population * elite)))
        self.hall_size = max(1, hall_of_fame)
        self.dca_amount = dca_amount
        self.seed = int(np.random.SeedSequence().generate_state(1)[0]) if seed is None else seed
        self.rng = np.random.default_rng(self.seed)
        self.ps = primitives or PrimitiveSet.default(indicators, state)
        self.settings = dict(population=population, generations=generations, max_depth=max_depth, max_size=max_size,
                             objective=objective, aggregate=aggregate, fee=fee, validation=validation, test=test,
                             segments=segments, min_trades=min_trades, parsimony=parsimony, tournament=tournament,
                             crossover=crossover, mutation=mutation, elite=elite, hall_of_fame=hall_of_fame,
                             dca_amount=dca_amount, primitives=self.ps)

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

    # -- scoring --------------------------------------------------------------------------------------
    def _objective(self, d: _Symbol, pos: np.ndarray, a: int, b: int) -> float:
        value = float(batch_fitness(simulate(d.close[a:b], pos[a:b], self.fee), d.ppy, self.objective))
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
        for d in self.data:
            pos = self._positions(d, program)
            if pos is None:
                scores = None
                break
            train = pos[:d.train_end]
            trades = (np.abs(np.diff(train, prepend=0.0)) > 0).sum() / 2
            if trades < self.min_trades:
                scores += [REJECTED] * len(d.segments)
            else:
                scores += [self._objective(d, pos, a, b) for a, b in d.segments]
        value = REJECTED if scores is None else float(AGGREGATES[self.aggregate](np.array(scores)[:, None])[0])
        value = max(value, REJECTED) - self.parsimony * program.size
        self._fitness[program.key] = value
        self.evaluations += 1
        return value

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

    def initial_population(self) -> list[Program]:
        """Ramped half-and-half: depths 2..max_depth, half "full" and half "grow" trees, no duplicates."""
        out: dict[str, Program] = {}
        depths = range(2, self.max_depth + 1)
        for i in range(self.population * 20):
            p = self._new(depths[i % len(depths)], full=bool((i // len(depths)) % 2))
            out.setdefault(p.key, p)
            if len(out) == self.population:
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

    def _nudge(self, node: Node) -> Node:
        s = node.spec
        span = s.high - s.low
        if s.integer:
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
        pop = self.initial_population()
        fit = np.array([self.fitness(p) for p in pop])
        hall: dict[str, Candidate] = {}
        history = []
        for gen in range(self.generations):
            order = np.argsort(-fit, kind="stable")
            pop, fit = [pop[i] for i in order], fit[order]
            for p, f in zip(pop[: self.hall_size], fit[: self.hall_size]):
                if f > REJECTED / 2 and p.key not in hall:
                    hall[p.key] = Candidate(p, float(f), gen)
            hall = dict(sorted(hall.items(), key=lambda kv: -kv[1].fitness)[: self.hall_size])
            history.append({"generation": gen, "best": float(fit[0]), "validation": self.validation(pop[0]),
                            "size": float(np.mean([p.size for p in pop])), "distinct": len({p.key for p in pop})})
            if progress:
                progress(gen, float(fit[0]), pop[0])
            if gen == self.generations - 1:
                break
            elite = pop[: self.n_elite]
            children = self.breed(pop, fit, self.population - len(elite), {p.key for p in elite})
            pop = elite + children
            fit = np.concatenate([fit[: len(elite)], [self.fitness(c) for c in children]])
        best = self._select(list(hall.values()) or [Candidate(pop[0], float(fit[0]), self.generations - 1)])
        return EvolutionResult(best.program, best.fitness, [self._report(d, best.program) for d in self.data], history,
                               sorted(hall.values(), key=lambda c: -c.fitness), self.objective, self.aggregate,
                               self.evaluations, time.monotonic() - started, self.seed, best.generation,
                               self.validation_start, self.test_start, self.skipped)

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
            res = from_positions(df, pos.iloc[a:b], name, {}, self.fee, periods_per_year=d.ppy)
            return res, baselines(df, res, self.fee, amount=self.dca_amount, periods_per_year=d.ppy)

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
    moves = moves[rng.permutation(len(moves))]
    opens, closes = [o[0]], [c[0]]
    highs, lows = [h[0]], [lo[0]]
    for gap, hi, low, close in moves:
        start = closes[-1] * gap
        opens.append(start)
        highs.append(start * hi)
        lows.append(start * low)
        closes.append(start * close)
    out = pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes}, index=df.index)
    return out


def evolve(df: pd.DataFrame | Mapping[str, pd.DataFrame], *, permutations: int = 0, progress: Progress | None = None,
           on_phase: Callable[[str], None] | None = None, **kwargs) -> EvolutionResult:
    """Run :class:`Evolver` (``kwargs`` are its settings); with ``permutations``, also run the same search that
    many times on shuffled candles and attach the :class:`PermutationTest`. ``on_phase(label)`` is called before
    each run ("evolving", "shuffled 1/5", ...)."""
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
    return result
