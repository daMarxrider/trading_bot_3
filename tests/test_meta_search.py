# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Meta-search (#90): combinations of optimizer settings, ranked by walk-forward, winner scored once on a holdout."""

import numpy as np
import pytest

from tradebot.algorithms import build_algorithm
from tradebot.optimize import meta_search
from tradebot.optimize.meta import DEFAULTS, MetaResult, Trial, auto_holdout, combinations, estimate, tuned_space

from .conftest import make_ohlc

FAST = dict(population=12, generations=2, backend="numpy", min_trades=0)


def test_space_drops_windows_that_dont_fit_and_adds_symbols_and_compute():
    assert tuned_space(1, train_length=300, warmup=100)["sample_length"] == [60, 120]
    assert tuned_space(1, train_length=100, warmup=100) == {"samples": [0]}
    assert tuned_space(4)["min_symbols_beating"] == [0, 1, 2, 3, 4]
    assert "min_symbols_beating" not in tuned_space(1) and "population" in tuned_space(1, compute=True)


def test_window_length_only_counts_with_random_windows():
    combos = combinations({"samples": [0, 8], "sample_length": [60, 120]}, DEFAULTS)
    assert len(combos) == 3  # samples 0 once, samples 8 with each length
    assert {c["sample_length"] for c in combos if not c["samples"]} == {DEFAULTS["sample_length"]}


def test_auto_holdout_is_one_test_stretch_and_estimate_counts_runs():
    assert auto_holdout(1000, 4, test_length=120) == 120
    assert auto_holdout(1000, 4, train_length=500) == 100
    assert estimate(10, 5, {}, DEFAULTS) == {"runs": 51, "evaluations": 51 * 256 * 30}
    assert estimate(10, 5, {}, DEFAULTS, "both")["runs"] == 102


def test_search_tries_yours_first_ranks_by_walk_forward_and_scores_the_winner_once_on_the_holdout():
    data = {f"S{i}": make_ohlc(1000, seed=30 + i) for i in range(2)}
    phases, seen, started = [], [], []
    res = meta_search(build_algorithm("sma_cross"), data, 4, 2, seed=1, on_phase=phases.append,
                      on_start=lambda space, n: started.append((space, n)),
                      on_trial=lambda t, n: seen.append((t.index, n)), **FAST)
    assert started == [(res.space, 4)] and seen == [(i, 4) for i in range(4)]
    assert res.trials[0].settings == {**DEFAULTS, "population": 12, "generations": 2}  # the caller's settings first
    assert len({tuple(sorted(t.settings.items())) for t in res.trials}) == 4
    ranked = res.ranked()
    assert ranked[0] is res.best and [t.score for t in ranked] == sorted((t.score for t in ranked), reverse=True)
    assert res.lead == pytest.approx(res.best.score - res.median)
    final = res.final
    assert final.folds == res.best.result.folds and final.holdout is not None
    assert final.folds[-1].end == final.holdout.split  # the folds never reach into the holdout
    assert final.holdout.end is None and len(final.holdout.result.symbols) == 2
    assert all(t.result.holdout is None for t in res.trials)  # no combination sees the holdout
    assert phases[0] == "combination 1/4 · fold 1/2" and phases[-1] == "winner · holdout"


def test_a_combination_that_cant_run_is_reported_not_fatal():
    res = meta_search(build_algorithm("sma_cross"), make_ohlc(800, seed=3), None, 2, seed=0,
                      space={"samples": [0, 4], "sample_length": [5000]}, **FAST)
    assert len(res.trials) == 2 and res.combinations == 2
    broken = [t for t in res.trials if t.error]
    assert len(broken) == 1 and broken[0].settings["samples"] == 4 and broken[0].score is None
    assert res.best is not None and res.best.error is None


def test_stop_and_time_limit_end_early_but_still_score_the_holdout():
    algo, df, calls = build_algorithm("sma_cross"), make_ohlc(800, seed=4), []
    stopped = meta_search(algo, df, 5, 2, seed=0, space={"min_symbols_beating": [0, 1]}, stop=lambda: len(calls) >= 1,
                          on_trial=lambda t, n: calls.append(t), **FAST)
    assert stopped.stopped == "stopped" and len(stopped.trials) == 1 and stopped.final.holdout is not None
    timed = meta_search(algo, df, 5, 2, seed=0, time_limit=0, **FAST)
    assert timed.stopped == "time" and timed.trials == [] and timed.final is None
    assert timed.warnings() == ["No combination produced an out-of-sample score."]


def test_bad_arguments_say_why():
    df = make_ohlc(600, seed=5)
    with pytest.raises(ValueError, match="holdout"):
        meta_search(build_algorithm("sma_cross"), df, 2, 2, holdout=0, **FAST)
    with pytest.raises(ValueError, match="can't tune"):
        meta_search(build_algorithm("sma_cross"), df, 2, 2, space={"objective": ["calmar"]}, **FAST)


class _WF:
    """Just enough of a WalkForwardResult for the meta-search statistics."""

    def __init__(self, scores, holdout=None):
        self.scores, self.mean = np.array(scores, dtype=float), float(np.mean(scores))
        self.holdout = type("H", (), {"out_of_sample": holdout})()


def meta(winner, holdout):
    scores = ([0.1] * 3, [0.0] * 3, [-0.1] * 3, winner)
    return MetaResult([Trial(i, {}, _WF(s), 0.0) for i, s in enumerate(scores)], _WF(winner, holdout), {}, 4, 100)


def test_flags_a_winner_within_noise_and_one_that_does_worse_on_the_holdout():
    clear = meta([1.0, 1.1, 0.9], holdout=1.0)
    assert clear.best.index == 3 and clear.warnings() == []
    text = " ".join(meta([3.0, -1.0, 0.5], holdout=-0.5).warnings())
    assert "fold-to-fold noise" in text and "below the median combination" in text


def test_cli_meta(monkeypatch):
    from typer.testing import CliRunner

    import tradebot.cli.main as cli

    monkeypatch.setattr(cli, "load_data", lambda s, *a, **k: make_ohlc(900, seed=len(s)))
    res = CliRunner().invoke(cli.app, ["optimize", "A,BB", "sma_cross", "--meta", "3", "--walk-forward", "2",
                                       "--population", "12", "--generations", "2", "--min-trades", "0",
                                       "--backend", "numpy", "--seed", "0"], env={"COLUMNS": "200"})
    assert res.exit_code == 0, res.output
    for text in ("3 combinations × 2 folds", "meta-search: 3 of", "winning settings:", "holdout",
                 "scored once", "tradebot backtest A,BB sma_cross"):
        assert text in res.output
    assert "(yours)" in res.output or "★" in res.output  # the first row is yours, or the winner if it beat the rest
    bad = CliRunner().invoke(cli.app, ["optimize", "A", "sma_cross", "--meta", "2", "--walk-forward", "500"])
    assert bad.exit_code != 0 and "folds don't fit" in bad.output
