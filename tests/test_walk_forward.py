# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Walk-forward validation (#89): fold layout, date splits, per-fold and combined results, flags."""

import pytest

from tradebot.algorithms import build_algorithm
from tradebot.optimize import GeneticOptimizer, walk_forward
from tradebot.optimize.walkforward import Fold, WalkForwardResult, plan

from .conftest import make_ohlc

FAST = dict(population=16, generations=2, seed=0, backend="numpy", min_trades=0)


def test_rolling_folds_follow_each_other_and_end_at_the_last_candle():
    windows = plan(1000, 4, train_length=300, test_length=150)
    assert windows == [(100, 400, 550), (250, 550, 700), (400, 700, 850), (550, 850, 1000)]
    assert all(split - start == 300 for start, split, _ in windows)


def test_anchored_folds_grow_from_the_first_candle():
    assert [w[0] for w in plan(1000, 4, train_length=300, test_length=150, anchored=True)] == [100] * 4


def test_holdout_is_last_and_untouched_by_the_folds():
    *folds, holdout = plan(1000, 3, train_length=200, test_length=100, holdout=100)
    assert holdout == (700, 900, 1000)
    assert max(end for _, _, end in folds) == 900


def test_default_lengths_fill_the_data_with_the_train_ratio():
    windows = plan(1000, 5, train_fraction=0.7)
    (start, split, end), last = windows[0], windows[-1]
    assert last[2] == 1000 and start >= 0
    assert (split - start) / (split - start + end - split) == pytest.approx(0.7, abs=0.02)


@pytest.mark.parametrize("kw", [dict(folds=0), dict(folds=500), dict(folds=3, train_length=900, test_length=100),
                                dict(folds=2, holdout=1000)])
def test_impossible_layouts_say_why(kw):
    with pytest.raises(ValueError):
        plan(1000, **kw)


def test_split_at_cuts_every_symbol_at_the_same_date():
    data = {"A": make_ohlc(400, seed=1), "B": make_ohlc(300, seed=2)}
    at = data["A"].index[250]
    opt = GeneticOptimizer(build_algorithm("sma_cross"), data, split_at=at, **FAST)
    assert [len(d.train_df) for d in opt.data] == [250, 250]
    assert len(opt.data[0].test_df) == 150 and len(opt.data[1].test_df) == 50


def test_each_fold_tunes_on_its_window_and_scores_the_next():
    df = make_ohlc(900, seed=7)
    res = walk_forward(build_algorithm("sma_cross"), df, 3, train_length=300, test_length=150, holdout=100, **FAST)
    assert len(res.folds) == 3 and res.holdout is not None and res.holdout.holdout
    for f in res.folds:
        assert len(f.result.train.equity) == 300 and len(f.result.test.equity) == 150
        assert f.result.test.equity.index[0] == f.split
    assert res.holdout.result.test.equity.index[-1] == df.index[-1]
    assert res.folds[-1].end == res.holdout.split  # no fold reaches into the holdout
    assert len(res.scores) == 3 and res.mean == pytest.approx(res.scores.mean())
    assert set(res.beat_counts()) == {"buy & hold", "monthly investing"}
    assert set(res.stability()) == {"fast", "slow"} and all(0 <= v <= 1 for v in res.stability().values())


@pytest.mark.parametrize("method", ["pso", "both"])
def test_several_symbols_and_methods(method):
    data = {f"S{i}": make_ohlc(700, seed=20 + i) for i in range(3)}
    phases = []
    res = walk_forward(build_algorithm("rsi_threshold"), data, 2, method=method, on_phase=phases.append, **FAST)
    assert len(res.folds) == 2 and all(len(f.result.symbols) == 3 for f in res.folds)
    assert all((f.comparison is not None) == (method == "both") for f in res.folds)
    assert phases == (["fold 1/2 GA", "fold 1/2 PSO", "fold 2/2 GA", "fold 2/2 PSO"] if method == "both"
                      else ["fold 1/2", "fold 2/2"])
    summary = res.folds[0].summary()
    assert summary["beats"]["buy & hold"][1] == 3


class _Result:
    """Just enough of an OptimizationResult for the combined statistics."""

    def __init__(self, out, params, beats=True):
        self.out, self.best_params, self.symbols = out, params, []
        self.test_baselines = type("B", (), {"not_beaten": [] if beats else ["buy & hold"], "strategy_xirr": 0.0,
                                             "excess": {"buy & hold": 0.0, "monthly investing": 0.0}})()
        self.test = type("T", (), {"trades": []})()

    def aggregated(self, period):
        return self.out


def fake(outs, params, beats):
    space = build_algorithm("sma_cross").param_space()
    folds = [Fold(i, None, None, None, _Result(o, p, b)) for i, (o, p, b) in enumerate(zip(outs, params, beats))]
    return WalkForwardResult(folds, None, space)


def test_flags_a_score_that_swings_between_folds_and_rarely_beats_buy_and_hold():
    steady = fake([1.0, 1.1, 0.9], [{"fast": 10, "slow": 50}] * 3, [True] * 3)
    assert not steady.unstable and not steady.rarely_beats and steady.warnings() == []
    lucky = fake([3.0, -1.0, -1.2], [{"fast": 2, "slow": 50}, {"fast": 100, "slow": 50}, {"fast": 50, "slow": 50}],
                 [True, False, False])
    assert lucky.unstable and lucky.rarely_beats
    assert lucky.stability()["fast"] == 1.0 and lucky.stability()["slow"] == 0.0
    text = " ".join(lucky.warnings())
    assert "varies more between folds" in text and "1 of 3 folds" in text and "fast" in text


def test_cli_walk_forward(monkeypatch):
    from typer.testing import CliRunner

    import tradebot.cli.main as cli

    monkeypatch.setattr(cli, "load_data", lambda *a, **k: make_ohlc(800, seed=11))
    res = CliRunner().invoke(cli.app, ["optimize", "X", "sma_cross", "--walk-forward", "3", "--holdout", "100",
                                       "--population", "12", "--generations", "2", "--min-trades", "0",
                                       "--backend", "numpy", "--seed", "0"], env={"COLUMNS": "200"})
    assert res.exit_code == 0, res.output
    assert "walk-forward" in res.output and "holdout" in res.output and "parameters per fold" in res.output
    assert "tradebot backtest X sma_cross" in res.output
    bad = CliRunner().invoke(cli.app, ["optimize", "X", "sma_cross", "--walk-forward", "500"])
    assert bad.exit_code != 0 and "folds don't fit" in bad.output
