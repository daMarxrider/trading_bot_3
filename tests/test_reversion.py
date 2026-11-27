# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""The chance that a pair's gap closes before the stop (#223): the Feynman–Kac table against independent answers,
the rolling reversion speed, and the entry filter in the backtest, the live bot and the command line."""

import math
import re

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from tradebot.analysis.reversion import check_filter, probability, reversion, reversion_speed, table
from tradebot.backtest.pairs import next_side, pair_backtest

from .test_pairs import StubData, frame_of, stub_broker, synthetic_pair

EXIT, STOP = 0.5, 4.0


def ou_path(kappa: float, n: int, seed: int) -> pd.Series:
    """An exact Ornstein–Uhlenbeck path with long-run spread 1: z[t] = e^-κ z[t-1] + √(1 - e^-2κ) ε."""
    rng = np.random.default_rng(seed)
    z = np.zeros(n)
    a, b = math.exp(-kappa), math.sqrt(1 - math.exp(-2 * kappa))
    for t in range(1, n):
        z[t] = a * z[t - 1] + b * rng.normal()
    return pd.Series(z)


def simulated_win_rate(x0: float, s_end: float, dt: float, paths: int = 20_000, seed: int = 1) -> float:
    """Monte Carlo of dx = -x ds + √2 dW from x0: share of paths reaching EXIT before STOP by s_end."""
    rng = np.random.default_rng(seed)
    x = np.full(paths, x0)
    done = np.zeros(paths, bool)
    won = np.zeros(paths, bool)
    for _ in range(int(round(s_end / dt))):
        alive = ~done
        x[alive] += -x[alive] * dt + math.sqrt(2 * dt) * rng.standard_normal(alive.sum())
        hit_exit, hit_stop = alive & (x <= EXIT), alive & (x >= STOP)
        won |= hit_exit
        done |= hit_exit | hit_stop
    return float(won.mean())


# -- the table against independent answers ---------------------------------------------------------------------
@pytest.mark.parametrize("x, closed_form", [
    # (S(4) - S(x)) / (S(4) - S(0.5)) with S' = exp(x²/2), by quadrature: the answer for an unlimited horizon
    (1.0, 0.99922),
    (2.0, 0.99478),
    (3.0, 0.95666),
])
def test_without_a_time_limit_it_is_the_closed_form(x, closed_form):
    assert probability(x, 1.0, 1000, EXIT, STOP) == pytest.approx(closed_form, abs=2e-4)


def test_with_a_time_limit_it_matches_the_converged_simulation():
    # κ·N = 0.5 from |z| = 2: Monte Carlo at steps 1e-3, 2.5e-4, 6.25e-5 gave 0.3166, 0.3232, 0.3267, closing in on
    # 0.333 as √step (barriers checked between steps miss crossings); a finer grid of the equation gives 0.3332
    assert probability(2.0, 0.05, 10, EXIT, STOP) == pytest.approx(0.3332, abs=1e-3)


def test_a_seeded_simulation_lands_just_below_the_table():
    # missed crossings make the simulation's barriers a little wider: below the exact value, by less than 0.03
    exact = probability(3.0, 0.1, 10, EXIT, STOP)  # κ·N = 1
    simulated = simulated_win_rate(3.0, 1.0, dt=2.5e-4)
    assert exact - 0.03 < simulated < exact


def test_close_to_the_exit_with_little_time_it_matches_the_fine_grid():
    # κ·N = 0.05 from |z| = 0.6: 0.7739 on a 4x finer time grid, 0.7695 by Monte Carlo (biased low, see above).
    # Crank–Nicolson gave 1.29 here: a "probability" above 1.
    assert probability(0.6, 0.005, 10, EXIT, STOP) == pytest.approx(0.774, abs=0.005)


SETTINGS = [(0.5, 4.0), (0.0, 2.0), (1.0, 1.5)]  # the default trade, an exit at 0, a narrow band


@pytest.mark.parametrize("exit, stop", SETTINGS)
def test_every_value_of_the_table_is_a_probability(exit, stop):
    _, _, u = table(exit, stop)
    assert u.min() >= 0.0 and u.max() <= 1.0


@pytest.mark.parametrize("exit, stop", SETTINGS)
def test_more_time_never_lowers_the_chance_anywhere(exit, stop):
    _, _, u = table(exit, stop)
    assert (np.diff(u, axis=0) >= -1e-12).all()


@pytest.mark.parametrize("exit, stop", SETTINGS)
def test_a_wider_gap_never_raises_the_chance_anywhere(exit, stop):
    _, _, u = table(exit, stop)
    assert (np.diff(u, axis=1) <= 1e-12).all()


def test_the_table_is_solved_once_per_exit_and_stop():
    assert table(EXIT, STOP) is table(EXIT, STOP)


# -- properties every probability has ------------------------------------------------------------------------
def test_inside_the_exit_band_it_is_certain():
    assert probability(0.3, 0.1, 10, EXIT, STOP) == 1.0


def test_at_the_stop_it_is_lost():
    assert probability(4.0, 0.1, 10, EXIT, STOP) == 0.0


def test_a_short_gap_and_a_long_gap_are_the_same_bet():
    assert probability(-2.5, 0.1, 10, EXIT, STOP) == probability(2.5, 0.1, 10, EXIT, STOP)


@pytest.mark.parametrize("shorter, longer", [(1, 5), (5, 20), (20, 100)])
def test_more_time_means_a_higher_chance(shorter, longer):
    assert probability(2.5, 0.1, shorter, EXIT, STOP) < probability(2.5, 0.1, longer, EXIT, STOP)


def test_a_wider_gap_means_a_lower_chance():
    chances = probability(np.array([1.0, 2.0, 3.0, 3.9]), 0.1, 20, EXIT, STOP)
    assert np.all(np.diff(chances) < 0)


def test_only_speed_times_horizon_matters():
    # a pair twice as fast over half the candles is the same bet
    assert probability(2.5, 0.2, 10, EXIT, STOP) == pytest.approx(probability(2.5, 0.1, 20, EXIT, STOP))


@pytest.mark.parametrize("z, kappa", [(np.nan, 0.1), (2.0, np.nan), (2.0, 0.0), (2.0, -0.1)])
def test_unknown_input_or_a_gap_that_does_not_revert_is_unknown(z, kappa):
    assert math.isnan(probability(z, kappa, 10, EXIT, STOP))


@pytest.mark.parametrize("horizon", [0, -5, np.nan])
def test_a_horizon_below_one_candle_is_refused(horizon):
    with pytest.raises(ValueError, match="horizon"):
        probability(2.0, 0.1, horizon, EXIT, STOP)


@pytest.mark.parametrize("exit, stop", [(1.0, 1.0), (2.0, 1.0), (-0.5, 4.0)])
def test_an_exit_band_not_inside_the_stop_is_refused(exit, stop):
    with pytest.raises(ValueError, match="exit < stop"):
        probability(1.5, 0.1, 10, exit, stop)


# -- the rolling reversion speed -----------------------------------------------------------------------------------
@pytest.mark.parametrize("kappa, window", [(0.05, 250), (0.1, 120)])
def test_the_half_life_is_found_without_bias(kappa, window):
    # the AR(1) fit alone is biased (13.9 candles came out as 10.9); with Kendall's correction it is not
    estimates = [reversion(ou_path(kappa, 4000, seed), 30, EXIT, STOP, window)["half_life"].median()
                 for seed in range(5)]
    assert np.median(estimates) == pytest.approx(math.log(2) / kappa, rel=0.1)


def test_a_random_walk_mostly_does_not_revert():
    walk = pd.Series(np.cumsum(np.random.default_rng(3).normal(size=3000)))
    assert reversion_speed(walk, 120).iloc[120:].isna().mean() > 0.5


def test_the_speed_of_a_bar_never_sees_that_bar():
    z = ou_path(0.1, 400, seed=4)
    changed = z.copy()
    changed.iloc[-1] = 50.0
    before = reversion_speed(z, 120).iloc[-1]
    assert not math.isnan(before) and reversion_speed(changed, 120).iloc[-1] == before


@pytest.mark.parametrize("half_life, window, unknown_at_most", [(7, 120, 0.01), (14, 120, 0.1)])
def test_a_reverting_gap_is_rarely_unknown_over_twice_the_window(half_life, window, unknown_at_most):
    # over 60 candles 6 % (half-life 7) and 20 % (14) of a reverting gap's candles came out "doesn't revert"
    speed = reversion_speed(ou_path(math.log(2) / half_life, 3000, seed=0), window)
    assert speed.iloc[window + 1:].isna().mean() <= unknown_at_most


def test_the_speed_is_unknown_until_the_window_is_full():
    assert reversion_speed(ou_path(0.1, 100, seed=5), 60).iloc[:61].isna().all()


def test_a_window_below_three_candles_is_refused():
    with pytest.raises(ValueError, match="at least 3 candles"):
        reversion_speed(ou_path(0.1, 100, seed=5), 2)


# -- the filter settings -------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("p, n", [(0, 1), (1, 30), ("0.6", "20")])
def test_filter_settings_that_work(p, n):
    assert check_filter(p, n) == (float(p), float(n))


@pytest.mark.parametrize("p, n, reason", [
    (-0.1, 30, "minimum probability"), (1.5, 30, "minimum probability"), (np.nan, 30, "minimum probability"),
    (0.5, 0, "horizon"), (0.5, np.inf, "horizon"), ("abc", 30, "must be numbers"),
])
def test_filter_settings_that_cant_work(p, n, reason):
    with pytest.raises(ValueError, match=reason):
        check_filter(p, n)


# -- the entry rule ----------------------------------------------------------------------------------------------------
def test_without_the_filter_the_probability_is_ignored():
    assert next_side(0.0, 2.5, 1.0, 2.0, EXIT, STOP, probability=0.01) == -1.0


def test_the_filter_blocks_an_unlikely_entry():
    assert next_side(0.0, 2.5, 1.0, 2.0, EXIT, STOP, probability=0.5, min_probability=0.6) == 0.0


def test_the_filter_lets_a_likely_entry_through():
    assert next_side(0.0, -2.5, 1.0, 2.0, EXIT, STOP, probability=0.6, min_probability=0.6) == 1.0


def test_an_unknown_probability_never_opens_with_the_filter_on():
    assert next_side(0.0, 2.5, 1.0, 2.0, EXIT, STOP, probability=float("nan"), min_probability=0.1) == 0.0


def test_the_filter_never_closes_an_open_trade():
    assert next_side(-1.0, 2.5, 1.0, 2.0, EXIT, STOP, probability=0.0, min_probability=0.9) == -1.0


# -- the backtest --------------------------------------------------------------------------------------------------------
def backtest(**kw):
    a, b = synthetic_pair(n=900, seed=5)
    return pair_backtest(frame_of(a), frame_of(b), window=60, **kw)


def entries(res) -> pd.DatetimeIndex:
    return res.positions.index[(res.positions != 0) & (res.positions.shift(1, fill_value=0) == 0)]


def test_every_filtered_entry_had_at_least_the_minimum_chance():
    from tradebot.analysis.pairs import rolling_pair

    a, b = synthetic_pair(n=900, seed=5)
    res = pair_backtest(frame_of(a), frame_of(b), window=60, min_probability=0.8, horizon=30)
    chance = reversion(rolling_pair(a, b, 60).zscore, 30, EXIT, STOP, 120)["probability"]
    assert len(entries(res)) > 0 and (chance.loc[entries(res)] >= 0.8).all()


def test_the_filter_skips_some_entries_the_plain_rule_makes():
    assert len(entries(backtest(min_probability=0.8, horizon=30))) < len(entries(backtest()))


def test_a_filter_of_certainty_never_opens():
    # strictly between the exit band and the stop the chance is always below 1
    assert backtest(min_probability=1.0).metrics["trades"] == 0


def test_the_filter_settings_are_recorded():
    assert {"min_probability": 0.7, "horizon": 20.0}.items() <= backtest(min_probability=0.7, horizon=20).params.items()


def test_the_filtered_backtest_has_no_look_ahead():
    a, b = synthetic_pair(n=900, seed=5)
    full = pair_backtest(frame_of(a), frame_of(b), window=60, min_probability=0.7)
    cut = pair_backtest(frame_of(a.iloc[:700]), frame_of(b.iloc[:700]), window=60, min_probability=0.7)
    pd.testing.assert_series_equal(full.positions.iloc[:700], cut.positions)


@pytest.mark.parametrize("kw, reason", [({"min_probability": 2}, "minimum probability"), ({"horizon": 0}, "horizon")])
def test_the_backtest_refuses_filter_settings_that_cant_work(kw, reason):
    with pytest.raises(ValueError, match=reason):
        backtest(**kw)


# -- the live pair bot -----------------------------------------------------------------------------------------------------
def run_bot(**kw):
    from tradebot.live import PairBot

    a, b = synthetic_pair(n=700, seed=11)
    data = StubData(a, b)
    bot = PairBot(stub_broker(data), ("AAA", "BBB"), "1d", window=60, **kw)
    for k in range(100, len(a) + 1):
        data.upto = k
        bot.step()
    return bot


def test_the_bot_reports_the_chance_with_every_candle_once_it_is_known():
    bot = run_bot()
    assert not math.isnan(bot.events[-1].probability)


def test_the_bot_follows_the_filtered_backtest():
    a, b = synthetic_pair(n=700, seed=11)
    bot = run_bot(min_probability=0.7)
    ref = pair_backtest(frame_of(a), frame_of(b), window=60, min_probability=0.7).positions
    assert bot.side == ref.iloc[-1]


def test_the_bot_says_when_the_filter_kept_it_out():
    notes = [e.note for e in run_bot(min_probability=0.99).events if e.note.startswith("not opened")]
    assert any(re.fullmatch(r"not opened: the gap closes before the stop with \d+\.\d% chance in 30 candles, "
                            r"below 99\.0%", n) for n in notes)


def test_the_bot_refuses_filter_settings_that_cant_work():
    from tradebot.live import PairBot

    with pytest.raises(ValueError, match="minimum probability"):
        PairBot(None, ("AAA", "BBB"), "1d", min_probability=-0.5)


# -- the command line ------------------------------------------------------------------------------------------------------
def plain(result) -> str:
    return " ".join(re.sub(r"\x1b\[[0-9;]*m", "", result.output).replace("│", " ").split())


@pytest.fixture
def pair_data(monkeypatch):
    from tradebot.cli import main

    a, b = synthetic_pair(n=900, seed=5)
    frames = {"AAA": frame_of(a), "BBB": frame_of(b)}
    monkeypatch.setattr(main, "load_data", lambda symbol, *args, **kw: frames[symbol])
    return main


def test_the_pair_command_shows_the_half_life_now_and_the_chance(pair_data):
    result = CliRunner().invoke(pair_data.app, ["pair", "AAA:BBB", "--horizon", "20"])
    assert result.exit_code == 0, result.output
    text = plain(result)
    assert re.search(r"half-life now \(last 120 candles\) \d+\.\d candles", text)
    assert re.search(r"before ±4σ in 20 candles (\d+%|inside the exit band)", text)


@pytest.mark.parametrize("args, reason", [
    (["pair-backtest", "AAA:BBB", "--min-probability", "1.5"], "minimum probability"),
    (["pair-backtest", "AAA:BBB", "--horizon", "0"], "horizon"),
    (["pair", "AAA:BBB", "--exit", "5"], "need 0 <= --exit < --stop"),
])
def test_the_command_line_refuses_filter_settings_that_cant_work(pair_data, args, reason):
    result = CliRunner().invoke(pair_data.app, args)
    assert result.exit_code == 2 and reason in plain(result)


def test_pair_run_refuses_a_wrong_filter_before_connecting(pair_data, monkeypatch):
    def must_not_run(*_a, **_k):
        raise AssertionError("refused too late")

    monkeypatch.setattr(pair_data, "create_broker", must_not_run)
    result = CliRunner().invoke(pair_data.app, ["bot", "pair-run", "AAA", "BBB", "--min-probability", "-1"])
    assert result.exit_code == 2 and "minimum probability" in plain(result)


def test_the_filtered_pair_backtest_runs(pair_data):
    result = CliRunner().invoke(pair_data.app, ["pair-backtest", "AAA:BBB", "--min-probability", "0.6"])
    assert result.exit_code == 0, result.output
