# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Money and safety settings of the live bots, as typed on the command line, in the app or in bots.toml: a stake
or a risk limit that can't work (NaN, negative) is refused with its name, never silently switched off."""

import pytest
from typer.testing import CliRunner

from tradebot.cli import main
from tradebot.live import PairBot, RiskLimits, TradingBot
from tradebot.live.manager import AccountLimits, BotSpec, load_bots
from tradebot.live.risk import check_limit, check_stake

NAN, INF = float("nan"), float("inf")


@pytest.fixture(autouse=True)
def private_folders(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.delenv("TRADEBOT_BOTS", raising=False)
    monkeypatch.setattr("sys.platform", "linux")


@pytest.fixture
def offline(monkeypatch):
    """Contacting a broker or asking for the live-risk confirmation fails the test."""
    def must_not_run(*_a, **_k):
        raise AssertionError("the input should have been refused before this")

    monkeypatch.setattr(main, "create_broker", must_not_run)
    monkeypatch.setattr(main, "confirm_live_risk", must_not_run)


def run(*args):
    return CliRunner().invoke(main.app, list(args))


def message(result) -> str:
    return " ".join(result.output.replace("│", " ").split())


# -- the stake ---------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("stake, expected", [(100, 100.0), ("250.5", 250.5), (0.01, 0.01)])
def test_a_positive_stake_is_accepted(stake, expected):
    assert check_stake(stake) == expected


@pytest.mark.parametrize("stake", [0, -100, NAN, INF, "abc", None])
def test_a_stake_that_cant_work_is_refused(stake):
    with pytest.raises(ValueError, match="the stake must be"):
        check_stake(stake)


def test_a_trading_bot_refuses_a_nan_stake():
    with pytest.raises(ValueError, match="the stake must be a positive amount"):
        TradingBot(None, None, "BTC/USDT", stake=NAN)


def test_a_pair_bot_refuses_a_zero_stake():
    with pytest.raises(ValueError, match="the stake must be a positive amount"):
        PairBot(None, ("AAA", "BBB"), "1d", stake=0.0)


def test_a_saved_bot_with_a_nan_stake_names_the_bot():
    with pytest.raises(ValueError, match="bot 'b1': the stake must be a positive amount"):
        BotSpec("b1", "BTC/USDT", "sma_cross", stake=NAN)


# -- per-bot risk limits --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("field", ["max_daily_loss", "max_drawdown", "max_exposure", "max_order_value", "price_band",
                                   "cooldown_minutes"])
@pytest.mark.parametrize("value", [NAN, INF, -1.0, "abc"])
def test_a_risk_limit_that_cant_work_is_refused_by_name(field, value):
    with pytest.raises(ValueError, match=f"risk limit {field}"):
        RiskLimits(**{field: value})


@pytest.mark.parametrize("value", [None, 0])
def test_a_risk_limit_of_zero_or_none_is_off(value):
    from datetime import datetime, timezone

    from tradebot.live import RiskGuard

    g = RiskGuard(RiskLimits(max_daily_loss=value), 100.0, kill_path=lambda: False)
    t = datetime(2026, 3, 2, tzinfo=timezone.utc)
    g.check(0.0, t)
    assert g.check(-1e9, t) is None


def test_risk_limits_written_as_text_are_read():
    assert RiskLimits(max_daily_loss="50").max_daily_loss == 50.0


@pytest.mark.parametrize("value", [2.5, -1, NAN])
def test_the_losing_streak_count_must_be_a_whole_number_of_at_least_zero(value):
    with pytest.raises(ValueError, match="cooldown_losses"):
        RiskLimits(cooldown_losses=value)


def test_a_capital_of_zero_means_the_stake():
    assert RiskLimits(capital=0).capital is None


@pytest.mark.parametrize("capital", [-100, NAN])
def test_a_capital_that_cant_work_is_refused(capital):
    with pytest.raises(ValueError, match="positive amount"):
        RiskLimits(capital=capital)


def test_a_saved_bot_with_a_nan_limit_names_the_bot_and_the_limit():
    with pytest.raises(ValueError, match="bot 'b1': risk limit max_daily_loss"):
        BotSpec("b1", "BTC/USDT", "sma_cross", risk={"max_daily_loss": NAN})


def test_the_limit_check_passes_none_through():
    assert check_limit("x", None) is None


# -- shared account limits ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("field, value", [("max_total_exposure", NAN), ("max_total_exposure", -5),
                                          ("max_daily_loss", INF), ("max_daily_loss", -1),
                                          ("max_open_positions", -1), ("max_open_positions", 1.5)])
def test_an_account_limit_that_cant_work_is_refused(field, value):
    with pytest.raises(ValueError, match=field):
        AccountLimits(**{field: value})


def test_account_limits_of_zero_are_off():
    assert not AccountLimits(max_total_exposure=0, max_daily_loss=0, max_open_positions=0).active()


# -- bots.toml written by hand --------------------------------------------------------------------------------------
def test_a_nan_limit_in_bots_toml_names_the_file(tmp_path):
    path = tmp_path / "bots.toml"
    path.write_text('[[bots]]\nname = "b1"\nsymbol = "BTC/USDT"\nalgorithm = "sma_cross"\n'
                    "risk = { max_daily_loss = nan }\n")
    with pytest.raises(ValueError, match=r"bots.toml: bot 'b1': risk limit max_daily_loss"):
        load_bots(path)


def test_a_negative_account_limit_in_bots_toml_names_the_file(tmp_path):
    path = tmp_path / "bots.toml"
    path.write_text("[limits]\nmax_daily_loss = -50\n")
    with pytest.raises(ValueError, match=r"bots.toml: risk limit max_daily_loss"):
        load_bots(path)


def test_a_share_in_percent_survives_saving_and_loading_bots(tmp_path):
    from tradebot.live.manager import BotSet, save_bots

    path = tmp_path / "bots.toml"
    spec = BotSpec("b1", "BTC/USDT", "sma_cross", params={"size.fraction": "50%"}, sizing="fixed")
    save_bots(BotSet([spec]), path)
    loaded = load_bots(path).bots[0]
    assert loaded.build_algorithm().values["size.fraction"] == 0.5


# -- the command line: refused before any broker is contacted ---------------------------------------------------------
@pytest.mark.parametrize("args, reason", [
    (["--stake", "0"], "the stake must be a positive amount"),
    (["--stake", "-10"], "the stake must be a positive amount"),
    (["--stake", "nan"], "the stake must be a positive amount"),
    (["--max-daily-loss", "-5"], "risk limit max_daily_loss"),
    (["--max-drawdown", "nan"], "risk limit max_drawdown"),
    (["--cooldown-minutes", "-1"], "risk limit cooldown_minutes"),
])
def test_bot_run_refuses_wrong_money_settings_before_connecting(offline, args, reason):
    result = run("bot", "run", "BTC/USDT", "sma_cross", "--live", *args)
    assert result.exit_code == 2
    assert reason in message(result)


def test_pair_run_refuses_a_wrong_stake_before_connecting(offline):
    result = run("bot", "pair-run", "AAA/USDT", "BBB/USDT", "--stake", "0")
    assert result.exit_code == 2
    assert "the stake must be a positive amount" in message(result)


def test_bots_add_refuses_a_nan_stake(offline):
    result = run("bots", "add", "b1", "BTC/USDT", "sma_cross", "--stake", "nan")
    assert result.exit_code == 2
    assert "the stake must be a positive amount" in message(result)


def test_bots_add_takes_a_share_in_percent(offline):
    result = run("bots", "add", "b1", "BTC/USDT", "sma_cross", "--sizing", "fixed", "-p", "size.fraction=50%")
    assert result.exit_code == 0, result.output
    assert load_bots().bots[0].build_algorithm().values["size.fraction"] == 0.5


@pytest.mark.parametrize("args, reason", [
    (["--max-daily-loss", "-5"], "risk limit max_daily_loss"),
    (["--max-total-exposure", "nan"], "risk limit max_total_exposure"),
    (["--max-open-positions", "-1"], "risk limit max_open_positions"),
])
def test_bots_limits_refuses_limits_that_cant_work(offline, args, reason):
    result = run("bots", "limits", *args)
    assert result.exit_code == 2
    assert reason in message(result)


def test_bots_limits_refused_leave_the_saved_limits_alone(offline):
    run("bots", "limits", "--max-daily-loss", "100")
    run("bots", "limits", "--max-daily-loss", "-5")
    assert load_bots().limits.max_daily_loss == 100.0
