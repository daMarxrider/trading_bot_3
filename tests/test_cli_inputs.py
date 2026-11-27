# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""What a user types on the command line: wrong input is a usage error (exit code 2, the reason named), never a
traceback, and is caught before any data is fetched or any broker contacted."""

import pytest
import typer
from typer.testing import CliRunner

from tradebot.cli import main


@pytest.fixture
def offline(monkeypatch):
    """Fetching data or contacting a broker fails the test: bad input must be refused before either."""
    def must_not_run(*_a, **_k):
        raise AssertionError("the input should have been refused before this")

    monkeypatch.setattr(main, "load_data", must_not_run)
    monkeypatch.setattr(main, "create_broker", must_not_run)


def run(*args):
    return CliRunner().invoke(main.app, list(args))


def message(result) -> str:
    """The usage error text with the box drawing and line breaks of the terminal layout removed."""
    return " ".join(result.output.replace("│", " ").split())


# -- parse_kv (-p key=value) --------------------------------------------------------------------------------
def test_key_value_pairs_are_read_as_numbers():
    assert main.parse_kv([" lower = 25 ", "upper=7.5e1"]) == {"lower": 25.0, "upper": 75.0}


def test_a_percentage_is_passed_on_for_the_parameter_to_read():
    assert main.parse_kv(["size.fraction=50%"]) == {"size.fraction": "50%"}


@pytest.mark.parametrize("item, reason", [
    ("lower", "expected key=value"),
    ("=25", "expected key=value"),
    ("lower=abc", "'abc' is not a number"),
    ("lower=", "'' is not a number"),
    ("lower=abc%", "'abc%' is not a number"),
])
def test_malformed_key_value_pairs_are_usage_errors(item, reason):
    with pytest.raises(typer.BadParameter, match=reason):
        main.parse_kv([item])


# -- parse_where (scanner filters) ---------------------------------------------------------------------------
@pytest.mark.parametrize("text, expected", [
    ("rsi < 30", {"output": "rsi", "op": "<", "value": 30.0}),
    ("  rsi>=30.5 ", {"output": "rsi", "op": ">=", "value": 30.5}),
    ("fast > slow", {"output": "fast", "op": ">", "value": "slow"}),  # compared with another output
    ("value != -1", {"output": "value", "op": "!=", "value": -1.0}),
])
def test_scanner_filters_that_are_understood(text, expected):
    assert main.parse_where(text) == expected


@pytest.mark.parametrize("text", ["", "rsi", "rsi <", "< 30", "rsi = 30", "rsi << 30", "rsi < 30 40"])
def test_scanner_filters_that_are_rejected(text):
    with pytest.raises(typer.BadParameter, match="use '<output> <op> <number|output>'"):
        main.parse_where(text)


# -- whole commands -------------------------------------------------------------------------------------------
@pytest.mark.parametrize("args, reason", [
    (["-p", "lower=abc"], "'abc' is not a number"),
    (["-p", "lower"], "expected key=value"),
    (["-p", "lower=nan"], "parameter lower: nan is not a finite number"),
    (["-p", "nope=1"], "unknown parameter(s) ['nope']"),
    (["-p", "lower=25%"], "parameter lower: '25%' is not a number (only shares"),
    (["-p", "lower=0", "-p", "period=0"], "parameter period: 0.0 is too small"),
    (["--sizing", "fixed", "-p", "size.fraction=150"], "must be a share from 0 to 1"),
    (["--sizing", "fixed", "-p", "size.fraction=-5%"], "must be a share from 0 to 1"),
    (["--fee", "-0.1"], "-0.1 is not in the range 0.0<=x<=1.0"),
    (["--fee", "1.5"], "1.5 is not in the range 0.0<=x<=1.0"),
    (["--leverage", "200"], "200.0 is not in the range 1.0<=x<=125.0"),
    (["--leverage", "0.5"], "0.5 is not in the range 1.0<=x<=125.0"),
    (["--borrow-fee", "nan"], "the borrow fee must be a finite number"),
    (["--borrow-fee", "-0.01"], "the borrow fee must be a finite number, not negative"),
])
def test_backtest_refuses_wrong_options_before_fetching_data(offline, args, reason):
    result = run("backtest", "AAPL", "rsi_threshold", "--no-plot", *args)
    assert result.exit_code == 2
    assert reason in message(result)


def test_a_misspelled_strategy_is_a_usage_error_listing_the_real_ones(offline):
    result = run("backtest", "AAPL", "rsi_treshold", "--no-plot")
    assert result.exit_code == 2
    assert "unknown algorithm 'rsi_treshold'; available:" in message(result)
    assert "rsi_threshold" in message(result)


@pytest.mark.parametrize("slippage, reason", [
    ("abc", "basis points or 'auto'"),
    ("nan", "finite number of basis points"),
    ("-2", "slippage cannot be negative"),
])
def test_backtest_refuses_a_wrong_slippage(monkeypatch, slippage, reason):
    from .conftest import make_ohlc

    monkeypatch.setattr(main, "load_data", lambda *a, **k: make_ohlc(100))
    result = run("backtest", "AAPL", "rsi_threshold", "--no-plot", "--slippage", slippage)
    assert result.exit_code == 2
    assert reason in message(result)


@pytest.mark.parametrize("train", ["0", "-0.5"])
def test_optimize_refuses_a_training_fraction_of_zero_or_less(monkeypatch, train):
    from .conftest import make_ohlc

    monkeypatch.setattr(main, "load_data", lambda *a, **k: make_ohlc(300))
    result = run("optimize", "AAPL", "sma_cross", "--train", train, "--generations", "1", "--population", "8")
    assert result.exit_code == 2
    assert "training fraction must be above 0" in message(result)


@pytest.mark.parametrize("target, reason", [
    ("AAPL=abc", "'abc' is not a number"),
    ("AAPL=40,AAPL=60", "AAPL is given twice"),
    ("AAPL=nan", "finite"),
    ("AAPL", "expected SYMBOL=percent"),
])
def test_portfolio_analytics_refuses_a_wrong_target_before_contacting_brokers(offline, target, reason):
    result = run("portfolio-analytics", "--target", target, "--no-plot")
    assert result.exit_code == 2
    assert reason in message(result)


@pytest.mark.parametrize("args, reason", [
    (["--where", "rsi <"], "use '<output> <op> <number|output>'"),
    (["-p", "fast=x"], "'fast=x': use name=number"),
])
def test_scan_refuses_wrong_filters_and_params(offline, args, reason):
    result = run("scan", "AAPL", *args)
    assert result.exit_code == 2
    assert reason in message(result)


# -- portfolio-analytics output (render_lines was never imported there, so the default --plot crashed) --------
def analytics(perf_summary: bool):
    import pandas as pd

    idx = pd.date_range("2024-01-01", periods=30, freq="D", tz="UTC")
    series = pd.Series([i / 100 for i in range(30)], index=idx)
    summary = {"value": 1100.0, "invested": 1000.0, "profit": 100.0, "twr": 0.1, "mwr": 0.09, "mwr_annual": 0.2,
               "benchmark": 0.05, "excess": 0.05}
    return {"performance": {"summary": summary if perf_summary else {}, "twr": series, "bench": series},
            "correlation": None, "risk": None, "rebalance": None}


def test_portfolio_analytics_draws_the_return_curves(capsys):
    main._print_analytics(analytics(perf_summary=True), plot=True, benchmark="SPY")
    assert "return since first fill" in capsys.readouterr().out


def test_portfolio_analytics_without_journal_history_says_so(capsys):
    main._print_analytics(analytics(perf_summary=False), plot=True, benchmark="SPY")
    assert "No journal fills" in capsys.readouterr().out


@pytest.mark.parametrize("value", ["50%", "50", "0.5"])
def test_a_sizing_share_from_the_command_line_is_applied_as_given(value):
    algo = main.make_algo("sma_cross", [f"size.fraction={value}"], None, sizing="fixed")
    assert algo.values["size.fraction"] == 0.5


# -- holdings from a file: problems are messages, not tracebacks ---------------------------------------------------
@pytest.fixture
def holdings_config(tmp_path, monkeypatch):
    """config.toml pointing the scalable broker at ``tmp_path / holdings.csv`` (written by the test or not)."""
    import json

    csv = tmp_path / "holdings.csv"
    config = tmp_path / "config.toml"
    config.write_text(f"[brokers.scalable]\nholdings_csv = {json.dumps(str(csv))}\n")  # JSON string = TOML string
    monkeypatch.setenv("TRADEBOT_CONFIG", str(config))
    return csv


@pytest.mark.parametrize("command", [["portfolio"], ["portfolio-analytics", "--no-plot"]])
def test_a_missing_holdings_file_is_a_clear_error(holdings_config, command):
    result = run(*command, "--broker", "scalable")
    assert result.exit_code == 1 and not isinstance(result.exception, FileNotFoundError)
    assert "does not exist; fix brokers.scalable.holdings_csv" in message(result)


@pytest.mark.parametrize("command", [["portfolio"], ["portfolio-analytics", "--no-plot"]])
def test_a_broken_holdings_file_names_the_line(holdings_config, command):
    holdings_config.write_text("symbol,quantity\nAAPL,ten\n")
    result = run(*command, "--broker", "scalable")
    assert result.exit_code == 1
    assert "line 2: quantity 'ten' is not a number" in message(result)


def test_an_unknown_broker_is_a_usage_error(holdings_config):
    result = run("portfolio", "--broker", "scalabel")
    assert result.exit_code == 2
    assert "unknown broker 'scalabel'" in message(result)
