# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Disclaimers, the one-time live-trading confirmation and the bundled licence notices (#58)."""

import sys
import tomllib
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tradebot.cli import main as cli


class Started(Exception):
    """Raised by the fake broker factory: the bot got past the confirmation."""


@pytest.fixture
def config_file(tmp_path, monkeypatch):
    path = tmp_path / "config.toml"
    monkeypatch.setenv("TRADEBOT_CONFIG", str(path))

    def create_broker(*args, **kwargs):
        raise Started

    monkeypatch.setattr(cli, "create_broker", create_broker)
    return path


def run_bot(*flags, input=None):
    return CliRunner().invoke(cli.app, ["bot", "run", "BTC/USDT", "sma_cross", *flags], input=input)


def test_version_shows_the_disclaimer():
    res = CliRunner().invoke(cli.app, ["--version"])
    assert res.exit_code == 0
    assert "not affiliated" in res.output and "not liable" in res.output


def test_help_mentions_the_disclaimer():
    assert "Not affiliated" in CliRunner().invoke(cli.app, ["--help"]).output


def test_declining_the_live_risk_stops_the_bot(config_file):
    res = run_bot("--live", input="n\n")
    assert res.exit_code == 1 and not isinstance(res.exception, Started)
    assert "real money" in res.output
    assert not config_file.exists()


def test_accepting_the_live_risk_is_remembered(config_file):
    res = run_bot("--live", input="y\n")
    assert isinstance(res.exception, Started)
    assert tomllib.loads(config_file.read_text())["live_risk_accepted"]

    again = run_bot("--live")  # no input: asking again would abort
    assert isinstance(again.exception, Started) and "real money" not in again.output


def test_dry_run_needs_no_confirmation(config_file):
    res = run_bot()
    assert isinstance(res.exception, Started) and not config_file.exists()


def test_notices_cover_bundled_libraries_only():
    sys.path.insert(0, str(Path(__file__).parents[1] / "packaging"))
    try:
        import third_party_notices
    finally:
        sys.path.pop(0)
    names = {d.metadata["Name"].lower() for d in third_party_notices.bundled()}
    assert {"ccxt", "yfinance", "numpy", "typer"} <= names
    assert not names & {"tradebot", "pytest", "pyinstaller", "pytr", "torch"}
    text = third_party_notices.render()
    assert "Python " in text and "MIT License" in text
