# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""config.toml and settings.json against hand-edited files and awkward values: written so they read back, and a
broken file is never made worse."""

import json
import logging

import pytest

from tradebot.config import (
    DEFAULT_WATCHLIST,
    Settings,
    broker_credentials,
    load_config,
    save_config_value,
    save_watchlist,
    watchlist,
    watchlists,
)


@pytest.fixture
def config_file(tmp_path, monkeypatch):
    path = tmp_path / "config.toml"
    monkeypatch.setenv("TRADEBOT_CONFIG", str(path))
    return path


# -- save_config_value: awkward values read back unchanged ----------------------------------------------------
@pytest.mark.parametrize("value", [
    'say "hi"',
    "back\\slash",
    "two\nlines",
    "tab\there",
    "Ünïcödé €",
    "# not a comment",
    "",
    True,
    0,
    -1.5,
    [],
    ["AAPL", "BRK.B", "BTC/USDT"],
])
def test_awkward_values_read_back_unchanged(config_file, value):
    save_config_value("key", value)
    assert load_config()["key"] == value


@pytest.mark.parametrize("key", ["with space", "a.b", 'quo"te', "Dépôt", "#hash", "[bracket]"])
def test_awkward_watchlist_names_read_back_unchanged(config_file, key):
    save_watchlist(key, ["AAPL"])
    assert watchlist(key) == ["AAPL"]


def test_a_value_toml_cannot_hold_is_refused_and_the_file_is_untouched(config_file):
    config_file.write_text('theme = "paper"\n')
    assert save_config_value("ratio", float("nan")) is False
    assert config_file.read_text() == 'theme = "paper"\n'


def test_a_nested_value_is_refused(config_file):
    assert save_config_value("nested", {"a": 1}) is False
    assert not config_file.exists()


def test_a_config_broken_elsewhere_is_never_rewritten(config_file):
    config_file.write_text("other = = broken\n")
    assert save_config_value("theme", "paper") is False
    assert config_file.read_text() == "other = = broken\n"


def test_a_broken_line_of_the_saved_key_itself_is_replaced_by_a_valid_one(config_file):
    config_file.write_text("theme = = broken\n")
    save_config_value("theme", "paper")
    assert load_config() == {"theme": "paper"}


def test_removing_a_key_that_does_not_exist_creates_no_file(config_file):
    assert save_config_value("gone", None, table="watchlists") is True
    assert not config_file.exists()


def test_removing_a_key_keeps_its_neighbours(config_file):
    config_file.write_text('a = 1\nb = 2\nc = 3\n')
    save_config_value("b", None)
    assert load_config() == {"a": 1, "c": 3}


def test_a_value_containing_a_hash_is_replaced_whole_and_the_comment_kept(config_file):
    config_file.write_text('theme = "a#b" # mine\n')
    save_config_value("theme", "paper")
    assert config_file.read_text() == 'theme = "paper" # mine\n'


def test_a_multi_line_string_value_is_replaced_whole(config_file):
    config_file.write_text('x = """one\ntwo"""\ny = 2\n')
    save_config_value("x", "z")
    assert load_config() == {"x": "z", "y": 2}


def test_a_table_with_a_dotted_name_is_found(config_file):
    config_file.write_text("[brokers.paper]\nfee = 0.001\n")
    save_config_value("fee", 0, table="brokers.paper")
    assert load_config() == {"brokers": {"paper": {"fee": 0}}}


def test_a_key_of_the_same_name_in_another_table_is_left_alone(config_file):
    config_file.write_text('[a]\nx = 1\n\n[b]\nx = 2\n')
    save_config_value("x", 9, table="b")
    assert load_config() == {"a": {"x": 1}, "b": {"x": 9}}


def test_an_unreadable_config_path_is_reported_not_raised(config_file):
    config_file.mkdir()  # a folder where the file should be
    assert save_config_value("theme", "paper") is False


# -- watchlists from hand-edited configs ---------------------------------------------------------------------
def test_an_unknown_watchlist_names_the_existing_ones(config_file):
    config_file.write_text('[watchlists]\nMain = ["AAPL"]\nCrypto = ["BTC/USDT"]\n')
    with pytest.raises(KeyError, match="no watchlist 'Stocks'; there are: Main, Crypto"):
        watchlist("Stocks")


def test_all_watchlists_combined_keep_the_first_place_of_each_symbol(config_file):
    config_file.write_text('[watchlists]\nA = ["X", "Y"]\nB = ["Y", "Z", "X"]\n')
    assert watchlist() == ["X", "Y", "Z"]


def test_entries_of_the_watchlists_table_that_are_not_lists_are_ignored(config_file):
    config_file.write_text('[watchlists]\nA = "AAPL"\nB = ["MSFT"]\n')
    assert watchlists() == {"B": ["MSFT"]}


def test_numbers_in_a_watchlist_become_symbols_as_text(config_file):
    config_file.write_text("[watchlists]\nHK = [700, 9988]\n")
    assert watchlist("HK") == ["700", "9988"]


def test_an_old_watchlist_written_as_one_string_is_not_split_into_letters(config_file, caplog):
    config_file.write_text('watchlist = "AAPL"\n')
    with caplog.at_level(logging.WARNING):
        lists = watchlists()
    assert lists == {"Main": DEFAULT_WATCHLIST}
    assert "must be a list" in caplog.text


def test_a_watchlists_value_that_is_not_a_table_falls_back_to_the_default(config_file):
    config_file.write_text("watchlists = 3\n")
    assert watchlists() == {"Main": DEFAULT_WATCHLIST}


def test_a_broken_config_does_not_lose_watchlists_on_save(config_file):
    config_file.write_text("this is = = not toml\n")
    assert save_watchlist("Main", ["AAPL"]) is False
    assert config_file.read_text() == "this is = = not toml\n"


# -- broker credentials -------------------------------------------------------------------------------------
def test_environment_secrets_win_over_the_config(config_file, monkeypatch):
    config_file.write_text('[brokers.binance]\napi_key = "from-file"\nsecret = "file-secret"\n')
    monkeypatch.setenv("TRADEBOT_BINANCE_API_KEY", "from-env")
    assert broker_credentials("binance") == {"api_key": "from-env", "secret": "file-secret"}


def test_another_brokers_environment_secrets_are_not_mixed_in(config_file, monkeypatch):
    monkeypatch.setenv("TRADEBOT_KUCOIN_PASSWORD", "x")
    assert broker_credentials("binance") == {}


# -- Settings (settings.json) --------------------------------------------------------------------------------
def test_a_value_json_cannot_hold_is_not_saved(tmp_path):
    s = Settings(tmp_path / "settings.json")
    s.set("bad", {1, 2})
    assert s.get("bad") is None


def test_a_refused_value_does_not_block_later_saves(tmp_path):
    path = tmp_path / "settings.json"
    s = Settings(path)
    s.set("bad", object())
    s.set("good", 1)
    assert json.loads(path.read_text()) == {"good": 1}


def test_an_empty_settings_file_means_defaults(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text("")
    assert Settings(path).get("anything", "default") == "default"


def test_a_settings_file_that_is_a_folder_means_defaults(tmp_path):
    path = tmp_path / "settings.json"
    path.mkdir()
    assert Settings(path).get("anything", 7) == 7


def test_settings_survive_a_restart(tmp_path):
    path = tmp_path / "settings.json"
    Settings(path).set("splitter", [1, 2, 3])
    assert Settings(path).get("splitter") == [1, 2, 3]


# -- every text file is UTF-8 on every platform --------------------------------------------------------------------
def text_io_without_encoding() -> list[str]:
    """``file:line`` of every ``read_text``/``write_text``/``open``/``fdopen`` call in the app that opens a file as
    text without naming its encoding (Windows would then use its ANSI code page)."""
    import ast
    from pathlib import Path

    import tradebot

    found = []
    for path in sorted(Path(tradebot.__file__).parent.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call):
                continue
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            keywords = {k.arg for k in node.keywords}
            if name in ("read_text", "write_text") and "encoding" not in keywords:
                found.append(f"{path.name}:{node.lineno}")
            elif name in ("open", "fdopen") and node.args and "encoding" not in keywords:
                mode = next((a.value for a in node.args[1:2] if isinstance(a, ast.Constant)), None)
                mode = next((k.value.value for k in node.keywords if k.arg == "mode"
                             and isinstance(k.value, ast.Constant)), mode)
                if isinstance(mode, str) and "b" not in mode:
                    found.append(f"{path.name}:{node.lineno}")
    return found


def test_every_text_file_is_read_and_written_as_utf8():
    # a Windows default (cp1252) config.toml with "Dépôt" in it can't be read back: tomllib only reads UTF-8
    assert text_io_without_encoding() == []


def test_a_non_ascii_watchlist_is_stored_as_utf8(config_file):
    save_watchlist("Dépôt €", ["SAP.DE"])
    assert "Dépôt €".encode("utf-8") in config_file.read_bytes()
