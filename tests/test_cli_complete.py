# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Shell completion: the callbacks, the disk caches, and the real completion protocol (no network)."""

import os
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

import pytest

from tradebot.cli import complete
from tradebot.symbols import CryptoPairs, SymbolMatch, cached_queries, yahoo_search_http


def values(items):
    return [v for v, _ in items]


def test_algorithms_and_members():
    assert "rsi_threshold" in values(complete.algorithms("rsi"))
    assert all(v.startswith("ma") for v in values(complete.algorithms("ma")))
    out = values(complete.members("rsi_threshold,ma"))
    assert out == ["rsi_threshold,macd_cross"]  # completes the last member, keeps the others
    assert "rsi_threshold,rsi_threshold" not in values(complete.members("rsi_threshold,"))  # no repeats


def test_indicator_specs_complete_names_then_params():
    assert "bbands" in values(complete.indicator_specs("bb"))
    out = complete.indicator_specs("bbands:")
    assert values(out) == ["bbands:period=", "bbands:width="]
    assert "default 20" in out[0][1]
    assert values(complete.indicator_specs("bbands:period=20,")) == ["bbands:period=20,width="]  # unused only
    assert complete.indicator_specs("nope:") == []


def test_algorithm_params_from_positional_or_option():
    ctx = types.SimpleNamespace(params={"algo": None, "members": None}, args=["AAPL", "rsi_threshold"])
    assert values(complete.algorithm_params(ctx, "")) == ["period=", "lower=", "upper="]
    ctx = types.SimpleNamespace(params={"algo": "vote", "members": "rsi_threshold"}, args=["AAPL"])
    assert "rsi_threshold.lower=" in values(complete.algorithm_params(ctx, "rsi"))
    ctx = types.SimpleNamespace(params={"algo": None}, args=["AAPL"])
    assert complete.algorithm_params(ctx, "") == []  # no algorithm yet


def test_brokers_timeframes_choices():
    assert ("binance", "live · crypto") in complete.brokers("bin")
    assert values(complete.timeframes("1")) == ["1m", "15m", "1h", "1d", "1w"]
    assert values(complete.choices(("ga", ""), ("pso", ""))("p")) == ["pso"]


def test_symbols_watchlist_offline_and_prefix_only(monkeypatch, tmp_path):
    import tradebot.symbols as symbols

    monkeypatch.setattr("tradebot.config.watchlist", lambda: ["AAPL", "SAP.DE", "AAPL"])
    remote = [SymbolMatch("SOL/USDT", "SOL / USDT", "binance", "crypto"), SymbolMatch("APLE", "Apple Hospitality")]
    monkeypatch.setattr(symbols, "cli_symbol_search",
                        lambda local, cache_dir: symbols.SymbolSearch(local, remote=[lambda q: remote]))
    assert complete.symbols("") == [("AAPL", "watchlist"), ("SAP.DE", "watchlist")]  # nothing typed: no network
    assert values(complete.symbols("sol")) == ["SOL/USDT"]  # case-insensitive ticker match
    assert values(complete.symbols("apple")) == []  # shells can't show name matches: prefix only


def test_crypto_pairs_disk_cache(tmp_path):
    cache = tmp_path / "pairs.json"
    pairs = CryptoPairs([], cache=cache)
    pairs._fetch = lambda: [SymbolMatch("SOL/USDT", "SOL / USDT", "binance", "crypto")]
    assert pairs.load()[0].symbol == "SOL/USDT" and cache.is_file()
    again = CryptoPairs([], cache=cache)
    again._fetch = lambda: pytest.fail("should come from the disk cache")
    assert again.load()[0].symbol == "SOL/USDT"  # a new process reads the file
    os.utime(cache, (time.time() - 2 * 86400,) * 2)
    fetched = threading.Event()

    def outage():
        fetched.set()
        return []

    stale = CryptoPairs([], cache=cache)
    stale._fetch = outage
    assert stale.load()[0].symbol == "SOL/USDT"  # a day old: still answers right away (#63) ...
    assert fetched.wait(5)  # ... while a refresh runs in the background
    time.sleep(0.05)
    assert stale.load()[0].symbol == "SOL/USDT" and cache.is_file()  # an empty outage result isn't cached

    os.utime(cache, (time.time() - 2 * 86400,) * 2)
    newer = CryptoPairs([], cache=cache)
    newer._fetch = lambda: [SymbolMatch("ETH/USDT", "ETH / USDT", "binance", "crypto")]
    newer.load()
    for _ in range(100):
        if newer.load()[0].symbol == "ETH/USDT":
            break
        time.sleep(0.02)
    assert newer.load()[0].symbol == "ETH/USDT"  # the refreshed list replaces the old one
    assert CryptoPairs([], cache=cache).load()[0].symbol == "ETH/USDT"  # and is saved


def test_query_cache(tmp_path):
    calls = []

    def source(query):
        calls.append(query)
        if query == "down":
            raise ConnectionError
        return [SymbolMatch("NVDA", "NVIDIA")]

    search = cached_queries(source, tmp_path / "q.json")
    assert search("nvd")[0].symbol == "NVDA"
    assert search("NVD ")[0].symbol == "NVDA" and calls == ["nvd"]  # normalized key, served from disk
    with pytest.raises(ConnectionError):
        search("down")
    with pytest.raises(ConnectionError):
        search("down")
    assert calls == ["nvd", "down", "down"]  # failures aren't cached


def test_yahoo_search_http_parses_and_filters(monkeypatch):
    class Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"quotes": [{"symbol": "NVDA", "longname": "NVIDIA Corporation", "exchDisp": "NASDAQ",
                                "quoteType": "EQUITY"},
                               {"symbol": "SNVDA=F", "quoteType": "FUTURE"}]}

    seen = {}
    fake = types.SimpleNamespace(get=lambda url, params, headers, timeout: seen.update(params=params, timeout=timeout)
                                 or Resp())
    monkeypatch.setitem(sys.modules, "requests", fake)
    assert yahoo_search_http("nvd") == [SymbolMatch("NVDA", "NVIDIA Corporation", "NASDAQ", "stock")]
    assert seen["params"]["q"] == "nvd" and seen["timeout"] <= 1.0


# -- the real protocol, as fish and bash call it ----------------------------------------------------
def run_completion(env_extra, tmp_path):
    env = {**os.environ, "XDG_CONFIG_HOME": str(tmp_path), "XDG_CACHE_HOME": str(tmp_path), **env_extra}
    code = "from tradebot.cli.main import app; app(prog_name='tradebot')"
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=60)
    return out.stdout.splitlines()


def test_fish_completion_end_to_end_is_case_insensitive(tmp_path):
    lines = run_completion({"_TRADEBOT_COMPLETE": "complete_fish", "_TYPER_COMPLETE_FISH_ACTION": "get-args",
                            "_TYPER_COMPLETE_ARGS": "tradebot backtest AAPL RSI"}, tmp_path)
    # "RSI" typed upper case still completes the lower-case name (Typer alone would drop it), with help
    assert lines and lines[0].startswith("rsi_threshold\t")


def test_bash_completion_end_to_end(tmp_path):
    lines = run_completion({"_TRADEBOT_COMPLETE": "complete_bash", "COMP_WORDS": "tradebot optimize X y --method ",
                            "COMP_CWORD": "5"}, tmp_path)
    assert lines == ["ga", "pso", "both"]


def test_completion_skips_torch():
    code = "import tradebot.core.ops as ops; print(ops.torch)"
    env = {**os.environ, "_TRADEBOT_COMPLETE": "complete_fish"}
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=60)
    assert out.stdout.strip() == "None"


def test_cache_dir_per_os(monkeypatch):
    from tradebot.config import os_cache_dir

    monkeypatch.delenv("TRADEBOT_CACHE_DIR")  # set for every test (conftest)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("XDG_CACHE_HOME", "/xdg-cache")
    assert os_cache_dir() == Path("/xdg-cache/tradebot")
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", "/local")
    assert os_cache_dir() == Path("/local/tradebot/Cache")
    monkeypatch.setattr(sys, "platform", "darwin")
    assert os_cache_dir() == Path("~/Library/Caches/tradebot").expanduser()
    monkeypatch.setenv("TRADEBOT_CACHE_DIR", "/custom")  # overrides the OS default
    assert os_cache_dir() == Path("/custom")
