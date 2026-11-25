# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Fundamentals and corporate events (#165). The provider is always replaced: no test touches the network."""

import time
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from tradebot import fundamentals as fund
from tradebot.screener import run_screen

TODAY = date.today()
INFO = {"longName": "Acme Corp", "sector": "Industrials", "currency": "USD", "trailingPE": 12.5, "forwardPE": 11.0,
        "priceToSalesTrailing12Months": 1.8, "profitMargins": 0.12, "operatingMargins": 0.2, "revenueGrowth": 0.07,
        "earningsGrowth": -0.03, "dividendRate": 2.0, "currentPrice": 100.0, "marketCap": 5e10}
RAW = {"info": INFO,
       "earnings": [TODAY - timedelta(days=90), TODAY + timedelta(days=20)],
       "splits": {pd.Timestamp("2020-08-31", tz="America/New_York"): 4.0},
       "dividends": {pd.Timestamp("2024-02-09"): 0.24},
       "ex_dividend": TODAY + timedelta(days=5)}


@pytest.fixture(autouse=True)
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADEBOT_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("TRADEBOT_CONFIG", str(tmp_path / "config.toml"))


def counting(raw=RAW):
    calls = []

    def fetch(symbol):
        calls.append(symbol)
        return raw

    fetch.calls = calls
    return fetch


def test_parse_info():
    f = fund.parse_info(INFO)
    assert (f.name, f.pe, f.ps, f.profit_margin, f.market_cap) == ("Acme Corp", 12.5, 1.8, 0.12, 5e10)
    assert f.dividend_yield == pytest.approx(0.02)  # rate / price, whatever scale Yahoo uses for its own field
    assert f.has_data()


def test_parse_info_missing_and_loss_making():
    f = fund.parse_info({"trailingPE": "Infinity", "profitMargins": None, "forwardPE": "x"})
    assert f.pe is None and f.profit_margin is None and f.forward_pe is None and not f.has_data()
    assert fund.parse_info({"trailingPE": -4.0}).pe is None


def test_parse_events():
    ev = fund.parse_events(RAW)
    kinds = [(e.kind, e.label) for e in ev]
    assert ("split", "Split 4:1") in kinds and ("earnings", "Earnings") in kinds
    assert sum(e.kind == "dividend" for e in ev) == 2  # the paid one and the announced ex-dividend date
    assert ev == sorted(ev)
    assert [e.upcoming for e in ev if e.kind == "split"] == [False]


def test_load_caches_and_refreshes():
    fetch = counting()
    first = fund.load("ACME", fetch)
    again = fund.load("ACME", fetch)
    assert len(fetch.calls) == 1 and again.fundamentals.pe == 12.5 and len(again.events) == len(first.events)
    fund.load("ACME", fetch, refresh=True)
    assert len(fetch.calls) == 2
    fund.load("ACME", fetch, max_age=-1)
    assert len(fetch.calls) == 3


def test_load_failure_is_graceful_and_serves_stale_cache():
    def broken(symbol):
        raise ConnectionError("offline")

    snap = fund.load("ACME", broken)
    assert snap.error == "offline" and not snap.fundamentals.has_data() and snap.events == []
    fund.load("ACME", counting())
    stale = fund.load("ACME", broken, max_age=0)
    assert stale.fundamentals.pe == 12.5 and not stale.error


def test_empty_answer_is_an_error_not_cached():
    assert fund.load("ACME", lambda s: {"info": {}}).error
    assert fund.load("ACME", counting()).fundamentals.has_data()


def test_crypto_and_pairs_never_fetch():
    fetch = counting()
    for s in ("BTC/USDT", "KO:PEP", "^GSPC"):
        assert fund.load(s, fetch).error
    assert not fetch.calls


def test_next_earnings_and_screen_values():
    snap = fund.load("ACME", counting())
    assert snap.next_earnings() == TODAY + timedelta(days=20)
    assert snap.next_event("dividend").day == TODAY + timedelta(days=5)
    row = fund.screen_values(snap)
    assert row["days_to_earnings"] == 20.0 and row["pe"] == 12.5
    empty = fund.screen_values(fund.Snapshot("X"))
    assert all(np.isnan(v) for v in empty.values())


def test_format_value():
    assert fund.format_value("pe", 12.46) == "12.5"
    assert fund.format_value("profit_margin", 0.123) == "12.3%"
    assert fund.format_value("market_cap", 5e10) == "50.0B"
    assert fund.format_value("pe", None) == "–" and fund.format_value("pe", float("nan")) == "–"


# -- the earnings rule ----------------------------------------------------------------------------------

def days(n, start="2024-01-01"):
    return pd.date_range(start, periods=n, freq="D", tz="UTC")


def test_blackout_mask_covers_the_days_before_and_including_earnings():
    idx = days(20)
    mask = fund.blackout_mask(idx, [date(2024, 1, 10)], 3)
    assert list(idx[mask.to_numpy()].day) == [7, 8, 9, 10]
    assert not fund.blackout_mask(idx, [date(2024, 1, 10)], 0).any()
    assert not fund.blackout_mask(idx, [], 3).any()


def test_apply_blackout_blocks_entries_but_keeps_open_positions():
    idx = days(8)
    pos = pd.Series([0, 1, 1, 1, 0, 1, 1, 1], index=idx, dtype=float)
    block = pd.Series([False, True, False, True, True, True, False, False], index=idx)
    out = fund.apply_blackout(pos, block)
    # day 1: entry blocked; day 2: entry allowed; day 3: already long, kept; day 5: entry blocked; day 6: free
    assert out.tolist() == [0, 0, 1, 1, 0, 0, 1, 1]
    short = pd.Series([0, -1, -1], index=days(3), dtype=float)
    assert fund.apply_blackout(short, pd.Series([False, True, True], index=short.index)).tolist() == [0, 0, 0]
    flip = pd.Series([1, -1], index=days(2), dtype=float)  # a long turned into a short is a new entry
    assert fund.apply_blackout(flip, pd.Series([False, True], index=flip.index)).tolist() == [1, 0]


class AlwaysLong:
    name, values = "always_long", {}

    def run_fills(self, df):
        return pd.Series(1.0, index=df.index), None


def test_backtest_with_blackout_skips_the_entry():
    from tradebot.backtest import backtest

    n = 60
    idx = days(n)
    close = 100 + np.arange(n, dtype=float)
    df = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1.0}, index=idx)
    algo = AlwaysLong()
    plain = backtest(df, algo)
    mask = fund.blackout_mask(idx, [idx[1].date()], 5)
    blocked = backtest(df, algo, blackout=mask)
    assert plain.positions.iloc[:2].sum() == 2 and blocked.positions.iloc[:2].sum() == 0  # no entry up to the earnings day
    assert blocked.positions.iloc[2] == 1  # the first candle after it enters
    assert blocked.positions.iloc[-1] == plain.positions.iloc[-1]


def test_blackout_days_setting(tmp_path):
    assert fund.blackout_days() == 0
    (tmp_path / "config.toml").write_text("[events]\nearnings_blackout_days = 4\n")
    assert fund.blackout_days() == 4
    (tmp_path / "config.toml").write_text('[events]\nearnings_blackout_days = "x"\n')
    assert fund.blackout_days() == 0


def test_earnings_blackout_off_or_unknown_dates():
    df = pd.DataFrame({"close": [1.0, 2.0]}, index=days(2))
    assert fund.earnings_blackout(df, "ACME", 0, counting()) is None
    assert fund.earnings_blackout(df, "BTC/USDT", 3, counting()) is None
    assert fund.earnings_blackout(df, "ACME", 3, counting({"info": INFO})) is None
    assert fund.earnings_blackout(df, "OTHER", 3, counting()) is not None


# -- screener -------------------------------------------------------------------------------------------

def frame(n=300):
    idx = pd.date_range("2023-01-01", periods=n, freq="D", tz="UTC")
    close = 50 + np.arange(n, dtype=float) * 0.1
    return pd.DataFrame({"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1000.0}, index=idx)


def test_screener_filters_on_fundamentals_without_the_network():
    cheap = {**RAW, "info": {**INFO, "trailingPE": 9.0}}
    snaps = {"CHEAP": fund.load("CHEAP", counting(cheap)), "DEAR": fund.load("DEAR", counting()),
             "BTC/USDT": fund.load("BTC/USDT", counting())}
    data = {s: frame() for s in snaps}
    table = run_screen(list(data), data.__getitem__, ["pe < 10", "days_to_earnings < 30"], market_cap=lambda s: None,
                       fundamentals=snaps.__getitem__)
    assert list(table.symbol) == ["CHEAP"]
    assert {"pe", "dividend_yield", "days_to_earnings"} <= set(table.columns)
    assert table.market_cap.iloc[0] == 5e10  # filled from the fundamentals when the market-cap source has none


def test_screener_columns_only_when_wanted():
    data = {"A": frame()}
    fund.load("A", counting())
    plain = run_screen(["A"], data.__getitem__, ["close > 0"], market_cap=lambda s: None)  # no filter on one: not asked
    assert "pe" not in plain.columns
    auto = run_screen(["A"], data.__getitem__, ["pe < 20"], market_cap=lambda s: None)  # served from the cache
    assert len(auto) == 1 and "pe" in auto.columns
    none = run_screen(["A"], data.__getitem__, ["close > 0"], market_cap=lambda s: None, fundamentals=False)
    assert "pe" not in none.columns


def test_screener_without_provider_data_has_nan(monkeypatch):
    monkeypatch.setattr(fund, "fetch_raw", lambda s: (_ for _ in ()).throw(ConnectionError("down")))
    monkeypatch.setattr(fund, "load", lambda s, fetch=fund.fetch_raw, **k: fund.Snapshot(s, error="down"))
    data = {"A": frame()}
    t = run_screen(["A"], data.__getitem__, market_cap=lambda s: None, fundamentals=True)
    assert np.isnan(t.pe.iloc[0])
    assert run_screen(["A"], data.__getitem__, ["pe < 20"], market_cap=lambda s: None, fundamentals=True).empty


def test_cache_file_is_json_in_the_cache_folder(tmp_path):
    fund.load("ACME", counting())
    files = list((tmp_path / "cache" / "fundamentals").glob("*.json"))
    assert [f.name for f in files] == ["ACME.json"]
    assert time.time() - fund._read("ACME").fetched < 60


def test_cli_fundamentals_command(monkeypatch):
    from typer.testing import CliRunner

    from tradebot.cli.main import app

    monkeypatch.setattr(fund, "fetch_raw", counting())
    out = CliRunner().invoke(app, ["fundamentals", "ACME", "BTC/USDT"]).output
    assert "12.5" in out and "Split 4:1" in out and "no fundamentals for crypto" in out
