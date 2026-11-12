# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Index universes (#110) with mocked fetchers."""

import dataclasses
import json

import pytest

from tradebot import universes
from tradebot.universes import INDICES, MAX_SYMBOLS, REFRESH_SECONDS, index_symbols, yahoo_symbol


def patched(monkeypatch, fetch):
    monkeypatch.setitem(INDICES, "dow30", dataclasses.replace(INDICES["dow30"], fetch=fetch))


def test_ticker_mapping():
    assert yahoo_symbol("BRK.B") == "BRK-B" and yahoo_symbol(" aapl ") == "AAPL"
    assert yahoo_symbol("AIR.PA", ".DE") == "AIR.DE" and yahoo_symbol("BT.A", ".L") == "BT-A.L" and yahoo_symbol("SAP", ".DE") == "SAP.DE" and yahoo_symbol("SAP.DE", ".DE") == "SAP.DE"
    assert yahoo_symbol("nan") == "" and yahoo_symbol("NYSE: KO") == "KO"


def test_every_index_has_ticker_title_icon_and_a_bundled_list():
    for key, index in INDICES.items():
        assert index.title and index.icon and index.ticker and universes.BUNDLED[key]


def test_fetch_is_cached_for_a_week_then_refreshed(monkeypatch, tmp_path):
    calls = []
    patched(monkeypatch, lambda: calls.append(1) or ["A", "B"])
    assert index_symbols("dow30", tmp_path, now=1000) == ["A", "B"]
    assert index_symbols("dow30", tmp_path, now=1000 + REFRESH_SECONDS - 1) == ["A", "B"] and len(calls) == 1
    index_symbols("dow30", tmp_path, now=1000 + REFRESH_SECONDS + 1)
    assert len(calls) == 2


def test_offline_uses_the_stale_cache_then_the_bundled_list(monkeypatch, tmp_path):
    (tmp_path / "index_dow30.json").write_text(json.dumps({"time": 0, "symbols": ["OLD"]}))

    def offline():
        raise OSError("no network")

    patched(monkeypatch, offline)
    assert index_symbols("dow30", tmp_path, now=10 * REFRESH_SECONDS) == ["OLD"]
    assert index_symbols("dow30", tmp_path / "empty", now=1) == universes.BUNDLED["dow30"]
    assert index_symbols("dow30") == universes.BUNDLED["dow30"]  # no cache folder at all


def test_cap_and_duplicates(monkeypatch, tmp_path, caplog):
    patched(monkeypatch, lambda: [f"S{i}" for i in range(MAX_SYMBOLS + 50)] + ["S0"])
    assert len(index_symbols("dow30", tmp_path)) == MAX_SYMBOLS
    assert "first 4000" in caplog.text


def test_nasdaq100_from_the_nasdaq_api_and_a_complete_bundled_dow(monkeypatch):
    rows = [{"symbol": "AAPL"}, {"symbol": "BRK.B"}, {"symbol": ""}]
    monkeypatch.setattr(universes, "_get", lambda url: json.dumps({"data": {"data": {"rows": rows}}}))
    assert INDICES["nasdaq100"].fetch() == ["AAPL", "BRK-B"]
    assert INDICES["dow30"].fetch() == universes.BUNDLED["dow30"] and len(universes.BUNDLED["dow30"]) == 30
