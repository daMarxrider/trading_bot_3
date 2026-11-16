# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Trade journal (#159): storage, FIFO lots, exports, bot and CLI hooks, offline."""

import csv
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest
from typer.testing import CliRunner

from tradebot.brokers.crypto import BinanceBroker
from tradebot.cli.main import app
from tradebot.journal import Journal, fifo_lots, unrealised, yearly_summary
from tradebot.live import TradingBot

from .test_brokers_data import FakeExchange
from .test_services import Scripted

T0 = datetime(2024, 12, 20, tzinfo=timezone.utc)


@pytest.fixture
def j(tmp_path):
    journal = Journal(tmp_path / "j.sqlite")
    yield journal
    journal.close()


def rec(j, side, amount, price, days, fee=0.0, symbol="ETH/USDT", mode="paper"):
    return j.record(symbol, side, amount, price, fee=fee, mode=mode, time=T0 + timedelta(days=days))


def test_fifo_lots_partial_fills_and_fees(j):
    rec(j, "buy", 2, 100, 0, fee=2)
    rec(j, "buy", 1, 120, 1, fee=1)
    rec(j, "sell", 2.5, 130, 2, fee=2.5)
    lots, opens = fifo_lots(j.fills())
    assert [(lot.amount, lot.open_price) for lot in lots] == [(2, 100), (0.5, 120)]
    assert lots[0].fees == pytest.approx(2 + 2.0)  # whole buy fee + 2/2.5 of the sell fee
    assert lots[1].fees == pytest.approx(0.5 + 0.5)
    assert lots[0].pnl == pytest.approx(2 * 30 - 4) and lots[1].pnl == pytest.approx(0.5 * 10 - 1)
    assert opens[0]["amount"] == pytest.approx(0.5) and opens[0]["fee"] == pytest.approx(0.5)
    assert unrealised(opens, {"ETH/USDT": 140}) == {"ETH/USDT": pytest.approx(0.5 * 20 - 0.5)}
    assert unrealised(opens, {}) == {}


def test_short_lot_and_modes_are_separate(j):
    rec(j, "sell", 1, 100, 0, symbol="BTC/USDT:USDT")
    rec(j, "buy", 1, 90, 1, symbol="BTC/USDT:USDT")
    rec(j, "buy", 1, 50, 0, mode="dry")
    lots = j.lots()
    assert len(lots) == 1 and lots[0].direction == "short" and lots[0].pnl == pytest.approx(10)
    assert j.summary("real")["fills"] == 2 and j.summary("dry")["open"] == 1


def test_trades_pnl_notes_tags_and_filters(j):
    a = rec(j, "buy", 1, 100, 0)
    b = rec(j, "sell", 1, 110, 1, fee=1)
    assert j.annotate(b, note="took profit", tags="swing,  eth ,")
    assert not j.annotate(999, note="x")
    rows = j.trades()
    assert rows[0]["id"] == a and rows[0]["pnl"] is None and rows[1]["pnl"] == pytest.approx(9)
    assert rows[1]["tags"] == "swing, eth" and rows[1]["note"] == "took profit"
    assert [r["id"] for r in j.trades(tag="eth")] == [b] and not j.trades(tag="nope")
    assert j.summary()["win_rate"] == 1.0 and j.summary()["fees"] == 1


def test_export_lots_per_year_and_long_term(j, tmp_path):
    rec(j, "buy", 1, 100, 0)
    rec(j, "sell", 1, 150, 5)  # closes in 2024
    rec(j, "buy", 1, 100, 10)
    rec(j, "sell", 1, 80, 400)  # held 390 days, closes in 2026
    path = tmp_path / "lots.csv"
    assert j.export_lots(path) == 2
    rows = list(csv.DictReader(path.open()))
    assert [r["year"] for r in rows] == ["2024", "2026"] and [r["term"] for r in rows] == ["short", "long"]
    assert j.export_lots(path, year=2026) == 1
    summary = yearly_summary(j.lots())
    assert summary[0]["gains"] == 50 and summary[1]["losses"] == -20 and summary[1]["long_term"] == -20
    assert j.export_trades(path) == 4
    assert "pnl" in path.read_text().splitlines()[0]


def make_bot(journal, dry_run=False):
    now = pd.Timestamp.now(tz="UTC").floor("1D")
    start = int((now - pd.Timedelta(days=9)).timestamp() * 1000)
    candles = [[start + i * 86_400_000, 100, 100, 100, 100, 1] for i in range(9)]
    algo = Scripted()
    return TradingBot(BinanceBroker(exchange=FakeExchange(candles)), algo, "BTC/USDT", "1d", 100.0,
                      dry_run=dry_run, journal=journal), algo


def test_bot_records_orders_and_dry_runs(j):
    bot, algo = make_bot(j)
    algo.next = 1.0
    bot.step()
    algo.next = 0.0
    bot.step()
    fills = j.fills()
    assert [f["side"] for f in fills] == ["buy", "sell"] and {f["mode"] for f in fills} == {"live"}
    assert fills[0]["symbol"] == "BTC/USDT" and fills[0]["broker"]
    dry, algo = make_bot(j, dry_run=True)
    algo.next = 1.0
    dry.step()
    assert j.fills("dry")[0]["price"] == 100 and len(j.fills("real")) == 2


def test_bot_survives_a_broken_journal(j):
    bot, algo = make_bot(j)
    j.close()
    algo.next = 1.0
    assert bot.step().action == "buy"
    j.__init__(j.path)  # reopen for the fixture's close()


def test_cli_show_note_export(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADEBOT_JOURNAL", str(tmp_path / "c.sqlite"))
    j = Journal()
    rec(j, "buy", 1, 100, 0)
    fid = rec(j, "sell", 1, 120, 1)
    j.close()
    r = CliRunner()
    assert r.invoke(app, ["journal", "note", str(fid), "--note", "good", "--tags", "a,b"]).exit_code == 0
    assert r.invoke(app, ["journal", "note", "99", "--note", "x"]).exit_code != 0
    out = r.invoke(app, ["journal", "show"])
    assert out.exit_code == 0 and "realised +20.00" in out.output and "good" in out.output
    path = tmp_path / "lots.csv"
    out = r.invoke(app, ["journal", "export", str(path), "--lots", "--year", "2024"])
    assert out.exit_code == 0 and "1 lots" in out.output and path.exists()
