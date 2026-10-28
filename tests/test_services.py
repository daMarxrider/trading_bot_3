# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Live bot, portfolio/dividend metrics, screener and news parsing (offline)."""

import numpy as np
import pandas as pd
import pytest

from tradebot.algorithms.base import BaseAlgorithm
from tradebot.brokers.crypto import BinanceBroker
from tradebot.core.models import AssetClass, Position
from tradebot.live import TradingBot
from tradebot.news import parse_yf_item
from tradebot.portfolio import DividendInfo, build_portfolio
from tradebot.screener import run_screen

from .conftest import make_ohlc
from .test_brokers_data import FakeExchange


class Scripted(BaseAlgorithm):
    def __init__(self):
        self.name = "scripted"
        self.next = 0.0
        super().__init__()

    def signals(self, o, p):
        return o.close * 0 + self.next


def test_bot_follows_target_and_ignores_forming_candle():
    day = 86_400_000
    now = pd.Timestamp.now(tz="UTC").floor("1D")
    start = int((now - pd.Timedelta(days=9)).timestamp() * 1000)
    candles = [[start + i * day, 100, 101, 99, 100 + i, 1] for i in range(10)]  # last one = today, still open
    ex = FakeExchange(candles)
    algo = Scripted()
    bot = TradingBot(BinanceBroker(exchange=ex), algo, "BTC/USDT", "1d", stake=216.0, dry_run=False)
    assert len(bot.closed_candles()) == 9
    assert bot.step().action == "hold"
    algo.next = 1.0
    ev = bot.step()
    assert ev.action == "buy" and ev.price == 108 and ev.amount == pytest.approx(2.0)
    assert ex.created[-1][:3] == ("BTC/USDT", "market", "buy")
    assert bot.step().action == "hold"
    algo.next = 0.0
    ev = bot.step()  # FakeExchange reports 0.5 BTC free -> sells only what is available
    assert ev.action == "sell" and ev.amount == pytest.approx(0.5) and bot.holding == 0


def test_dry_run_places_nothing():
    ex = FakeExchange([[i * 86_400_000, 1, 1, 1, 1, 1] for i in range(5)])
    algo = Scripted()
    algo.next = 1.0
    bot = TradingBot(BinanceBroker(exchange=ex), algo, "BTC/USDT", "1d")
    assert bot.step().action == "buy" and not ex.created


def test_portfolio_dividend_metrics():
    positions = [
        Position("KO", 100, avg_cost=40.0),
        Position("NVDA", 10, avg_cost=None),
        Position("BTC", 0.5, asset_class=AssetClass.CRYPTO),
        Position("USDT", 500, asset_class=AssetClass.CRYPTO),
    ]
    prices = {"KO": 60.0, "NVDA": 100.0, "BTC/USDT": 20_000.0}
    divs = {"KO": DividendInfo(forward_annual=2.0, trailing_annual=1.8), "NVDA": DividendInfo(0.04, 0.04)}
    pf = build_portfolio(positions, prices.__getitem__, lambda s: divs[s])
    ko = pf.holdings[0]
    assert ko.current_yield == pytest.approx(2 / 60)
    assert ko.annual_yield == pytest.approx(1.8 / 60)
    assert ko.yield_on_cost == pytest.approx(0.05)
    assert pf.holdings[1].yield_on_cost is None
    assert pf.total_value == pytest.approx(6000 + 1000 + 10000 + 500)
    assert pf.annual_income == pytest.approx(200.4)
    assert pf.yield_on_cost == pytest.approx(0.05)  # only holdings with a known cost basis
    alloc = pf.allocation("asset_class")
    assert alloc["crypto"] == pytest.approx(10500 / 17500)
    assert pf.to_frame()["weight"].sum() == pytest.approx(1.0)


def test_screener_filters():
    data = {"UP": make_ohlc(300, seed=1), "DOWN": make_ohlc(300, seed=2)}
    for sym, ramp in (("UP", np.linspace(10, 100, 300)), ("DOWN", np.linspace(100, 10, 300))):
        for col in ("open", "high", "low", "close"):
            data[sym][col] = ramp
    table = run_screen(list(data) + ["MISSING"], data.__getitem__, ["close > sma200", "change_1y > 0"],
                       extra={"stoch_k": ("stoch", {}, "k")}, market_cap=lambda s: 1e9)
    assert list(table["symbol"]) == ["UP"]
    assert table.loc[0, "rsi"] == pytest.approx(100)
    assert "stoch_k" in table


def test_news_parsing_both_formats():
    old = {"title": "A", "publisher": "Reuters", "link": "http://a", "providerPublishTime": 1_700_000_000}
    new = {"id": "x", "content": {"title": "B", "pubDate": "2026-10-01T10:00:00Z", "summary": "s",
                                  "provider": {"displayName": "Yahoo"}, "canonicalUrl": {"url": "http://b"}}}
    a, b = parse_yf_item("AAPL", old), parse_yf_item("AAPL", new)
    assert (a.title, a.publisher, a.url) == ("A", "Reuters", "http://a")
    assert (b.title, b.publisher, b.url, b.published.year) == ("B", "Yahoo", "http://b", 2026)
    assert parse_yf_item("AAPL", {"content": {}}) is None
