# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Paper broker: fills, fees, persistence across restarts, bot resume (no network)."""

import pytest

from tradebot.brokers import BROKERS
from tradebot.brokers.paper import PaperBroker
from tradebot.core.models import Side


class Data:
    price = 100.0

    def latest_price(self, symbol):
        return self.price


@pytest.fixture
def make(tmp_path):
    data = Data()
    return lambda **kw: PaperBroker(data=data, path=tmp_path / "acc.json", starting_balance=1000, **kw), data


def test_registered():
    assert BROKERS.get("paper") is PaperBroker


def test_buy_sell_fee_and_pnl(make):
    mk, data = make
    b = mk()
    b.place_order("ETH/USDT", Side.BUY, 5)
    assert b.cash == pytest.approx(1000 - 500 * 1.001)
    data.price = 120
    o = b.place_order("ETH/USDT", Side.SELL, 5)
    assert o.price == 120
    assert b.cash == pytest.approx(1000 - 500.5 + 600 - 0.6)
    assert not b.positions
    s = b.summary()
    assert s["pnl"] == pytest.approx(s["realised_pnl"] - 0.5) and s["orders"] == 2


def test_rejects_overspend_and_oversell(make):
    b = make[0]()
    with pytest.raises(ValueError):
        b.place_order("ETH/USDT", Side.BUY, 11)
    with pytest.raises(ValueError):
        b.place_order("ETH/USDT", Side.SELL, 1)


def test_survives_restart_and_reset(make, tmp_path):
    mk, _ = make
    mk().place_order("ETH/USDT", Side.BUY, 2)
    again = mk()
    assert again.positions["ETH"]["quantity"] == 2 and len(again.orders) == 1
    assert not list(tmp_path.glob("*.tmp"))
    again.reset(500)
    assert mk().cash == 500 and not mk().orders


def test_bot_resumes_holding(make):
    from tradebot.live import TradingBot

    mk, _ = make
    mk().place_order("ETH/USDT", Side.BUY, 2)
    bot = TradingBot(mk(), None, "ETH/USDT", dry_run=False)
    assert bot.holding == 2
