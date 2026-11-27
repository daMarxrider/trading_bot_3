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


# -- wrong input: settings, amounts, prices, the account file ------------------------------------------------
def test_a_zero_fee_in_the_config_is_kept(tmp_path):
    b = PaperBroker({"fee": 0}, data=Data(), path=tmp_path / "acc.json")
    assert b.fee == 0.0


def test_a_zero_fee_buys_at_the_bare_price(tmp_path):
    b = PaperBroker({"fee": 0, "starting_balance": 1000}, data=Data(), path=tmp_path / "acc.json")
    b.place_order("ETH/USDT", Side.BUY, 5)
    assert b.cash == pytest.approx(500.0)  # 1000 - 5 * 100, no fee


def test_missing_settings_use_the_defaults(tmp_path):
    b = PaperBroker({}, data=Data(), path=tmp_path / "acc.json")
    assert (b.fee, b.starting_balance, b.currency) == (0.001, 10_000.0, "USDT")


def test_settings_written_as_strings_are_read_as_numbers(tmp_path):
    b = PaperBroker({"fee": "0.002", "starting_balance": "500"}, data=Data(), path=tmp_path / "acc.json")
    assert (b.fee, b.starting_balance) == (0.002, 500.0)


@pytest.mark.parametrize("fee", [-0.001, 1.0, 5, float("nan"), float("inf"), "abc"])
def test_an_impossible_fee_is_rejected(tmp_path, fee):
    with pytest.raises(ValueError, match="fee"):
        PaperBroker({"fee": fee}, data=Data(), path=tmp_path / "acc.json")


@pytest.mark.parametrize("balance", [0, -100, float("nan"), "lots"])
def test_an_impossible_starting_balance_is_rejected(tmp_path, balance):
    with pytest.raises(ValueError, match="starting_balance"):
        PaperBroker({"starting_balance": balance}, data=Data(), path=tmp_path / "acc.json")


@pytest.mark.parametrize("amount", [0, -1, float("nan"), float("inf")])
@pytest.mark.parametrize("side", [Side.BUY, Side.SELL])
def test_an_impossible_amount_is_rejected(make, side, amount):
    b = make[0]()
    with pytest.raises(ValueError, match="amount"):
        b.place_order("ETH/USDT", side, amount)


def test_a_rejected_amount_leaves_the_account_untouched(make, tmp_path):
    b = make[0]()
    with pytest.raises(ValueError):
        b.place_order("ETH/USDT", Side.BUY, float("nan"))
    assert b.cash == 1000 and not b.orders and not (tmp_path / "acc.json").exists()


@pytest.mark.parametrize("price", [0.0, -5.0, float("nan"), float("inf")])
def test_no_usable_price_fills_nothing(make, price):
    mk, data = make
    b = mk()
    data.price = price
    with pytest.raises(ValueError, match="no usable price"):
        b.place_order("ETH/USDT", Side.BUY, 1)


def test_limit_orders_are_not_simulated(make):
    with pytest.raises(NotImplementedError):
        make[0]().place_order("ETH/USDT", Side.BUY, 1, type="limit", price=90)


def test_an_unknown_side_is_rejected(make):
    with pytest.raises(ValueError):
        make[0]().place_order("ETH/USDT", "hold", 1)


def test_spending_exactly_all_cash_is_allowed(make):
    b = make[0]()
    b.place_order("ETH/USDT", Side.BUY, 1000 / 100.1)  # 9.99 units * 100 * 1.001 = 1000
    assert b.cash == pytest.approx(0.0, abs=1e-9)


def test_selling_part_keeps_the_average_cost(make):
    mk, data = make
    b = mk()
    b.place_order("ETH/USDT", Side.BUY, 4)
    data.price = 200
    b.place_order("ETH/USDT", Side.SELL, 1)
    assert b.positions["ETH"] == {"quantity": 3, "avg_cost": 100.0}


def test_a_second_buy_averages_the_cost(make):
    mk, data = make
    b = mk()
    b.place_order("ETH/USDT", Side.BUY, 1)
    data.price = 130
    b.place_order("ETH/USDT", Side.BUY, 2)
    assert b.positions["ETH"]["avg_cost"] == pytest.approx(120.0)  # (100 + 2 * 130) / 3


@pytest.mark.parametrize("content", ["{not json", "[]", '{"cash": 1}', '{"starting_balance": "x", "cash": 1, '
                                     '"currency": "USDT", "fee": 0, "positions": {}, "orders": [], "created": ""}'])
def test_a_corrupt_account_file_says_how_to_fix_it(tmp_path, content):
    (tmp_path / "acc.json").write_text(content)
    with pytest.raises(RuntimeError, match="paper reset"):
        PaperBroker(data=Data(), path=tmp_path / "acc.json")


def test_summary_values_an_unpriceable_holding_at_cost(make):
    mk, data = make
    b = mk()
    b.place_order("ETH/USDT", Side.BUY, 2)

    def offline(_symbol):
        raise ConnectionError("offline")

    data.latest_price = offline
    assert b.summary()["positions_value"] == pytest.approx(200.0)
