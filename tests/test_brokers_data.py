# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Broker abstractions, data-engine fallback and plugin loading (no network)."""

import textwrap

import pandas as pd
import pytest

from tradebot.algorithms import ALGORITHMS
from tradebot.brokers import BROKERS, ReadOnlyBrokerError
from tradebot.brokers.base import ManualExecutionBroker
from tradebot.brokers.crypto import BinanceBroker, KucoinBroker
from tradebot.brokers.stocks import ScalableCapitalBroker
from tradebot.core.models import ExecutionMode, OrderStatus, OrderType, Side
from tradebot.core.registry import load_plugins
from tradebot.data import DataUnavailable, MarketDataEngine
from tradebot.data.providers import BaseDataProvider, CCXTDataProvider, YFinanceProvider
from tradebot.core.models import AssetClass

from .conftest import make_ohlc


class FakeExchange:
    """Minimal stand-in for a ccxt exchange object."""

    def __init__(self, candles=None):
        self.created, self.canceled = [], []
        self.candles = candles or []

    def load_markets(self):
        pass

    def amount_to_precision(self, symbol, amount):
        return f"{amount:.3f}"

    def price_to_precision(self, symbol, price):
        return f"{price:.2f}"

    def fetch_ticker(self, symbol):
        return {"last": 101.5, "bid": 101.0, "ask": 102.0, "timestamp": 1_700_000_000_000}

    def create_order(self, symbol, type, side, amount, price):
        self.created.append((symbol, type, side, amount, price))
        return {"id": 42, "symbol": symbol, "status": "closed", "amount": amount, "filled": amount, "average": 100.0}

    def cancel_order(self, order_id, symbol):
        self.canceled.append(order_id)

    def fetch_balance(self):
        return {"free": {"BTC": 0.5, "USDT": 1000.0, "ETH": 0.0}, "used": {"BTC": 0.1, "USDT": 0, "ETH": 0}}

    def fetch_ohlcv(self, symbol, timeframe, since=None, limit=None):
        rows = [r for r in self.candles if since is None or r[0] >= since]
        return rows[:limit]


def test_ccxt_broker_unified_api():
    ex = FakeExchange()
    b = BinanceBroker(exchange=ex)
    assert b.execution_mode is ExecutionMode.LIVE
    assert b.get_ticker("BTC/USDT").last == 101.5
    order = b.place_order("BTC/USDT", "buy", 0.12345)
    assert ex.created == [("BTC/USDT", "market", "buy", 0.123, None)]
    assert order.id == "42" and order.status is OrderStatus.CLOSED and order.price == 100.0
    b.place_order("BTC/USDT", Side.SELL, 1, OrderType.LIMIT, 123.456)
    assert ex.created[-1][4] == 123.46
    with pytest.raises(ValueError):
        b.place_order("BTC/USDT", "sell", 1, "limit")
    assert b.cancel_order("42", "BTC/USDT") and ex.canceled == ["42"]
    bal = b.get_balance()
    assert set(bal) == {"BTC", "USDT"} and bal["BTC"].total == pytest.approx(0.6)


def test_registered_brokers():
    for name in ("binance", "kucoin", "pionex", "scalable", "traderepublic"):
        assert name in BROKERS
    assert KucoinBroker.credential_keys[-1] == "password"


def test_semi_auto_broker_creates_intents(tmp_path):
    csv = tmp_path / "h.csv"
    csv.write_text("Symbol,Quantity,Avg_Cost\nAAPL,10,150\nSAP.DE,5,\n")
    seen = []
    b = ScalableCapitalBroker({"holdings_csv": str(csv)}, on_intent=seen.append)
    pos = b.get_positions()
    assert [(p.symbol, p.quantity, p.avg_cost) for p in pos] == [("AAPL", 10, 150), ("SAP.DE", 5, None)]
    order = b.place_order("AAPL", "buy", 3)
    assert order.status is OrderStatus.PENDING_MANUAL
    assert seen and "BUY 3 AAPL" in seen[0].describe()
    assert b.cancel_order(order.id) and not b.pending


def test_read_only_broker_refuses_orders():
    class Tracker(ManualExecutionBroker):  # no class-level name -> stays out of the registry
        execution_mode = ExecutionMode.READ_ONLY

        def get_balance(self):
            return {}

    b = Tracker()
    b.name = "tracker"
    assert not b.can_trade and "tracker" not in BROKERS
    with pytest.raises(ReadOnlyBrokerError):
        b.place_order("AAPL", "buy", 1)


class FakeProvider(BaseDataProvider):
    asset_classes = frozenset({AssetClass.STOCK, AssetClass.CRYPTO})
    timeframes = frozenset({"1d"})

    def __init__(self, name, df=None, fail=False):
        self.name, self.df, self.fail, self.calls = name, df, fail, []

    def fetch(self, symbol, timeframe, start, end):
        self.calls.append((start, end))
        if self.fail:
            raise ConnectionError("down")
        part = self.df[(self.df.index >= start) & (self.df.index < end)]
        if part.empty:
            raise DataUnavailable("empty")
        return part


def test_engine_fills_missing_history_from_next_provider():
    full = make_ohlc(300)
    recent = full.iloc[200:].copy()
    recent["close"] += 0  # identical prices across venues
    broken = FakeProvider("broken", fail=True)
    exchange = FakeProvider("exchange", recent)
    archive = FakeProvider("archive", full)
    eng = MarketDataEngine([broken, exchange, archive])
    df = eng.get_ohlc("BTC/USDT", "1d", start=full.index[0], end=full.index[-1] + pd.Timedelta(days=1))
    assert len(df) == 300
    pd.testing.assert_frame_equal(df, full, check_freq=False)
    # archive was only asked for the head gap
    assert archive.calls == [(full.index[0], recent.index[0])]


def test_engine_raises_when_no_data():
    with pytest.raises(DataUnavailable, match="down"):
        MarketDataEngine([FakeProvider("x", fail=True)]).get_ohlc("AAPL", start="2024-01-01", end="2024-02-01")


def test_ccxt_provider_paginates():
    day = 86_400_000
    candles = [[i * day, 1, 2, 0.5, 1.5, 10] for i in range(25)]
    prov = CCXTDataProvider("binance", page_limit=10, exchange=FakeExchange(candles))
    df = prov.fetch("BTC/USDT", "1d", pd.Timestamp(0, tz="UTC"), pd.Timestamp(20 * day, unit="ms", tz="UTC"))
    assert len(df) == 20 and list(df.columns) == ["open", "high", "low", "close", "volume"]


def test_yfinance_symbol_mapping():
    assert YFinanceProvider.map_symbol("BTC/USDT") == "BTC-USD"
    assert YFinanceProvider.map_symbol("ETH/EUR") == "ETH-EUR"
    assert YFinanceProvider.map_symbol("AAPL") == "AAPL"


def test_plugin_loader(tmp_path):
    (tmp_path / "my_algo.py").write_text(textwrap.dedent("""
        from tradebot.algorithms.base import BaseAlgorithm
        from tradebot.core import ops
        class AlwaysLong(BaseAlgorithm):
            name = "always_long_plugin"
            def signals(self, o, p):
                return o.close * 0 + 1
    """))
    (tmp_path / "broken.py").write_text("raise RuntimeError('boom')")
    loaded = load_plugins([tmp_path])
    assert loaded == ["tradebot_plugin_my_algo"]
    algo = ALGORITHMS.get("always_long_plugin")()
    assert algo.run(make_ohlc(10)).eq(1).all()


def test_quiet_yfinance_beats_yfinance_own_filter():
    import warnings

    from tradebot.data.providers import quiet_yfinance

    with warnings.catch_warnings(record=True) as caught:
        warnings.resetwarnings()
        quiet_yfinance()
        warnings.warn_explicit("inplace is deprecated", DeprecationWarning, "yfinance/utils.py", 1,
                               module="yfinance.utils")
        warnings.warn_explicit("other", UserWarning, "tradebot/x.py", 1, module="tradebot.x")
    assert [str(w.message) for w in caught] == ["other"]
