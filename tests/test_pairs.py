# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Pairs: spread symbols and the gap statistics (#117)."""

import numpy as np
import pandas as pd
import pytest

from tradebot.analysis.pairs import half_life, pair_stats
from tradebot.core.models import split_pair
from tradebot.data import MarketDataEngine
from tradebot.data.providers import DataUnavailable

from .test_brokers_data import FakeProvider
from .conftest import make_ohlc


def test_split_pair():
    assert split_pair("KO:PEP") == ("KO", "PEP")
    assert split_pair("BTC/USDT:ETH/USDT") == ("BTC/USDT", "ETH/USDT")
    assert split_pair("BTC/USDT:USDT") is None  # a perpetual contract, one symbol
    assert split_pair("AAPL") is None and split_pair("A:") is None and split_pair("A:B:C") is None


def test_half_life_of_mean_reverting_series():
    rng = np.random.default_rng(1)
    x = np.zeros(3000)
    for i in range(1, len(x)):
        x[i] = 0.9 * x[i - 1] + rng.normal()
    assert half_life(x) == pytest.approx(-np.log(2) / np.log(0.9), rel=0.2)
    assert half_life(np.cumsum(rng.normal(size=3000))) is None or half_life(np.cumsum(rng.normal(size=3000))) > 200


def test_pair_stats():
    rng = np.random.default_rng(2)
    base = np.cumsum(rng.normal(0, 0.01, 800))
    gap = np.zeros(800)
    for i in range(1, 800):
        gap[i] = 0.8 * gap[i - 1] + rng.normal(0, 0.01)
    gap[-1] = 5 * gap.std()  # drifted apart right now
    idx = pd.date_range("2024-01-01", periods=800, freq="D", tz="UTC")
    a, b = pd.Series(np.exp(2 * base + gap), idx), pd.Series(np.exp(base), idx)
    st = pair_stats(a, b)
    assert st.beta == pytest.approx(2, abs=0.2) and st.correlation > 0.5
    assert st.zscore > 2 and st.half_life is not None and st.half_life < 10
    with pytest.raises(ValueError):
        pair_stats(a.iloc[:5], b.iloc[:5])


def engine(**frames):
    class P(FakeProvider):
        def fetch(self, symbol, timeframe, start, end):
            df = frames[symbol]
            return df[(df.index >= start) & (df.index < end)]

    return MarketDataEngine([P("p")], disk=False)


def test_spread_symbol_is_the_ratio_on_common_candles():
    a, b = make_ohlc(60, seed=1), make_ohlc(60, seed=2)
    b = b.iloc[5:]  # b has fewer candles
    out = engine(KO=a, PEP=b).get_ohlc("KO:PEP", start=a.index[0], end=a.index[-1] + pd.Timedelta(days=1))
    assert list(out.index) == list(b.index)
    assert out["close"].iloc[3] == pytest.approx(a["close"].loc[out.index[3]] / b["close"].iloc[3])
    assert (out["high"] >= out["low"]).all()


def test_spread_without_common_candles_raises():
    a = make_ohlc(10, seed=1)
    b = a.copy()
    b.index = b.index + pd.Timedelta(hours=7)
    with pytest.raises(DataUnavailable):
        engine(KO=a, PEP=b).get_ohlc("KO:PEP", start=a.index[0], end=a.index[-1] + pd.Timedelta(days=2))


def test_pair_across_currencies_is_converted():
    a, b = make_ohlc(30, seed=1), make_ohlc(30, seed=2)
    fx = a.copy() * 0 + 1.25  # 1 EUR = 1.25 USD, every day
    fx = fx.iloc[:-3]  # a rate is missing for the last days: the last known one applies
    eng = engine(KO=a, SAP=b, **{"EURUSD=X": fx})
    eng.currency_lookup = {"KO": "USD", "SAP": "EUR"}.get
    end = a.index[-1] + pd.Timedelta(days=1)
    out = eng.get_ohlc("KO:SAP", start=a.index[0], end=end)
    assert out["close"].iloc[-1] == pytest.approx(a["close"].iloc[-1] / (b["close"].iloc[-1] * 1.25))
    with pytest.raises(DataUnavailable, match="intraday"):
        eng.get_ohlc("KO:SAP", "1h", start=a.index[0], end=end)


def test_pair_without_known_currency_or_rate_is_refused():
    a, b = make_ohlc(30, seed=1), make_ohlc(30, seed=2)
    end = a.index[-1] + pd.Timedelta(days=1)
    eng = engine(KO=a, SAP=b)
    eng.currency_lookup = {"KO": "USD"}.get  # SAP unknown
    with pytest.raises(DataUnavailable, match="currencies"):
        eng.get_ohlc("KO:SAP", start=a.index[0], end=end)
    eng.currency_lookup = {"KO": "USD", "SAP": "EUR"}.get  # no FX series
    with pytest.raises(DataUnavailable, match="exchange rate"):
        eng.get_ohlc("KO:SAP", start=a.index[0], end=end)


def test_quote_currency():
    from tradebot.data.currency import quote_currency

    assert quote_currency("BTC/USDT") == ("USD", 1.0)
    assert quote_currency("ETH/EUR:EUR") == ("EUR", 1.0)
    assert quote_currency("BP.L", {"BP.L": "GBp"}.get) == ("GBP", 0.01)


# -- rolling, cointegration, health, finder, two-legged backtest (#117) ---------------------------------------
def synthetic_pair(n=900, seed=5, phi=0.85, beta=1.5, noise=0.01):
    rng = np.random.default_rng(seed)
    base = np.cumsum(rng.normal(0, 0.01, n))
    gap = np.zeros(n)
    for i in range(1, n):
        gap[i] = phi * gap[i - 1] + rng.normal(0, noise)
    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    return pd.Series(100 * np.exp(beta * base + gap), idx), pd.Series(100 * np.exp(base), idx)


def frame_of(close):
    return pd.DataFrame({"open": close, "high": close, "low": close, "close": close})


def test_rolling_uses_only_earlier_candles():
    from tradebot.analysis.pairs import rolling_pair

    a, b = synthetic_pair()
    r1 = rolling_pair(a, b, 60)
    a2 = a.copy()
    a2.iloc[-1] *= 1.5  # a change in the last candle
    r2 = rolling_pair(a2, b, 60)
    assert r1.beta.iloc[-1] == r2.beta.iloc[-1]  # β for the last bar doesn't see that bar
    assert r2.zscore.iloc[-1] > r1.zscore.iloc[-1]
    assert r1.beta.iloc[:60].isna().all() and r1.beta.iloc[100:].mean() == pytest.approx(1.5, abs=0.3)


def test_cointegration_tells_pairs_from_random_walks():
    from tradebot.analysis.pairs import cointegration

    a, b = synthetic_pair()
    good = cointegration(a, b)
    assert good.passed and good.level <= 0.05 and good.beta == pytest.approx(1.5, abs=0.2)
    rng = np.random.default_rng(9)
    idx = a.index
    x, y = (pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx)))), idx) for _ in range(2))
    assert cointegration(x, y).stat > good.stat
    with pytest.raises(ValueError):
        cointegration(a.iloc[:10], b.iloc[:10])


def test_pair_health_flags_a_pair_that_decoupled():
    from tradebot.analysis.pairs import pair_health

    a, b = synthetic_pair()
    assert not pair_health(a, b).broken
    broken = a.copy()
    broken.iloc[-60:] *= np.exp(np.linspace(0, 0.6, 60))  # a walks off for good
    h = pair_health(broken, b)
    assert h.broken and h.reasons


def test_find_pairs_confirms_on_later_data_and_counts_chance():
    from tradebot.analysis.pairs import find_pairs

    a, b = synthetic_pair()
    rng = np.random.default_rng(3)
    loner = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(a)))), a.index)
    scan = find_pairs({"AAA": a, "BBB": b, "CCC": loner}, min_correlation=0.3)
    assert [c.symbol for c in scan.found] == ["AAA:BBB"]
    assert scan.tested + scan.skipped == 3 and scan.expected_by_chance == pytest.approx(scan.tested * 0.05)
    assert scan.found[0].half_life is not None
    with pytest.raises(ValueError):
        find_pairs({"AAA": a.iloc[:50], "BBB": b.iloc[:50]})


def test_pair_backtest_is_market_neutral_and_pays_costs():
    from tradebot.backtest.pairs import pair_backtest

    a, b = synthetic_pair(n=1500, seed=11)
    res = pair_backtest(frame_of(a), frame_of(b), window=60, fee=0.0, borrow=0.0)
    assert res.metrics["trades"] > 5 and res.metrics["total_return"] > 0  # the gap comes back, so it pays
    costly = pair_backtest(frame_of(a), frame_of(b), window=60, fee=0.005, borrow=0.1)
    assert costly.metrics["total_return"] < res.metrics["total_return"]
    assert set(res.positions.unique()) <= {-1.0, 0.0, 1.0}
    assert res.positions.iloc[:60].eq(0).all()  # nothing before the window is full
    with pytest.raises(ValueError):
        pair_backtest(frame_of(a), frame_of(b), entry=1, exit=2)


def test_p_value_and_multiple_testing_correction():
    from tradebot.analysis.pairs import CRITICAL, p_value, significant

    assert p_value(CRITICAL[0.05]) == pytest.approx(0.05) and p_value(CRITICAL[0.01]) == pytest.approx(0.01)
    assert 0.01 < p_value(-3.6) < 0.05 and p_value(-2.0) == 1.0 and p_value(-6.0) < 0.001
    p = [0.001, 0.009, 0.02, 0.04, 0.2]
    assert significant(p, "none", 0.05) == [True, True, True, True, False]
    assert significant(p, "bonferroni", 0.05) == [True, True, False, False, False]  # p <= 0.05 / 5
    assert significant(p, "fdr", 0.05) == [True, True, True, True, False]  # BH: largest k with p_k <= 0.05 k / 5
    assert significant([0.04, 0.5, 0.6, 0.7], "fdr", 0.05) == [False] * 4  # 0.04 > 0.05 / 4
    with pytest.raises(ValueError):
        significant(p, "magic", 0.05)


def test_find_pairs_group_prefilter_and_correction():
    from tradebot.analysis.pairs import find_pairs

    a, b = synthetic_pair()
    rng = np.random.default_rng(3)
    loners = {f"L{i}": pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(a)))), a.index) for i in range(6)}
    closes = {"AAA": a, "BBB": b, **loners}
    plain = find_pairs(closes, min_correlation=-1, correction="none")
    fdr = find_pairs(closes, min_correlation=-1, correction="fdr")
    bonf = find_pairs(closes, min_correlation=-1, correction="bonferroni")
    assert plain.passed_train >= fdr.passed_train >= bonf.passed_train
    assert "AAA:BBB" in [c.symbol for c in bonf.found]  # a real pair survives the strictest correction
    sectors = {"AAA": "x", "BBB": "x", "L0": "x", "L1": "y"}  # L2.. unknown: never paired
    grouped = find_pairs(closes, min_correlation=-1, groups=sectors)
    assert grouped.tested == 3 and grouped.skipped == len(closes) * (len(closes) - 1) // 2 - 3
    assert "AAA:BBB" in [c.symbol for c in grouped.found]


def test_kalman_beta_follows_a_changing_hedge_ratio():
    from tradebot.analysis.pairs import kalman_pair

    a, b = synthetic_pair(n=1500, beta=1.5)
    k = kalman_pair(a, b)
    assert k.beta.iloc[:30].isna().all() and abs(k.beta.iloc[-1] - 1.5) < 0.3
    z = k.zscore.dropna()
    assert abs(z.mean()) < 0.5 and 0.5 < z.std() < 2  # roughly standardised forecast errors
    # no look-ahead: changing the last candle leaves every earlier value alone
    a2 = a.copy()
    a2.iloc[-1] *= 3
    k2 = kalman_pair(a2, b)
    pd.testing.assert_series_equal(k.beta.iloc[:-1], k2.beta.iloc[:-1])
    pd.testing.assert_series_equal(k.zscore.iloc[:-1], k2.zscore.iloc[:-1])


def test_pair_backtest_with_kalman_beta():
    from tradebot.backtest.pairs import pair_backtest

    a, b = synthetic_pair(n=1500, seed=11)
    res = pair_backtest(frame_of(a), frame_of(b), fee=0.0, borrow=0.0, method="kalman")
    assert res.params["method"] == "kalman" and res.metrics["trades"] > 5 and res.metrics["total_return"] > 0
    with pytest.raises(ValueError):
        pair_backtest(frame_of(a), frame_of(b), method="magic")


def test_pair_view_draws_without_network(tmp_path):
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PyQt6")
    from PyQt6.QtWidgets import QApplication

    from tradebot.gui.pairs import PairView
    from tradebot.gui.theme import ThemeManager

    app = QApplication.instance() or QApplication([])  # noqa: F841 - must stay referenced
    a, b = synthetic_pair()
    view = PairView(engine(KO=frame_of(a), PEP=frame_of(b)), ThemeManager(user_dir=tmp_path), ["KO", "PEP"])
    view._show("KO", "PEP", 60, "rolling", a, b)
    view._show("KO", "PEP", 60, "kalman", a, b)
    assert "cointegration" in view.numbers.text() and "half-life" in view.numbers.text()
    assert len(view.prices.getPlotItem().listDataItems()) == 2 and len(view.zplot.getPlotItem().listDataItems()) == 1
    view._show("KO", "PEP", 60, "rolling", a.iloc[:20], b.iloc[:20])  # too short: says so instead of failing
    assert "need at least" in view.numbers.text()


class StubData:
    def __init__(self, a, b):
        self.a, self.b, self.upto = frame_of(a), frame_of(b), len(a)

    def pair_legs(self, legs, timeframe="1d", start=None, end=None, asset_class=None, bars=None):
        return self.a.iloc[:self.upto], self.b.iloc[:self.upto]


def stub_broker(data, fail_on=None):
    from tradebot.brokers.base import BaseBroker
    from tradebot.core.models import Balance, Order, OrderStatus, OrderType, Ticker

    class Stub(BaseBroker):
        name = "stub-pairs"

        def __init__(self):
            super().__init__(data=data)
            self.orders = []

        def get_ticker(self, symbol):
            return Ticker(symbol, 1.0)

        def place_order(self, symbol, side, amount, type=OrderType.MARKET, price=None):
            if symbol == fail_on and side.value == "sell":
                raise RuntimeError("can't short on spot")
            self.orders.append((symbol, side.value, amount))
            return Order(str(len(self.orders)), symbol, side, type, amount, price, OrderStatus.CLOSED)

        def cancel_order(self, order_id, symbol=None):
            return True

        def get_balance(self):
            return {}

    return Stub()


def test_pair_bot_follows_the_backtest_and_trades_both_legs():
    from tradebot.backtest.pairs import pair_backtest
    from tradebot.live import PairBot

    a, b = synthetic_pair(n=700, seed=11)
    data = StubData(a, b)
    broker = stub_broker(data)
    bot = PairBot(broker, ("AAA", "BBB"), "1d", stake=1000.0, window=60, dry_run=False)
    for k in range(100, len(a) + 1):
        data.upto = k
        bot.step()
    ref = pair_backtest(frame_of(a), frame_of(b), window=60).positions
    # the bot decides on the same candles the backtest does: same position on the last bar it saw
    assert bot.side == ref.iloc[-1]
    opens = [e for e in bot.events if e.action.startswith("open")]
    closes = [e for e in bot.events if e.action == "close"]
    assert opens and len(opens) - len(closes) in (0, 1)
    first = opens[0]
    assert first.amount_a * first.price_a + first.amount_b * first.price_b == pytest.approx(1000.0)
    side_a = [o[1] for o in broker.orders if o[0] == "AAA"]
    side_b = [o[1] for o in broker.orders if o[0] == "BBB"]
    assert len(side_a) == len(side_b) == len(opens) + len(closes)
    assert all(x != y for x, y in zip(side_a, side_b))  # always one bought, the other sold
    assert broker.orders[0][:2] == (("AAA", "buy") if first.action == "open long" else ("BBB", "buy"))  # the long leg


def test_pair_bot_dry_run_places_nothing_and_a_failed_short_is_undone():
    from tradebot.live import PairBot

    a, b = synthetic_pair(n=700, seed=11)
    data = StubData(a, b)
    quiet = stub_broker(data)
    bot = PairBot(quiet, ("AAA", "BBB"), "1d", window=60)
    for k in range(100, len(a) + 1):
        data.upto = k
        bot.step()
    assert quiet.orders == [] and any(e.action != "hold" for e in bot.events)

    spot = stub_broker(data, fail_on="BBB")  # shorting BBB isn't possible; AAA short works but BBB short doesn't
    live = PairBot(spot, ("AAA", "BBB"), "1d", window=60, dry_run=False)
    with pytest.raises(RuntimeError):
        for k in range(100, len(a) + 1):
            data.upto = k
            live.step()
    (s1, d1, x1), (s2, d2, x2) = spot.orders[-2:]
    assert (s1, d1, d2) == ("AAA", "buy", "sell") and s2 == "AAA" and x1 == x2  # bought AAA, short failed: undone
    with pytest.raises(ValueError):
        PairBot(spot, ("AAA", "BBB"), entry=1, exit=2)
