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
    view._show("KO", "PEP", 60, a, b)
    assert "cointegration" in view.numbers.text() and "half-life" in view.numbers.text()
    assert len(view.prices.getPlotItem().listDataItems()) == 2 and len(view.zplot.getPlotItem().listDataItems()) == 1
    view._show("KO", "PEP", 60, a.iloc[:20], b.iloc[:20])  # too short: says so instead of failing
    assert "need at least" in view.numbers.text()
