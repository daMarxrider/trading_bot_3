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
