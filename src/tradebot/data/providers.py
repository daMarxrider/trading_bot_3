# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Historical OHLC data providers.

Each provider returns a normalized frame (see :func:`tradebot.core.models.normalize_ohlc`)
or raises; the :class:`~tradebot.data.engine.MarketDataEngine` chains them.
Symbols are canonical: ``AAPL`` / ``SAP.DE`` for stocks, ``BTC/USDT`` for crypto.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import ClassVar

import pandas as pd
import requests

from tradebot.core.models import AssetClass, normalize_ohlc, to_ohlc_frame
from tradebot.core.timeframes import resample, to_timedelta

log = logging.getLogger(__name__)


def quiet_yfinance() -> None:
    """Hide yfinance's internal deprecation warnings (e.g. pandas ``inplace=`` notices) from users.

    Importing yfinance installs its own ``default`` filter for its DeprecationWarnings
    at the front of the filter list, so ours must be added *after* that import to win.
    """
    import warnings

    import yfinance  # noqa: F401

    warnings.filterwarnings("ignore", category=DeprecationWarning, module="yfinance")


class DataUnavailable(RuntimeError):
    pass


class BaseDataProvider(ABC):
    name: ClassVar[str]
    asset_classes: ClassVar[frozenset[AssetClass]]
    timeframes: ClassVar[frozenset[str]]

    def supports(self, asset_class: AssetClass, timeframe: str) -> bool:
        return asset_class in self.asset_classes

    @abstractmethod
    def fetch(self, symbol: str, timeframe: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """Candles with ``start <= index < end``."""

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name}>"


class CCXTDataProvider(BaseDataProvider):
    """Public (unauthenticated) candles from any ccxt exchange, with pagination."""

    asset_classes = frozenset({AssetClass.CRYPTO})
    timeframes = frozenset({"1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"})

    def __init__(self, exchange_id: str = "binance", page_limit: int = 1000, exchange=None):
        import ccxt

        self.name = f"ccxt:{exchange_id}"
        self.exchange = exchange or getattr(ccxt, exchange_id)({"enableRateLimit": True})
        self.page_limit = page_limit

    def fetch(self, symbol, timeframe, start, end):
        since, end_ms = int(start.timestamp() * 1000), int(end.timestamp() * 1000)
        step = int(to_timedelta(timeframe).total_seconds() * 1000)
        rows: list = []
        while since < end_ms:
            page = self.exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=self.page_limit)
            if not page:
                break
            rows.extend(r for r in page if r[0] < end_ms)
            nxt = page[-1][0] + step
            if nxt <= since:
                break
            since = nxt
        if not rows:
            raise DataUnavailable(f"{self.name}: no candles for {symbol} {timeframe}")
        return to_ohlc_frame(rows)


class YFinanceProvider(BaseDataProvider):
    """Yahoo Finance via ``yfinance`` - stocks, ETFs and major crypto (``BTC/USDT`` -> ``BTC-USD``)."""

    name = "yfinance"
    asset_classes = frozenset({AssetClass.STOCK, AssetClass.CRYPTO})
    timeframes = frozenset({"1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"})
    _INTERVALS = {"1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m", "1h": "1h", "4h": "1h", "1d": "1d", "1w": "1wk"}

    @staticmethod
    def map_symbol(symbol: str) -> str:
        if "/" not in symbol:
            return symbol
        base, quote = symbol.split("/")
        # prices are assumed identical across venues, so a stablecoin quote maps to USD
        quote = "USD" if quote in {"USDT", "USDC", "BUSD", "TUSD", "FDUSD"} else quote
        return f"{base}-{quote}"

    def fetch(self, symbol, timeframe, start, end):
        import yfinance as yf

        raw = yf.Ticker(self.map_symbol(symbol)).history(
            start=start, end=end, interval=self._INTERVALS[timeframe], auto_adjust=False, actions=False
        )
        if raw is None or raw.empty:
            raise DataUnavailable(f"yfinance: no candles for {symbol}")
        df = normalize_ohlc(raw)
        return resample(df, timeframe) if timeframe == "4h" else df


class AlphaVantageProvider(BaseDataProvider):
    """Alpha Vantage daily candles (needs ``ALPHAVANTAGE_API_KEY``; free tier is rate limited)."""

    name = "alphavantage"
    asset_classes = frozenset({AssetClass.STOCK, AssetClass.CRYPTO})
    timeframes = frozenset({"1d"})
    URL = "https://www.alphavantage.co/query"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("ALPHAVANTAGE_API_KEY")

    def supports(self, asset_class, timeframe):
        return bool(self.api_key) and timeframe in self.timeframes and super().supports(asset_class, timeframe)

    def fetch(self, symbol, timeframe, start, end):
        if "/" in symbol:
            base, quote = symbol.split("/")
            params = {"function": "DIGITAL_CURRENCY_DAILY", "symbol": base, "market": quote.replace("USDT", "USD")}
        else:
            params = {"function": "TIME_SERIES_DAILY", "symbol": symbol, "outputsize": "full"}
        data = requests.get(self.URL, params={**params, "apikey": self.api_key}, timeout=30).json()
        series = next((v for k, v in data.items() if k.startswith("Time Series")), None)
        if not series:
            raise DataUnavailable(f"alphavantage: {data.get('Note') or data.get('Information') or data}")
        df = pd.DataFrame.from_dict(series, orient="index")
        # keys look like "1. open"; crypto adds the market e.g. "1a. open (USD)" in older responses
        df.columns = [c.split(". ", 1)[1].split(" ")[0] for c in df.columns]
        df = df.loc[:, ~df.columns.duplicated()]
        df.index = pd.to_datetime(df.index)
        df = normalize_ohlc(df)
        return df[(df.index >= start) & (df.index < end)]


class FMPProvider(BaseDataProvider):
    """Financial Modeling Prep end-of-day candles (needs ``FMP_API_KEY``)."""

    name = "fmp"
    asset_classes = frozenset({AssetClass.STOCK, AssetClass.CRYPTO})
    timeframes = frozenset({"1d"})
    URL = "https://financialmodelingprep.com/stable/historical-price-eod/full"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("FMP_API_KEY")

    def supports(self, asset_class, timeframe):
        return bool(self.api_key) and timeframe in self.timeframes and super().supports(asset_class, timeframe)

    def fetch(self, symbol, timeframe, start, end):
        sym = symbol.replace("/", "").replace("USDT", "USD")
        params = {"symbol": sym, "from": start.date().isoformat(), "to": end.date().isoformat(), "apikey": self.api_key}
        data = requests.get(self.URL, params=params, timeout=30).json()
        rows = data.get("historical", []) if isinstance(data, dict) else data
        if not rows:
            raise DataUnavailable(f"fmp: no candles for {symbol}: {data if isinstance(data, dict) else ''}")
        df = pd.DataFrame(rows).set_index("date")
        df.index = pd.to_datetime(df.index)
        df = normalize_ohlc(df)
        return df[(df.index >= start) & (df.index < end)]
