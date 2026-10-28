# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Market data engine: one call, many providers, complete history.

Prices for a symbol are assumed identical across venues, so candles from
different providers can be stitched together. The engine asks providers in
order of preference and only requests the *missing* head/tail ranges from
the next provider - e.g. Binance supplies recent BTC/USDT candles, Yahoo
fills in the years before the Binance listing.
"""

from __future__ import annotations

import logging

import pandas as pd

from tradebot.core.models import AssetClass, guess_asset_class
from tradebot.core.timeframes import to_timedelta
from tradebot.data.providers import (
    AlphaVantageProvider,
    BaseDataProvider,
    CCXTDataProvider,
    DataUnavailable,
    FMPProvider,
    YFinanceProvider,
)

log = logging.getLogger(__name__)


def default_providers() -> list[BaseDataProvider]:
    """Preference order: exchanges first for crypto, then Yahoo, then keyed APIs (skipped without a key)."""
    providers: list[BaseDataProvider] = []
    for exchange_id in ("binance", "kucoin", "kraken"):
        try:
            providers.append(CCXTDataProvider(exchange_id))
        except Exception as exc:  # pragma: no cover - ccxt missing/broken
            log.debug("ccxt %s unavailable: %s", exchange_id, exc)
    providers += [YFinanceProvider(), AlphaVantageProvider(), FMPProvider()]
    return providers


class MarketDataEngine:
    def __init__(self, providers: list[BaseDataProvider] | None = None, cache: bool = True):
        self.providers = providers if providers is not None else default_providers()
        self._cache: dict[tuple, pd.DataFrame] | None = {} if cache else None

    def get_ohlc(
        self,
        symbol: str,
        timeframe: str = "1d",
        start: pd.Timestamp | str | None = None,
        end: pd.Timestamp | str | None = None,
        asset_class: AssetClass | None = None,
    ) -> pd.DataFrame:
        """OHLC candles for ``[start, end)``; defaults to the last 500 bars."""
        step = to_timedelta(timeframe)
        end = _utc(end) if end is not None else pd.Timestamp.now(tz="UTC")
        start = _utc(start) if start is not None else end - 500 * step
        asset_class = asset_class or guess_asset_class(symbol)
        key = (symbol, timeframe, start, end.floor(step))
        if self._cache is not None and key in self._cache:
            return self._cache[key]

        result: pd.DataFrame | None = None
        errors: list[str] = []
        for provider in self.providers:
            if not provider.supports(asset_class, timeframe):
                continue
            gaps = _missing_ranges(result, start, end, step)
            if not gaps:
                break
            for gap_start, gap_end in gaps:
                try:
                    part = provider.fetch(symbol, timeframe, gap_start, gap_end)
                except Exception as exc:
                    errors.append(f"{provider.name}: {exc}")
                    log.info("provider %s failed for %s: %s", provider.name, symbol, exc)
                    break
                part = part[(part.index >= start) & (part.index < end)]
                if not part.empty:
                    # earlier providers win on overlapping bars
                    result = part if result is None else result.combine_first(part)
        if result is None or result.empty:
            raise DataUnavailable(f"no data for {symbol} {timeframe}: " + "; ".join(errors or ["no provider"]))
        if self._cache is not None:
            self._cache[key] = result
        return result

    def latest_price(self, symbol: str) -> float:
        return float(self.get_ohlc(symbol, "1d", start=pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=10))["close"].iloc[-1])


def _utc(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _missing_ranges(df: pd.DataFrame | None, start, end, step) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Head/tail ranges not yet covered. Tolerates a few bars of slack (weekends, holidays, listing gaps)."""
    if df is None or df.empty:
        return [(start, end)]
    gaps = []
    slack = max(step * 3, pd.Timedelta(days=4))
    if df.index[0] - start > slack:
        gaps.append((start, df.index[0]))
    if end - df.index[-1] > slack:
        gaps.append((df.index[-1] + step, end))
    return gaps
