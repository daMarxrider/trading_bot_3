# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Market data engine: one call, many providers, complete history.

Prices for a symbol are assumed identical across venues, so candles from
different providers can be stitched together. The engine asks providers in
order of preference and only requests the *missing* head/tail ranges from
the next provider - e.g. Binance supplies recent BTC/USDT candles, Yahoo
fills in the years before the Binance listing.

Requests are either a date range or a candle count (``bars=N``). A count is
exact: markets that don't trade around the clock (weekends, trading hours)
have far fewer candles than ``N x interval`` suggests, so the engine widens
the range until it has N candles or the history runs out.
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
MAX_BAR_ROUNDS = 6  #: widening steps for a ``bars=N`` request before settling for what exists


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
        bars: int | None = None,
    ) -> pd.DataFrame:
        """OHLC candles for ``[start, end)``, or the last ``bars`` candles before ``end`` when no
        ``start`` is given (default: 500). ``start`` wins over ``bars``.

        A ``bars`` request returns fewer candles only when no more history exists (provider limit
        or listing date); that is logged as a warning.
        """
        step = to_timedelta(timeframe)
        end = _utc(end) if end is not None else pd.Timestamp.now(tz="UTC")
        asset_class = asset_class or guess_asset_class(symbol)
        if start is None:
            return self._last_bars(symbol, timeframe, bars or 500, end, asset_class)
        start = _utc(start)
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
            limit = provider.history_limit(timeframe)
            earliest = pd.Timestamp.now(tz="UTC") - limit if limit is not None else None
            for gap_start, gap_end in gaps:
                if earliest is not None:  # ask only for what the provider can serve, instead of failing
                    if gap_end <= earliest:
                        continue
                    gap_start = max(gap_start, earliest)
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

    def _last_bars(self, symbol, timeframe, bars, end, asset_class) -> pd.DataFrame:
        step = to_timedelta(timeframe)
        span = bars * step  # exact for 24/7 markets, too short for everything else
        best: pd.DataFrame | None = None
        error: DataUnavailable | None = None
        for _ in range(MAX_BAR_ROUNDS):
            try:
                df = self.get_ohlc(symbol, timeframe, start=end - span, end=end, asset_class=asset_class)
            except DataUnavailable as exc:  # e.g. only a weekend in range: look further back
                if best is not None:
                    break  # a wider range found less than a narrower one: past the provider's history
                error, span = exc, span * 4
                continue
            if len(df) >= bars:
                return df.iloc[-bars:]
            if best is not None and len(df) <= len(best):
                break  # widening found nothing older: the history ends here
            best = df
            span *= min(max(1.5, 1.2 * bars / len(df)), 50.0)  # scale by the observed candle density
        if best is None:
            raise error or DataUnavailable(f"no data for {symbol} {timeframe}")
        log.warning("only %d of %d %s candles available for %s (provider history limit or listing date)",
                    len(best), bars, timeframe, symbol)
        return best

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
