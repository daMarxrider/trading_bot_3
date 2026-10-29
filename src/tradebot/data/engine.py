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

With a :class:`~tradebot.data.cache.DiskCache` (#63), candles also survive restarts: a request reads the
cached series and asks the providers only for what's missing, plus the last two cached candles again (the
newest may have been incomplete; the one before shows whether a stock's history was split- or
dividend-adjusted since, which makes the engine fetch it again in full). Offline, the cached candles are
returned with ``df.attrs["offline"] = True``.
"""

from __future__ import annotations

import logging
import time

import pandas as pd

from tradebot.core.models import AssetClass, guess_asset_class
from tradebot.core.timeframes import to_timedelta
from tradebot.data.cache import DiskCache
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
ADJUSTED = 0.005  #: a re-fetched complete candle differing from the cached one by more: history was adjusted
FRESH_FOR = 60.0  #: seconds a series counts as just refreshed, so its last candles aren't fetched again


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
    def __init__(self, providers: list[BaseDataProvider] | None = None, cache: bool = True,
                 disk: DiskCache | bool | None = True):
        """``disk``: True = the cache as config.toml sets it up, False/None = memory only."""
        self.providers = providers if providers is not None else default_providers()
        self._cache: dict[tuple, pd.DataFrame] | None = {} if cache else None
        self.disk = DiskCache.from_config() if disk is True else (disk or None)
        self._refreshed: dict[tuple[str, str], float] = {}  # series -> when its newest candles were fetched

    def clear_memory(self) -> None:
        """Forget the candles held in memory, e.g. after the disk cache was cleared."""
        if self._cache is not None:
            self._cache.clear()

    def peek(self, symbol: str, timeframe: str = "1d", start=None, end=None, bars: int | None = None) -> pd.DataFrame | None:
        """What the disk cache has for a request, without any network: for showing a chart right away."""
        stored = self.disk.load(symbol, timeframe) if self.disk else None
        if stored is None:
            return None
        df = stored[0]
        if end is not None:
            df = df[df.index < _utc(end)]
        df = df[df.index >= _utc(start)] if start is not None else df.iloc[-(bars or 500):]
        return df if not df.empty else None

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

        stored = self.disk.load(symbol, timeframe) if self.disk else None
        cached, listed = stored if stored is not None else (None, None)
        fresh = time.monotonic() - self._refreshed.get((symbol, timeframe), -FRESH_FOR) < FRESH_FOR
        result, errors, fetched = self._fetch(symbol, timeframe, asset_class, start, end, step, cached, listed, fresh)
        if cached is not None and _adjusted(cached, result):
            log.info("%s %s history changed (split or dividend adjustment): fetching it again", symbol, timeframe)
            cached = listed = None
            result, errors, fetched = self._fetch(symbol, timeframe, asset_class, start, end, step, None, None, False)
        if errors and cached is not None:  # offline or failing: fill in what the cache has
            part = cached[(cached.index >= start) & (cached.index < end)]
            result = part if result is None else result.combine_first(part)
        if result is None or result.empty:
            raise DataUnavailable(f"no data for {symbol} {timeframe}: " + "; ".join(errors or ["no provider"]))
        if self.disk is not None and fetched:
            self._store(symbol, timeframe, result, cached, listed, start, step, errors)
            if not errors and end - result.index[-1] <= _slack(step):
                self._refreshed[symbol, timeframe] = time.monotonic()
        if errors and cached is not None:
            result = result.copy()
            result.attrs["offline"] = True
        if self._cache is not None and not errors:
            self._cache[key] = result
        return result

    def _fetch(self, symbol, timeframe, asset_class, start, end, step, cached, listed, fresh):
        """``(candles, errors, fetched)`` for ``[start, end)``: the cached candles except the last two (all of
        them if ``fresh``: just refreshed), plus the missing head and tail from the providers. Not before
        ``listed``, where the history begins."""
        result: pd.DataFrame | None = None
        refresh_from = None  # where the re-fetched last two cached candles start, if the request covers them
        if cached is not None and len(cached) > 2:
            known = cached if fresh else cached.iloc[:-2]
            known = known[(known.index >= start) & (known.index < end)]
            result = known if not known.empty else None
            if not fresh and start <= cached.index[-2] < end:
                refresh_from = cached.index[-2]
        fetch_from = max(start, listed) if listed is not None else start
        errors: list[str] = []
        fetched = False
        for provider in self.providers:
            if not provider.supports(asset_class, timeframe):
                continue
            gaps = _missing_ranges(result, fetch_from, end, step)
            if refresh_from is not None and (result is None or result.index[-1] < refresh_from) \
                    and not any(gap_end >= end for _, gap_end in gaps):
                gaps.append((refresh_from, end))  # within the slack, but they must come back
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
                    fetched = True
                    # earlier providers win on overlapping bars
                    result = part if result is None else result.combine_first(part)
        return result, errors, fetched

    def _store(self, symbol, timeframe, result, cached, listed, start, step, errors):
        """Merge new candles into the cached series. The series stays one contiguous run: new candles that
        don't touch it replace it, so a hole in the middle never needs filling."""
        series = result
        slack = _slack(step)
        if cached is not None and not cached.empty and (
                result.index[0] <= cached.index[-1] + slack and result.index[-1] >= cached.index[0] - slack):
            series = result.combine_first(cached)  # new candles win
        else:
            listed = None
        if not errors and result.index[0] - start > slack:  # asked for older candles, none exist
            listed = result.index[0] if listed is None else min(listed, result.index[0])
        self.disk.save(symbol, timeframe, series, listed)

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
    slack = _slack(step)
    if df.index[0] - start > slack:
        gaps.append((start, df.index[0]))
    if end - df.index[-1] > slack:
        gaps.append((df.index[-1] + step, end))
    return gaps


def _slack(step) -> pd.Timedelta:
    return max(step * 3, pd.Timedelta(days=4))


def _adjusted(cached: pd.DataFrame, fresh: pd.DataFrame | None) -> bool:
    """Whether the providers now report a different close for the last complete cached candle."""
    if fresh is None or len(cached) < 2:
        return False
    t = cached.index[-2]
    if t not in fresh.index:
        return False
    old, new = float(cached["close"].iloc[-2]), float(fresh.loc[t, "close"])
    return old > 0 and abs(new / old - 1) > ADJUSTED
