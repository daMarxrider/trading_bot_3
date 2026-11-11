# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Market index universes for the Scanner (#110): ``index_symbols("sp500")`` -> provider-ready symbols.

Constituents come from public lists (Wikipedia tables, CoinGecko for crypto), are kept in the cache folder's
``symbols/`` for a week, and fall back to the stale cache when offline and to a small bundled list when there's
no cache, so an index is never empty. At most :data:`MAX_SYMBOLS` symbols per index.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)

MAX_SYMBOLS = 4000
REFRESH_SECONDS = 7 * 24 * 3600
_UA = {"User-Agent": "Mozilla/5.0 (tradebot index lists)"}
STABLECOINS = {"USDT", "USDC", "DAI", "BUSD", "TUSD", "FDUSD", "USDE", "USDS", "PYUSD"}


def _get(url: str) -> str:
    import urllib.request

    with urllib.request.urlopen(urllib.request.Request(url, headers=_UA), timeout=15) as r:
        return r.read().decode("utf-8", "replace")


def wiki_column(page: str, columns: tuple[str, ...], suffix: str = "") -> list[str]:
    """The first Wikipedia table of ``page`` with one of ``columns``: its tickers as Yahoo symbols."""
    import pandas as pd

    for table in pd.read_html(StringIO(_get(f"https://en.wikipedia.org/wiki/{page}"))):
        table.columns = [" ".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in table.columns]
        for col in table.columns:
            if col in columns:
                symbols = [yahoo_symbol(str(s), suffix) for s in table[col].dropna()]
                return [s for s in symbols if s]
    raise ValueError(f"no table with {columns} on {page}")


def yahoo_symbol(ticker: str, suffix: str = "") -> str:
    """``BRK.B`` -> ``BRK-B``; exchange suffixes (``.DE``) are added unless the ticker has one."""
    ticker = ticker.strip().split(":")[-1].split("[")[0].strip().upper()
    if not ticker or ticker == "NAN":
        return ""
    if suffix == ".DE":  # Xetra: the Wikipedia ticker may name another venue (AIR.PA)
        return ticker.split(".")[0] + suffix
    if suffix and not ticker.endswith(suffix):
        return ticker.replace(".", "-") + suffix
    return ticker if suffix else ticker.replace(".", "-")


def coingecko_top(n: int = 100) -> list[str]:
    rows = json.loads(_get("https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc"
                           f"&per_page={n + 20}&page=1"))
    coins = [r["symbol"].upper() for r in rows if r["symbol"].upper() not in STABLECOINS]
    return [f"{c}/USDT" for c in dict.fromkeys(coins)][:n]


@dataclass(frozen=True)
class Index:
    key: str
    ticker: str
    title: str
    icon: str
    fetch: Callable[[], list[str]]


INDICES: dict[str, Index] = {i.key: i for i in (
    Index("sp500", "^GSPC", "S&P 500", "🇺🇸", lambda: wiki_column("List_of_S%26P_500_companies", ("Symbol",))),
    Index("nasdaq100", "^NDX", "Nasdaq 100", "🇺🇸", lambda: wiki_column("Nasdaq-100", ("Ticker", "Symbol"))),
    Index("dow30", "^DJI", "Dow Jones Industrial Average", "🇺🇸",
          lambda: wiki_column("Dow_Jones_Industrial_Average", ("Symbol",))),
    Index("dax40", "^GDAXI", "DAX 40", "🇩🇪", lambda: wiki_column("DAX", ("Ticker",), ".DE")),
    Index("ftse100", "^FTSE", "FTSE 100", "🇬🇧", lambda: wiki_column("FTSE_100_Index", ("Ticker", "EPIC"), ".L")),
    Index("crypto_top100", "—", "Top 100 crypto by market cap", "₿", coingecko_top),
)}

#: Small lists for when there's neither network nor cache (the biggest members of each index)
BUNDLED: dict[str, list[str]] = {
    "sp500": "AAPL MSFT NVDA AMZN GOOGL META BRK-B TSLA AVGO JPM LLY V XOM UNH MA COST HD PG JNJ ABBV WMT NFLX BAC "
             "CRM ORCL KO CVX MRK AMD PEP TMO ADBE LIN MCD CSCO ACN ABT WFC DIS GE IBM INTU QCOM TXN CAT VZ AMGN "
             "PFE NOW".split(),
    "nasdaq100": "AAPL MSFT NVDA AMZN GOOGL META AVGO TSLA COST NFLX AMD PEP ADBE CSCO TMUS LIN INTU QCOM TXN AMGN "
                 "AMAT ISRG BKNG HON CMCSA PANW MU ADP GILD LRCX VRTX SBUX ADI MDLZ KLAC INTC".split(),
    "dow30": "AAPL AMGN AXP BA CAT CRM CSCO CVX DIS GS HD HON IBM JNJ JPM KO MCD MMM MRK MSFT NKE NVDA PG SHW TRV "
             "UNH V VZ WMT AMZN".split(),
    "dax40": [f"{s}.DE" for s in "SAP SIE ALV DTE AIR MBG MUV2 BMW VOW3 BAS BAYN ADS DHL IFX DB1 HEI RWE EOAN "
                                 "DBK CBK ENR FRE HNR1 MRK PAH3 QIA RHM SHL SY1 VNA ZAL BEI CON 1COV DTG BNR "
                                 "P911 SRT3 HEN3 MTX".split()],
    "ftse100": [f"{s}.L" for s in "AZN SHEL HSBA ULVR BP GSK RIO DGE BATS LSEG REL NG LLOY BARC VOD GLEN CPG "
                                  "AAL PRU NWG RR TSCO BA STAN LGEN BT-A".split()],
    "crypto_top100": [f"{c}/USDT" for c in "BTC ETH XRP BNB SOL DOGE ADA TRX LINK AVAX XLM SUI BCH HBAR LTC "
                                           "DOT UNI AAVE NEAR ATOM".split()],
}


def _cap(key: str, symbols: list[str]) -> list[str]:
    symbols = list(dict.fromkeys(symbols))
    if len(symbols) > MAX_SYMBOLS:
        log.warning("index %s has %d symbols; using the first %d", key, len(symbols), MAX_SYMBOLS)
        symbols = symbols[:MAX_SYMBOLS]
    return symbols


def index_symbols(key: str, cache_dir: Path | None = None, now: float | None = None) -> list[str]:
    """The constituents of index ``key``: a fresh cache, else a fetch, else a stale cache, else the bundled list."""
    index = INDICES[key]
    now = time.time() if now is None else now
    path = cache_dir / f"index_{key}.json" if cache_dir else None
    cached: dict | None = None
    if path is not None:
        try:
            cached = json.loads(path.read_text())
            if now - float(cached["time"]) < REFRESH_SECONDS and cached["symbols"]:
                return _cap(key, cached["symbols"])
        except (OSError, ValueError, KeyError, TypeError):
            cached = None
    try:
        symbols = _cap(key, index.fetch())
        if not symbols:
            raise ValueError("empty list")
        if path is not None:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"time": now, "symbols": symbols}))
            except OSError as exc:
                log.warning("could not cache index %s: %s", key, exc)
        return symbols
    except Exception as exc:
        log.info("index %s: %s; using %s", key, exc, "the cached list" if cached else "the bundled list")
        return _cap(key, cached["symbols"] if cached and cached.get("symbols") else BUNDLED[key])
