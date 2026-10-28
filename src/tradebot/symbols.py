# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Symbol search for autocompletion: local symbols, Yahoo search and exchange crypto pairs.

:class:`SymbolSearch` merges three sources into one ranked list:

* local symbols (watchlist, portfolio): instant and offline, always ranked first
* Yahoo search by ticker or company name (stocks, ETFs, indices, funds)
* the spot pairs of a crypto exchange (ccxt ``load_markets``, loaded once and cached)

Remote sources are slow, so :meth:`SymbolSearch.local` answers immediately and
:meth:`SymbolSearch.search` adds the network results; errors there are swallowed,
so search degrades to local suggestions when offline.

Shell completion starts a new process for every TAB, so it can't keep anything in
memory: it uses :func:`yahoo_search_http` (no yfinance import), and keeps the
crypto pairs and recent queries in on-disk caches (see ``cache`` arguments).
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import astuple, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable

log = logging.getLogger(__name__)

#: Yahoo quote types the data providers can chart (no futures, options or FX)
YAHOO_TYPES = {"EQUITY": "stock", "ETF": "ETF", "INDEX": "index", "MUTUALFUND": "fund", "CRYPTOCURRENCY": "crypto"}
CRYPTO_QUOTES = ("USDT", "USDC", "USD", "EUR", "BTC")  #: pairs offered for completion, most common first


@dataclass(frozen=True)
class SymbolMatch:
    symbol: str
    name: str = ""
    exchange: str = ""
    kind: str = ""

    @property
    def label(self) -> str:
        """``AAPL — Apple Inc. · NASDAQ``"""
        details = " · ".join(p for p in (self.exchange, self.kind) if p)
        return " — ".join(p for p in (self.symbol, self.name) if p) + (f" · {details}" if details else "")


def rank(query: str, match: SymbolMatch) -> int | None:
    """Lower is better; None = no match. Exact symbol < symbol prefix < name prefix < contains."""
    q = query.strip().lower()
    symbol, name = match.symbol.lower(), match.name.lower()
    base = symbol.split("/")[0].split(".")[0].split("-")[0]  # "sol" matches "SOL/USDT", "sap" matches "SAP.DE"
    if not q:
        return None
    if symbol == q or base == q:
        return 0
    if symbol.startswith(q):
        return 1
    if name.startswith(q) or any(word.startswith(q) for word in name.split()):
        return 2
    if q in symbol or q in name:
        return 3
    return None


def yahoo_search(query: str, limit: int = 12) -> list[SymbolMatch]:
    import yfinance as yf

    return _yahoo_matches(yf.Search(query, max_results=limit, news_count=0).quotes)


YAHOO_SEARCH_URL = "https://query2.finance.yahoo.com/v1/finance/search"


def yahoo_search_http(query: str, limit: int = 12, timeout: float = 1.0) -> list[SymbolMatch]:
    """Same as :func:`yahoo_search` over plain HTTP: no ~0.5 s yfinance import, and a hard timeout."""
    import requests

    resp = requests.get(YAHOO_SEARCH_URL, params={"q": query, "quotesCount": limit, "newsCount": 0},
                        headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout)
    resp.raise_for_status()
    return _yahoo_matches(resp.json().get("quotes", []))


def _yahoo_matches(quotes: list[dict]) -> list[SymbolMatch]:
    out = []
    for q in quotes:
        kind = YAHOO_TYPES.get(q.get("quoteType", ""))
        symbol = q.get("symbol", "")
        if not kind or not symbol:
            continue
        if kind == "crypto":
            symbol = symbol.replace("-", "/")  # Yahoo "SOL-USD" -> "SOL/USD", the app's pair notation
        name = q.get("longname") or q.get("shortname") or ""
        out.append(SymbolMatch(symbol, name, q.get("exchDisp", ""), kind))
    return out


#: Names of common quote currencies, as charting platforms show them ("Bitcoin / TetherUS")
QUOTE_NAMES = {"USD": "U.S. Dollar", "USDT": "TetherUS", "USDC": "USD Coin", "EUR": "Euro", "GBP": "British Pound",
               "JPY": "Japanese Yen", "CHF": "Swiss Franc", "BTC": "Bitcoin", "ETH": "Ethereum", "BNB": "BNB"}


@lru_cache(maxsize=256)
def symbol_name(symbol: str, search: Callable[[str], list[SymbolMatch]] | None = None) -> str | None:
    """Full name for a chart title: ``AAPL`` -> ``Apple Inc.``, ``BTC/USDT`` -> ``Bitcoin / TetherUS``.

    None when Yahoo doesn't know it (or is unreachable); callers then keep showing the ticker. Slow: network.
    """
    search = search or yahoo_search_http
    if "/" not in symbol:
        return _exact_name(symbol, search)
    base, _, quote = symbol.partition("/")
    base_name = _crypto_name(base, search)
    if base_name is None:
        return None
    return f"{base_name} / {QUOTE_NAMES.get(quote) or _crypto_name(quote, search) or quote}"


def _exact_name(ticker: str, search) -> str | None:
    try:
        matches = search(ticker)
    except Exception as exc:  # offline: no name, not an error
        log.debug("name lookup for %s failed: %s", ticker, exc)
        return None
    return next((m.name for m in matches if m.symbol.upper() == ticker.upper() and m.name), None)


def _crypto_name(asset: str, search) -> str | None:
    """Yahoo lists coins as ``BTC-USD`` named ``Bitcoin USD``; the coin's name is the part before `` USD``."""
    name = _exact_name(f"{asset}/USD", lambda q: search(q.replace("/", "-")))
    return name.removesuffix(" USD").strip() or None if name else None


class CryptoPairs:
    """Spot pairs of the first ccxt exchange that answers, loaded once per session.

    With ``cache`` (a JSON file), the list also survives across processes for ``max_age`` seconds.
    """

    def __init__(self, exchange_ids: Iterable[str] = ("binance", "kucoin", "kraken"), cache: Path | None = None,
                 max_age: float = 86400.0):
        self.exchange_ids = tuple(exchange_ids)
        self.cache, self.max_age = cache, max_age
        self._pairs: list[SymbolMatch] | None = None
        self._lock = threading.Lock()

    def load(self) -> list[SymbolMatch]:
        with self._lock:  # one download even if several searches start at once
            if self._pairs is None:
                self._pairs = self._read_cache()
            if self._pairs is None:
                self._pairs = self._fetch()
                if self._pairs:  # don't cache an outage
                    _write_json(self.cache, [astuple(m) for m in self._pairs])
            return self._pairs

    def _read_cache(self) -> list[SymbolMatch] | None:
        if self.cache is None or not self.cache.is_file() or time.time() - self.cache.stat().st_mtime > self.max_age:
            return None
        try:
            return [SymbolMatch(*row) for row in json.loads(self.cache.read_text())]
        except (OSError, ValueError, TypeError):
            return None

    def _fetch(self) -> list[SymbolMatch]:
        import ccxt

        for exchange_id in self.exchange_ids:
            try:
                markets = getattr(ccxt, exchange_id)().load_markets()
            except Exception as exc:
                log.debug("crypto pairs from %s unavailable: %s", exchange_id, exc)
                continue
            spot = [m for m in markets.values()
                    if m.get("spot") and m.get("active", True) is not False and m.get("quote") in CRYPTO_QUOTES]
            spot.sort(key=lambda m: CRYPTO_QUOTES.index(m["quote"]))  # SOL/USDT before SOL/BTC
            return [SymbolMatch(m["symbol"], f"{m['base']} / {m['quote']}", exchange_id, "crypto") for m in spot]
        return []


class SymbolSearch:
    def __init__(self, local: Callable[[], Iterable[str]] = list,
                 remote: Iterable[Callable[[str], list[SymbolMatch]]] | None = None, limit: int = 15):
        """``local`` returns the user's own symbols; ``remote`` sources take the query and may raise."""
        self.local_symbols = local
        # exchange pairs before Yahoo, so "sol" offers the tradable SOL/USDT ahead of Yahoo's SOL/USD
        self.remote = list(remote) if remote is not None else [_crypto_source(), yahoo_search]
        self.limit = limit

    def local(self, query: str) -> list[SymbolMatch]:
        return self._ranked(query, [SymbolMatch(s, kind="watchlist") for s in dict.fromkeys(self.local_symbols())])

    def search(self, query: str) -> list[SymbolMatch]:
        """Local matches first, then remote ones; slow (network), run it off the GUI thread."""
        found: list[SymbolMatch] = []
        for source in self.remote:
            try:
                found += source(query)
            except Exception as exc:  # offline or rate limited: local suggestions still work
                log.debug("symbol search source failed: %s", exc)
        local = self.local(query)
        seen = {m.symbol for m in local}
        remote = [m for m in self._ranked(query, found) if not (m.symbol in seen or seen.add(m.symbol))]
        return (local + remote)[: self.limit]

    def _ranked(self, query: str, matches: list[SymbolMatch]) -> list[SymbolMatch]:
        scored = [(r, i, m) for i, m in enumerate(matches) if (r := rank(query, m)) is not None]
        return [m for _, _, m in sorted(scored, key=lambda t: t[:2])]  # stable: source order breaks ties


def _crypto_source(cache: Path | None = None) -> Callable[[str], list[SymbolMatch]]:
    pairs = CryptoPairs(cache=cache)
    return lambda query: pairs.load()


def cached_queries(source: Callable[[str], list[SymbolMatch]], cache: Path, max_age: float = 7 * 86400.0,
                   keep: int = 300) -> Callable[[str], list[SymbolMatch]]:
    """Wrap a search source with a JSON cache of recent queries (failures aren't cached)."""

    def search(query: str) -> list[SymbolMatch]:
        key = query.strip().lower()
        try:
            entries = json.loads(cache.read_text())
        except (OSError, ValueError):
            entries = {}
        hit = entries.get(key)
        if hit and time.time() - hit[0] <= max_age:
            return [SymbolMatch(*row) for row in hit[1]]
        found = source(query)  # may raise: SymbolSearch treats that as "source unavailable"
        entries[key] = [time.time(), [astuple(m) for m in found]]
        newest = sorted(entries.items(), key=lambda kv: kv[1][0])[-keep:]
        _write_json(cache, dict(newest))
        return found

    return search


def cli_symbol_search(local: Callable[[], Iterable[str]], cache_dir: Path) -> SymbolSearch:
    """The search for shell completion: everything that can come from disk does."""
    return SymbolSearch(local, remote=[_crypto_source(cache_dir / "crypto_pairs.json"),
                                       cached_queries(yahoo_search_http, cache_dir / "symbol_search.json")])


def _write_json(path: Path | None, data) -> None:
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data))
        tmp.replace(path)
    except OSError as exc:
        log.debug("could not write cache %s: %s", path, exc)
