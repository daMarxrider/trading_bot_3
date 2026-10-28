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
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
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

    out = []
    for q in yf.Search(query, max_results=limit, news_count=0).quotes:
        kind = YAHOO_TYPES.get(q.get("quoteType", ""))
        symbol = q.get("symbol", "")
        if not kind or not symbol:
            continue
        if kind == "crypto":
            symbol = symbol.replace("-", "/")  # Yahoo "SOL-USD" -> "SOL/USD", the app's pair notation
        name = q.get("longname") or q.get("shortname") or ""
        out.append(SymbolMatch(symbol, name, q.get("exchDisp", ""), kind))
    return out


class CryptoPairs:
    """Spot pairs of the first ccxt exchange that answers, loaded once per session."""

    def __init__(self, exchange_ids: Iterable[str] = ("binance", "kucoin", "kraken")):
        self.exchange_ids = tuple(exchange_ids)
        self._pairs: list[SymbolMatch] | None = None
        self._lock = threading.Lock()

    def load(self) -> list[SymbolMatch]:
        with self._lock:  # one download even if several searches start at once
            if self._pairs is None:
                self._pairs = self._fetch()
            return self._pairs

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


def _crypto_source() -> Callable[[str], list[SymbolMatch]]:
    pairs = CryptoPairs()
    return lambda query: pairs.load()
