# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Financial news for watchlist / portfolio symbols from several free sources (#193).

Sources are RSS/Atom feeds (no API keys) plus the Yahoo Finance feed of yfinance. A *symbol feed* is asked once
per symbol (``{q}`` in its URL); a *general feed* is fetched once and its items are matched to the symbols by
ticker or coin name. Every source is fetched on its own: one that is down is reported, the others still count.
"""

from __future__ import annotations

import html
import logging
import re
import time
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class NewsItem:
    symbol: str
    title: str
    publisher: str
    url: str
    published: datetime
    summary: str = ""


def parse_yf_item(symbol: str, item: dict) -> NewsItem | None:
    """Handles both the legacy flat format and the newer ``{"content": {...}}`` format."""
    content = item.get("content") or item
    title = content.get("title")
    if not title:
        return None
    if "providerPublishTime" in content:
        published = datetime.fromtimestamp(content["providerPublishTime"], timezone.utc)
    else:
        raw = content.get("pubDate") or content.get("displayTime")
        published = datetime.fromisoformat(raw.replace("Z", "+00:00")) if raw else datetime.now(timezone.utc)
    url = content.get("link") or (content.get("canonicalUrl") or {}).get("url") or (content.get("clickThroughUrl") or {}).get("url", "")
    publisher = content.get("publisher") or (content.get("provider") or {}).get("displayName", "")
    return NewsItem(symbol, title, publisher, url, published, content.get("summary", ""))


def yf_symbol(symbol: str) -> str:
    from tradebot.data.providers import YFinanceProvider

    return YFinanceProvider.map_symbol(symbol)


@dataclass(frozen=True)
class Source:
    id: str
    name: str
    scope: str  # "stocks", "crypto" or "any": which symbols it is asked about / matched to
    url: str = ""  # "" for the yfinance source; ``{q}`` is replaced by the query of a symbol feed
    per_symbol: bool = False
    default: bool = True  # on unless the user turns it off (#194)


_GN = "https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
SOURCES: dict[str, Source] = {s.id: s for s in (
    Source("yfinance", "Yahoo Finance (yfinance)", "any", per_symbol=True),
    Source("yahoo_rss", "Yahoo Finance RSS", "any", "https://feeds.finance.yahoo.com/rss/2.0/headline?s={q}&region=US&lang=en-US", True),
    Source("google_stocks", "Google News (stocks)", "stocks", _GN, True),
    Source("google_crypto", "Google News (crypto)", "crypto", _GN, True),
    Source("cnbc", "CNBC Markets", "stocks", "https://www.cnbc.com/id/100003114/device/rss/rss.html"),
    Source("marketwatch", "MarketWatch", "stocks", "https://feeds.content.dowjones.io/public/rss/mw_topstories"),
    Source("coindesk", "CoinDesk", "crypto", "https://www.coindesk.com/arc/outboundfeeds/rss/"),
    Source("cointelegraph", "Cointelegraph", "crypto", "https://cointelegraph.com/rss"),
    Source("decrypt", "Decrypt", "crypto", "https://decrypt.co/feed"),
    Source("theblock", "The Block", "crypto", "https://www.theblock.co/rss.xml"),
    Source("bitcoinmagazine", "Bitcoin Magazine", "crypto", "https://bitcoinmagazine.com/feed", default=False),
)}

CRYPTO_NAMES = {"BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana", "XRP": "ripple|xrp", "ADA": "cardano",
                "DOGE": "dogecoin", "BNB": "binance", "LTC": "litecoin", "DOT": "polkadot", "AVAX": "avalanche",
                "LINK": "chainlink", "MATIC": "polygon", "TRX": "tron", "XMR": "monero", "ATOM": "cosmos"}
USER_AGENT = "Mozilla/5.0 (compatible; tradebot-news)"
TIMEOUT = 10
MAX_BYTES = 5_000_000
CACHE_SECONDS = 60
_cache: dict[str, tuple[float, bytes]] = {}


def is_crypto(symbol: str) -> bool:
    return "/" in symbol or symbol.upper().endswith(("-USD", "-USDT", "-EUR", "-USDC"))


def base_ticker(symbol: str) -> str:
    return re.split(r"[/-]", symbol)[0].upper()


def _text(el, *names) -> str:
    for child in el:
        if child.tag.rsplit("}", 1)[-1] in names and (child.text or "").strip():
            return child.text.strip()
    return ""


def _clean(text: str, limit: int = 300) -> str:
    text = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", text))).strip()
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "…"


def _date(raw: str) -> datetime:
    try:
        d = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        try:
            d = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return datetime.now(timezone.utc)
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def parse_feed(data: bytes, publisher: str) -> list[NewsItem]:
    """The entries of an RSS 2.0 or Atom document, with ``symbol`` left empty."""
    if re.search(rb"<!ENTITY", data, re.I):  # entity expansion bombs; real feeds do not declare entities
        raise ValueError("feed declares XML entities")
    root = ET.fromstring(data)
    items = []
    for el in root.iter():
        if el.tag.rsplit("}", 1)[-1] not in ("item", "entry"):
            continue
        title = _clean(_text(el, "title"), 400)
        link = _text(el, "link")
        if not link:  # Atom: <link href="..."/>
            link = next((c.get("href", "") for c in el if c.tag.rsplit("}", 1)[-1] == "link"), "")
        if not title or not link:
            continue
        source = _text(el, "source") or publisher  # Google News names the real publisher here
        items.append(NewsItem("", title, source, link, _date(_text(el, "pubDate", "published", "updated", "date")),
                              _clean(_text(el, "description", "summary"))))
    return items


def _download(url: str) -> bytes:
    import requests

    hit = _cache.get(url)
    if hit and time.time() - hit[0] < CACHE_SECONDS:
        return hit[1]
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT, stream=True)
    r.raise_for_status()
    data = r.raw.read(MAX_BYTES + 1, decode_content=True)
    if len(data) > MAX_BYTES:
        raise ValueError("feed too large")
    _cache[url] = (time.time(), data)
    return data


def _query(source: Source, symbol: str) -> str:
    from urllib.parse import quote_plus

    if source.id == "google_stocks":
        return quote_plus(f'"{symbol}" stock')
    if source.id == "google_crypto":
        return quote_plus(f"{base_ticker(symbol)} {CRYPTO_NAMES.get(base_ticker(symbol), 'crypto').split('|')[0]} crypto")
    return quote_plus(yf_symbol(symbol))


def match_symbols(item: NewsItem, symbols: list[str]) -> str:
    """The first of ``symbols`` that the headline or summary talks about (ticker or coin name), or ''."""
    text = f"{item.title} {item.summary}"
    for sym in symbols:
        base = base_ticker(sym)
        pattern = rf"(?<![A-Za-z0-9])\$?{re.escape(base)}(?![A-Za-z0-9])"
        if is_crypto(sym) and base in CRYPTO_NAMES:
            pattern += f"|{CRYPTO_NAMES[base]}"
        if re.search(pattern, text, re.I if is_crypto(sym) else 0):
            return sym
    return ""


def default_sources() -> list[str]:
    return [s.id for s in SOURCES.values() if s.default]


def _yfinance(sym: str) -> list[NewsItem]:
    import yfinance as yf

    # Ticker.news is sometimes empty server-side; the search endpoint is a reliable fallback
    raw = yf.Ticker(yf_symbol(sym)).news or yf.Search(yf_symbol(sym), news_count=20).news or []
    return [n for n in (parse_yf_item(sym, it) for it in raw) if n]


def _title_key(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", re.sub(r" [-–|] [^-–|]{2,40}$", "", title.lower()))


def fetch_news_report(symbols: list[str], limit: int = 50, sources: list[str] | None = None
                      ) -> tuple[list[NewsItem], dict[str, str]]:
    """Newest first, de-duplicated by URL and headline across symbols and sources; plus ``{source id: error}``
    for the sources that failed."""
    chosen = [SOURCES[i] for i in (default_sources() if sources is None else sources) if i in SOURCES]
    errors: dict[str, str] = {}
    jobs = []  # (source, symbol or None)
    for src in chosen:
        if src.per_symbol:
            jobs += [(src, s) for s in symbols if src.scope == "any" or (src.scope == "crypto") == is_crypto(s)]
        else:
            jobs.append((src, None))

    def run(job):
        src, sym = job
        try:
            if src.id == "yfinance":
                return job, _yfinance(sym), None
            if sym is None:
                items = parse_feed(_download(src.url), src.name)
                pool = [s for s in symbols if src.scope == "any" or (src.scope == "crypto") == is_crypto(s)]
                return job, [replace_symbol(i, match_symbols(i, pool)) for i in items], None
            items = parse_feed(_download(src.url.format(q=_query(src, sym))), src.name)
            return job, [replace_symbol(i, sym) for i in items], None
        except Exception as exc:  # one broken feed must not hide the others
            return job, [], f"{type(exc).__name__}: {exc}"

    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(run, jobs))
    items = []
    for (src, sym), batch, err in results:
        if err:
            errors.setdefault(src.id, err)
            log.warning("news source %s failed%s: %s", src.id, f" for {sym}" if sym else "", err)
        items += [i for i in batch if i.symbol]
    seen_url, seen_title, out = set(), set(), []
    for n in sorted(items, key=lambda n: n.published, reverse=True):
        url_key, title_key = n.url.split("?")[0] if "news.google" not in n.url else n.url, _title_key(n.title)
        if url_key in seen_url or title_key in seen_title:
            continue
        seen_url.add(url_key)
        seen_title.add(title_key)
        out.append(n)
    return out[:limit], errors


def replace_symbol(item: NewsItem, symbol: str) -> NewsItem:
    from dataclasses import replace

    return replace(item, symbol=symbol)


def fetch_news(symbols: list[str], limit: int = 50, sources: list[str] | None = None) -> list[NewsItem]:
    """Newest first, merged from the enabled ``sources`` (ids of :data:`SOURCES`; default: all default-on ones)."""
    return fetch_news_report(symbols, limit, sources)[0]
