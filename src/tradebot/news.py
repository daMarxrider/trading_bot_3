# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Financial news for watchlist / portfolio symbols (free Yahoo Finance feed via yfinance)."""

from __future__ import annotations

import logging
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


def fetch_news(symbols: list[str], limit: int = 50) -> list[NewsItem]:
    """Newest first, de-duplicated by URL/title across symbols."""
    import yfinance as yf

    def one(sym):
        try:
            return [n for n in (parse_yf_item(sym, it) for it in yf.Ticker(yf_symbol(sym)).news or []) if n]
        except Exception as exc:
            log.warning("news failed for %s: %s", sym, exc)
            return []

    with ThreadPoolExecutor(8) as pool:
        items = [n for batch in pool.map(one, symbols) for n in batch]
    seen, out = set(), []
    for n in sorted(items, key=lambda n: n.published, reverse=True):
        key = n.url or n.title
        if key not in seen:
            seen.add(key)
            out.append(n)
    return out[:limit]
