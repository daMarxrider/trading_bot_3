# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""News from several RSS/Atom sources (#193): parsing, matching, merging, failing feeds. No network."""

from datetime import timezone

import pytest

from tradebot import news
from tradebot.news import NewsItem, SOURCES, fetch_news_report, match_symbols, parse_feed

RSS = b"""<?xml version="1.0"?><rss version="2.0"><channel><title>x</title>
<item><title>Bitcoin jumps as ETF flows return</title><link>https://a.example/1?utm=1</link>
<pubDate>Mon, 23 Nov 2026 10:00:00 GMT</pubDate><description>&lt;p&gt;BTC rose &lt;b&gt;4%&lt;/b&gt;&lt;/p&gt;</description></item>
<item><title>Apple (AAPL) beats estimates</title><link>https://a.example/2</link>
<pubDate>Mon, 23 Nov 2026 11:00:00 GMT</pubDate><source url="https://s">Some Wire</source></item>
<item><title></title><link>https://a.example/3</link></item></channel></rss>"""

ATOM = b"""<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"><title>f</title>
<entry><title>8-K filed</title><link href="https://sec.example/8k"/><updated>2026-11-23T09:30:00Z</updated>
<summary>Current report</summary></entry></feed>"""


def test_parse_rss_and_atom():
    a, b = parse_feed(RSS, "Feed")
    assert (a.title, a.url, a.publisher, a.summary) == ("Bitcoin jumps as ETF flows return", "https://a.example/1?utm=1", "Feed", "BTC rose 4%")
    assert a.published.tzinfo is not None and a.published.astimezone(timezone.utc).hour == 10
    assert b.publisher == "Some Wire"  # the item's own <source> wins
    (e,) = parse_feed(ATOM, "SEC")
    assert (e.title, e.url, e.published.minute) == ("8-K filed", "https://sec.example/8k", 30)


def test_entity_declarations_are_refused():
    with pytest.raises(ValueError):
        parse_feed(b'<?xml version="1.0"?><!DOCTYPE r [<!ENTITY a "aaaa">]><rss><channel/></rss>', "x")


def test_matching_by_ticker_and_coin_name():
    item = lambda t: NewsItem("", t, "p", "u", news.datetime.now(timezone.utc))  # noqa: E731
    syms = ["AAPL", "BTC/USDT", "ETH/USDT"]
    assert match_symbols(item("Apple (AAPL) beats estimates"), syms) == "AAPL"
    assert match_symbols(item("Bitcoin jumps"), syms) == "BTC/USDT"
    assert match_symbols(item("Ethereum upgrade ships"), syms) == "ETH/USDT"
    assert match_symbols(item("Maple syrup prices"), syms) == ""  # AAPL is not a substring match
    assert match_symbols(item("Rail strike"), ["T"]) == ""  # one-letter ticker only as a whole word, case-sensitive for stocks


@pytest.fixture
def feeds(monkeypatch):
    """Every download answers from a dict; a missing URL raises like a dead feed."""
    served = {}
    monkeypatch.setattr(news, "_download", lambda url: served[url])
    monkeypatch.setattr(news, "_yfinance", lambda sym: [])
    return served


def test_sources_merge_dedupe_and_survive_a_broken_feed(feeds):
    feeds[SOURCES["cointelegraph"].url] = RSS
    feeds[SOURCES["decrypt"].url] = RSS  # the same stories again: dropped as duplicates
    # coindesk, theblock, ... are not served: they fail, the rest still deliver
    items, errors = fetch_news_report(["AAPL", "BTC/USDT"], 10, ["cointelegraph", "decrypt", "coindesk"])
    assert [i.symbol for i in items] == ["BTC/USDT"]  # crypto feeds are only matched against crypto symbols
    assert "coindesk" in errors and "decrypt" not in errors
    items, _ = fetch_news_report(["AAPL", "BTC/USDT"], 10, ["cointelegraph"])
    assert len(items) == 1


def test_stock_feed_matches_stock_symbols_only_and_is_sorted(feeds):
    feeds[SOURCES["cnbc"].url] = RSS
    items, errors = fetch_news_report(["AAPL", "BTC/USDT"], 10, ["cnbc"])
    assert errors == {} and [(i.symbol, i.title) for i in items] == [("AAPL", "Apple (AAPL) beats estimates")]


def test_symbol_feeds_ask_per_symbol_and_scope(feeds, monkeypatch):
    asked = []
    monkeypatch.setattr(news, "_download", lambda url: asked.append(url) or RSS)
    fetch_news_report(["AAPL", "BTC/USDT"], 10, ["google_stocks", "google_crypto"])
    assert len(asked) == 2 and "AAPL" in asked[0] and "BTC" in asked[1]


def test_limit_and_default_sources(feeds):
    assert "bitcoinmagazine" not in news.default_sources() and "cnbc" in news.default_sources()
    feeds[SOURCES["cnbc"].url] = RSS
    assert fetch_news_report(["AAPL"], 0, ["cnbc"])[0] == []


# -- the user's choice of sources (#194) --------------------------------------------------------------------
@pytest.fixture
def config(tmp_path, monkeypatch):
    path = tmp_path / "config.toml"
    monkeypatch.setenv("TRADEBOT_CONFIG", str(path))
    return path


def test_enabled_sources_default_until_the_user_chooses(config):
    assert news.enabled_sources() == news.default_sources()
    assert news.set_enabled("cnbc", False) and news.set_enabled("bitcoinmagazine", True)
    on = news.enabled_sources()
    assert "cnbc" not in on and "bitcoinmagazine" in on and "coindesk" in on  # the rest keeps its default
    assert news.set_enabled("bitcoinmagazine", True) and news.enabled_sources().count("bitcoinmagazine") == 1


def test_fetch_uses_only_the_enabled_sources(config, feeds):
    feeds[SOURCES["cnbc"].url] = RSS
    feeds[SOURCES["marketwatch"].url] = RSS
    news.set_enabled("marketwatch", False)
    on = news.enabled_sources()
    assert "cnbc" in on and "marketwatch" not in on
    items, errors = fetch_news_report(["AAPL"], 10)
    assert [i.publisher for i in items] == ["Some Wire"] and "marketwatch" not in errors  # never asked


def test_custom_feed_is_stored_enabled_and_removable(config, feeds):
    assert news.add_custom("My | feed", "https://x.example/feed.xml")
    (sid, src), = news.custom_sources().items()
    assert sid == "custom:https://x.example/feed.xml" and src.name == "My feed" and sid in news.enabled_sources()
    assert "cnbc" in news.enabled_sources()  # adding one doesn't change the other choices
    feeds[src.url] = RSS
    items, errors = fetch_news_report(["BTC/USDT"], 10, [sid])
    assert errors == {} and [(i.symbol, i.publisher) for i in items] == [("BTC/USDT", "My feed")]
    assert news.remove_custom(sid)
    assert news.custom_sources() == {} and sid not in news.enabled_sources()


def test_failed_sources_are_remembered_until_they_work(config, feeds):
    news.LAST_ERRORS.clear()
    fetch_news_report(["AAPL"], 10, ["cnbc"])  # not served: fails
    assert "cnbc" in news.LAST_ERRORS
    feeds[SOURCES["cnbc"].url] = RSS
    fetch_news_report(["AAPL"], 10, ["cnbc"])
    assert "cnbc" not in news.LAST_ERRORS


def test_validate_feed(monkeypatch):
    monkeypatch.setattr(news, "_download", lambda url: {"ok": RSS, "html": b"<html><body><p>hi</body>", "empty": b"<rss><channel/></rss>"}[url.rsplit("/")[-1]])
    assert news.validate_feed("https://x/ok") == 2
    for bad in ("https://x/html", "https://x/empty", "ftp://x/ok"):
        with pytest.raises(ValueError):
            news.validate_feed(bad)
