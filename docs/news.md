# News

The **📰 News** page lists headlines for the symbols of all your watchlists and your portfolio, newest first, merged
from several free sources. The publisher of each headline is shown.

* The filter field narrows the list by text.
* **Refresh** reloads; **auto-refresh every 5 min** is on by default.
* Double-click a headline to open it in your browser.

## Sources

No accounts or API keys: RSS feeds plus Yahoo Finance. Headlines that several sources carry are shown once.

| Source | Covers | How it finds your symbols |
|---|---|---|
| Yahoo Finance (yfinance), Yahoo Finance RSS | stocks and crypto | asked per symbol |
| Google News (stocks), Google News (crypto) | stocks / crypto | search per symbol (ticker, coin name) |
| CNBC Markets, MarketWatch | stocks | general feed, matched by ticker in the headline or summary |
| CoinDesk, Cointelegraph, Decrypt, The Block | crypto | general feed, matched by ticker or coin name |
| Bitcoin Magazine | crypto | like the other crypto feeds, but off by default |

A source that is down is skipped (the command line prints a warning); the others still deliver. Feeds are cached for a
minute. `tradebot news --list-sources` lists the ids (and which are on), and `--source` picks some of them.

**Settings → News sources** has a checkbox per source, grouped Stocks / Crypto / General. A change applies at the next
refresh, no restart. A source that failed at its last refresh is marked there. **Add your own feed** takes a name and an
RSS or Atom address, fetches it once to check it and keeps it, switched on; a feed of your own can be removed again. It is
a general feed: its headlines are matched to your symbols by ticker or name. The choice is stored in `config.toml`:

```toml
[news]
sources = ["cnbc", "coindesk", "custom:https://example.com/feed.xml"]
custom = ["SEC filings | https://example.com/feed.xml"]
```

Without a `[news]` table the default-on sources are used. The News page can show the headlines of one publisher.

Each headline carries a sentiment score (a triangle, the score and the word); see
[News sentiment and the economic calendar](sentiment-calendar.md).

The Chart's sidebar has the same headlines for the one symbol on the chart (see [Chart](chart.md#sidebar-watchlists-and-news)).

```sh
tradebot news AAPL NVDA BTC/USDT
tradebot news --watchlist Crypto   # without symbols: one watchlist (default: all of them)
tradebot news --limit 10
```
