# News

The **📰 News** page lists headlines (Yahoo Finance) for the symbols of all your watchlists and your portfolio, newest
first.

* The filter field narrows the list by text.
* **Refresh** reloads; **auto-refresh every 5 min** is on by default.
* Double-click a headline to open it in your browser.

The Chart's sidebar has the same headlines for the one symbol on the chart (see [Chart](chart.md#sidebar-watchlists-and-news)).

```sh
tradebot news AAPL NVDA BTC/USDT
tradebot news --watchlist Crypto   # without symbols: one watchlist (default: all of them)
tradebot news --limit 10
```
