# Alerts

Alerts tell you when something happens, through your [notification channels](live-trading.md#notifications) (event
`alert`; scheduled scans use `scanner`). They are stored in `alerts.json` in the config folder.

## Kinds

* **Price alert**: a symbol reaches a price (`>=`, `<=`, `>`, `<`). The price is the last close of the alert's timeframe
  (default 1m), delayed by the data provider.
* **Indicator alert**: an indicator's values or alert outputs on a symbol match (all conditions), like the
  [Scanner's](scanner.md) filters.
* **Scheduled scan**: a scan of a watchlist, index or symbol list that reports only the matches that are **new** since its
  last run.

Each alert has `--every N` minutes, or `--every close` (once per finished candle of its timeframe). A price or indicator
alert fires when its condition turns true, then switches off; with `--repeat` it waits until the condition was false
again and re-arms.

## Command line

```sh
tradebot alert price BTC/USDT ">=100000"                            # also <=, >, <; --every 5 minutes, --repeat
tradebot alert indicator AAPL -i rsi --where "value < 30" --tf 1d   # or --alert NAME conditions
tradebot alert scan -i ma_cross --alert cross_up --index sp500      # scheduled scan, reports only new matches
tradebot alert list
tradebot alert rm <id>
tradebot watch                      # headless: run the alerts and scans until Ctrl-C (--once for cron, --poll SECONDS)
```

## In the app

Right-click a level on the [Chart](chart.md) and choose *Alert me at ...*; the sidebar's **Alerts** button lists,
disables and removes alerts. The app checks them every 30 s while it is open. Without the app, run `tradebot watch`.
**Run only one of the two, or alerts fire twice.**
