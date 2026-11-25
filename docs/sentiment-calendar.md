# News sentiment and the economic calendar

## News sentiment

Every headline gets a score from -1 (very negative) to +1 (very positive), shown with a triangle so colour is never
the only cue: ▲ positive, ▼ negative, – neutral (within ±0.15). The chart's news page adds a line with the mean
score and how many headlines are positive, neutral and negative.

The score comes from a word list for financial headlines (surge, plunge, probe, record, ...), with negations
("profits do **not** rise") and intensifiers ("fall **sharply**"). It needs no key or download and works offline. It
is a rough guide to the tone of the news, not a forecast: sarcasm, context and headlines with no known word (score 0)
are beyond it.

```sh
tradebot news AAPL BTC/USDT          # a score per headline and a summary per symbol
tradebot sentiment BTC/USDT --days 7 # the mean score per day
```

### As an indicator

`news_sentiment` (parameter `smooth`: candles averaged) is the mean headline score per hour of the symbol, lined up
with the candles like the other [market data](market-data.md): a candle sees the hours that were over when it
opened, nothing newer, and a value older than three days is NaN. It works for **every** symbol, in a chart pane, in
combo rules (`news_sentiment < -0.3 & rsi < 30`) and in backtests.

News feeds only serve their latest headlines, so there is no deep history to download. The indicator's history is
what has been **collected**: each time the chart, a backtest or `tradebot market`-style fetch asks for it, the hours
seen are added to the cache (`market/news_sentiment/<symbol>.npz` in the cache folder). A backtest before that has
NaN, so its rules never fire. For the same reason `news_sentiment` is not part of the default columns, not in the
Scanner and not offered to [evolve](evolve.md) unless named or with `--market-data`.

## Economic calendar

CPI, rate decisions, jobs data and other releases with their impact (low, medium, high), forecast and previous
value, all times in UTC.

```sh
tradebot calendar                       # the next 14 days, high impact
tradebot calendar --days 30 --impact medium --country USD
```

| Source | Covers |
|---|---|
| Free weekly feed of Forex Factory (`nfs.faireconomy.media`), no key | the current week, every country |
| The Fed's published 2026 FOMC decision days (marked `~`) | rate decisions further ahead |
| The US jobs report rule (marked `~`): the third Friday after the week that contains the 12th | jobs data, any year |

Every fetch is merged into `economic_calendar.json` in the cache folder (the feed is asked at most once an hour), so
the calendar grows while you use the app. A feed entry replaces the scheduled one of the same kind on the same day.
`~` dates are rules, not announcements: a shutdown or holiday can move them. CPI dates have no rule and appear when
the feed lists them.

### On the chart

With the **Events** button on, high-impact events are marked above the candles: **R** rate decision, **C** CPI,
**J** jobs data, **!** anything else. Hover a mark for the event, forecast and previous value. The marks show for
every symbol, and for the weeks the calendar knows.

### Staying flat around events

An optional rule for backtests: **close open positions before, and open none until after, a high-impact event.**

```sh
tradebot backtest BTC/USDT sma_cross --tf 1h --flat-around-events 2
tradebot backtest SPY sma_cross --flat-around-events 24 --event-impact medium
```

Every candle that touches the window from N hours before to N hours after an event is forced flat, so a position
that was open is closed at the start of the window and the strategy may enter again after it. Off by default; to use
it in every backtest, set it once in `config.toml`:

```toml
[events]
flat_around_hours = 2
flat_impact = "high"      # low, medium or high
```

The option beats the setting (`--flat-around-events 0` turns it off for one run). With no known event in the tested
period the backtest runs as usual, with a warning. It cannot be combined with intrabar fills (`fills` `intrabar`),
and the desktop app's backtest page does not apply it yet. From code:
`backtest(df, algo, flat=economic.event_flat(df, hours))`.
