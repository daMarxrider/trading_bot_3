# Fundamentals and corporate events

Valuation and quality figures of a stock, and the dates of its earnings, splits and dividends. They come free from
Yahoo Finance (through yfinance); there is nothing to configure. Crypto pairs, pair symbols (`KO:PEP`) and indices
have none: those show a short note instead, and a stock Yahoo has no data for does too. A failed request never stops
anything else.

One request per symbol is cached for a day in the [cache folder](data.md#the-cache) (`fundamentals/<symbol>.json`).
When Yahoo can't be reached the last cached answer is used, however old. Yahoo's figures can be incomplete or
late; they are information, not advice.

## Figures

| Figure | Meaning |
|---|---|
| P/E, Forward P/E | price over trailing / expected earnings; none for a loss-making company |
| P/S | price over trailing 12-month sales |
| Profit margin, Operating margin | share of sales left as profit / operating profit |
| Revenue growth, Earnings growth | year-over-year change, shown with a triangle (up or down) as well as a colour |
| Dividend yield | yearly dividend per share over the price |
| Market cap | from Yahoo |

## Chart: info page and event markers

The chart's sidebar has a **Fundamentals** page (the circled *i*) with these figures and a list of the upcoming and
the last ten earnings, split and ex-dividend dates. It fetches in the background only while it is open.

The **Events** button in the chart toolbar marks those dates under the candles: **E** earnings, **S** split,
**D** ex-dividend. The letter is always drawn, so the kind never depends on colour. Where several fall on one candle
the earnings mark wins; the page lists all of them. Off by default, remembered. It does nothing on crypto.

## Screener

[Screener](screener.md) columns `pe`, `forward_pe`, `ps`, `profit_margin`, `operating_margin`, `revenue_growth`,
`earnings_growth`, `dividend_yield` (fractions: 0.03 = 3 %) and `days_to_earnings` (to the next known earnings date)
are added as soon as a filter uses one, e.g. `pe < 15; profit_margin > 0.10`. A symbol without the figure fails that
filter. Presets: cheap and profitable, dividend payers, earnings within 14 days.

## Command line

```sh
tradebot fundamentals AAPL MSFT          # figures and the next earnings date, then the events
tradebot fundamentals -w Main --no-events --refresh
tradebot screen -f "pe < 15" -f "dividend_yield > 0.03"
tradebot screen --fundamentals           # show the columns without filtering on them
```

## No new position before earnings

In the app, the Backtest lab has a **No entry before earnings** field (days, off by default) that applies to Run.
The Scanner's column menu (☰) has a **Fundamentals (stocks)** submenu that adds the same figures as columns to the
scan results; they are fetched in the background after a scan.

An optional rule for backtests: **open no new position in the N days before a company's earnings date** (the earnings
day included). It only blocks entries: a position opened earlier is kept through the announcement, and the first
candle after the earnings day may open again. A strategy that flips from long to short counts as a new entry.

```sh
tradebot backtest AAPL sma_cross --no-entry-before-earnings 3
```

Off by default. To use it for every backtest on the command line, set it once in `config.toml`:

```toml
[events]
earnings_blackout_days = 3
```

The option beats the setting (`--no-entry-before-earnings 0` turns it off for one run). Stocks only; a symbol with no
known earnings dates is backtested as usual, with a warning. It cannot be combined with intrabar fills (`fills`
`intrabar`). The desktop app's backtest page does not apply it yet.

Earnings dates are as reliable as Yahoo's calendar: past dates are usually exact, future ones may be estimates.
From code: `tradebot.fundamentals.load(symbol)` and `backtest(..., blackout=blackout_mask(df.index, dates, days))`.
