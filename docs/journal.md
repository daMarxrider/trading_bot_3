# Journal

The **📒 Journal** page and `tradebot journal` keep a record of every order the bots place or simulate.

Every order `bot run` and `bot pair-run` place or simulate goes into a local SQLite database, `journal.sqlite` in the
app's config folder (`$TRADEBOT_JOURNAL` overrides the path; `--no-journal` switches it off). A fill stores broker, mode
(`live`, `paper`, or `dry` for the intended fills of a dry run), symbol, side, amount, price, fee (as the exchange or the
paper account reports it), a note and tags.

## The page

It shows the fills with the realised P&L, fees, win rate and the unrealised P&L of what is still open (at the latest
price). Select a trade to edit its note and tags. Views and exports default to live and paper trades; dry runs are a
separate filter.

## Realised P&L and tax lots

Realised P&L matches fills first-in-first-out per symbol. Every closed piece is a **tax lot** with open and close date,
cost, proceeds, fees and profit (fees count in full, split by amount); a sell without holdings opens a short lot (futures
legs). The lots are a bookkeeping aid, not tax advice: check the rules of your country.

## Command line

```sh
tradebot journal show               # realised P&L per year, fees, latest fills (--symbol, --tag, --limit)
tradebot journal note 12 --note "news-driven" --tags swing,eth   # annotate fill #12
tradebot journal export lots_2026.csv --lots --year 2026         # FIFO tax lots (or without --lots: all fills)
```

`journal export --lots` writes the lots with their closing year and a short/long-term flag (held over 365 days); `--year`
keeps one year. Without `--lots` it writes every fill with notes, tags and realised P&L. See the
[CLI reference](cli-reference.md#tradebot-journal-show).
