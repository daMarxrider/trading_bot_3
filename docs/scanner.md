# Scanner and scripts

The **📡 Scanner** page (and `tradebot scan`) runs **one indicator** over a whole list of symbols and keeps
the ones that match your filters. It's separate from the **Screener** page, which stays as it is.

## Using the page

1. **Source** (top left): the pill shows what gets scanned, for example `⭐ Main · 24 symbols`. Click it to pick
   a **watchlist** or a **market index** (S&P 500, Nasdaq 100, Dow, DAX 40, FTSE 100, top-100 crypto). Exactly
   one source is active. Your choice is remembered. If it disappears (a deleted watchlist), the Scanner
   falls back to the first watchlist and says so in the status bar.
2. **Indicator** (next to it): `ƒ MA Cross (9, 21, 0)` is the default. Click it to pick another one. The picker
   groups them as *My scripts*, *Built-in* and *Community*, with a search field. Indicators that can't be
   screened (Elliott waves, anything with no plot and no alert) are hidden; the footer counts them.
   **⚙** edits the numeric inputs, remembered per indicator.
3. **Timeframe**: `1m 5m 15m 30m 1h 2h 4h 1D 1W 1M`. The last 500 candles of each symbol are used.
4. **Filters** (chips under the toolbar, all must match):
   * the indicator's first two alerts appear as **off** chips (`◻ Fast MA crosses above slow MA`); click to turn on
   * **＋ Filter** adds a *plot value* filter (`fast > slow`, `rsi < 30`) or an *alert* filter (is true / is false)
   * click a chip to edit it, ✕ removes it. Filters are remembered per indicator.
5. **Scan**. A bar shows what's running, with **Stop**; after 2 seconds a progress bar joins it. The summary reads
   `503 scanned · 12 match · 3 skipped · 4.1 s · 1D`; click "3 skipped" to see why (usually no data).
6. **Results**: the columns are the indicator's own outputs (the first 10 plots). **☰** and **＋ Column** show or
   hide any output, alerts included (`✓` / `–`). Click a header to sort (empty values last). **Double-click
   a row** to open it in the Chart with the scan's timeframe and the indicator added.
7. Change the source, indicator, inputs, timeframe or filters and the old results grey out with
   **Settings changed. Rescan**. Undo the change and the notice goes away.

Only one scan runs at a time; a second one is refused with "Another scan is running".

## Command line

```bash
tradebot scan --watchlist Main                                   # MA Cross (9/21) on the daily candles
tradebot scan AAPL MSFT NVDA -i rsi -p period=7 -t 4h --where "value < 30"
tradebot scan -i ma_cross -p fast=5 -p slow=20 --alert cross_up --alert '!cross_down'
```

`--where "<output> <op> <number|output>"` (`> >= < <= == !=`), `--alert NAME` or `--alert '!NAME'`.

## Writing your own indicator

A script is an ordinary indicator plugin: a `BaseIndicator` subclass in a `.py` file in
`~/.config/tradebot/plugins/` (or `$TRADEBOT_PLUGIN_DIR`). Scripts load at the next start, in the CLI and on the
other pages as well. The full reference is the docstring at the top of `src/tradebot/indicators/base.py`.

Open the editor from the indicator picker:

* **New script …** starts from a small working template
* **Edit** (on a My scripts row) opens that file
* **Copy to my scripts** (on any other row) opens a copy of a built-in under a new name

The editor opens next to the table, so you can edit and scan side by side. It has Python colours, line numbers,
Tab = 4 spaces, Enter keeps the indentation (and adds a level after `:`), and **Ctrl+/** comments lines.
No autocomplete, linting or debugger.

**Ctrl+S** (or *Save*) writes the file and reloads it:

1. the source is compiled; on a `SyntaxError` the line is marked and the old version stays active
2. the module is imported again; the new class replaces the old one with the same `name`
3. the indicator must be screenable and survive a dry run on up to 500 candles (your first source symbol's
   cached candles if there are any); errors show the line, and the old version is restored
4. on success the bar reads `Loaded near_high: 2 plots, 2 alerts`. If it's the Scanner's current indicator,
   its columns and filters update, the results are marked stale, and charts showing it redraw

**Auto-reload** (a checkbox, off by default) reloads one second after you stop typing, *without* writing the
file; `●` in the title means unsaved changes. The editor asks before you close it or open another script with unsaved
changes. Renaming `name` registers a new indicator; the old one stays until the next start.

### What a script returns

```python
class NearHigh(BaseIndicator):
    name = "near_high"                       # picker, CLI and chart use this
    params = {"period": IntParam(20, 2, 200)}   # numeric inputs (the settings dialog shows only these)
    outputs = ("high_n", "pct_below", "breakout", "near")
    alerts = {"breakout": "...", "near": "..."}  # outputs that are true/false conditions
    units = {"pct_below": "ratio"}           # how the table formats a column ("ratio" shows as %)

    @classmethod
    def compute(cls, o, period):             # o.open / o.high / o.low / o.close are arrays
        ...
        return {"high_n": ..., "breakout": ops.flag(condition, *inputs_that_may_be_nan), ...}
```

* Every output is one value per candle. Use only this candle and earlier ones (no look-ahead).
* **Alerts** are outputs holding 0/1 (`ops.flag(condition, *inputs)` makes one and gives NaN while the inputs
  are still warming up). They're hidden on the chart. The Scanner uses the first two as filter chips.
* Helpers live in `tradebot.core.ops`: `sma`, `ema`, `shift`, `rolling_max`, `rolling_min`, `cross_above`, …
* Scripts receive only their own symbol's candles; there is no data-request API.

A complete example is `examples/plugins/near_high.py`. Copy it into the editor and try:

1. save it, pick **Near High** in the picker, scan, and turn on the `Close within 2% of the N-bar high` chip
2. break it on purpose (`retrun {`) and watch the marked line while the old version keeps working
3. change `0.02` to `0.10`, save, and Rescan

> **Scripts are trusted local code.** They run with the app's permissions, exactly like any plugin; nothing is
> sandboxed, and a script with an endless loop freezes the app.
