# Extending

Built-ins and plugins register themselves when their class is defined, so a plugin is just a `.py` file that subclasses
a base class. Everything a plugin registers shows up in `tradebot list`, the CLI, the desktop app and the optimizer.

## Where plugins live

* `~/.config/tradebot/plugins/*.py`, or the folders in `$TRADEBOT_PLUGIN_DIR` (separated like `PATH`; it replaces the
  default folder). Files starting with `_` are ignored.
* Installed packages that publish an entry point in the `tradebot.plugins` group. The standalone binary can't see
  those: use the plugin folder there, or install tradebot from source.

A plugin that fails to import is logged and skipped. A plugin can import any `tradebot` module and the bundled
libraries. Plugins are **trusted local code**: they run with the app's permissions and nothing is sandboxed.

Examples in the repository:

| File | Shows |
|---|---|
| [`examples/plugins/donchian_breakout.py`](../examples/plugins/donchian_breakout.py) | one indicator and one algorithm |
| [`examples/plugins/near_high.py`](../examples/plugins/near_high.py) | an indicator with alerts, for the [Scanner](scanner.md) |
| [`examples/plugins/zscore_reversion.py`](../examples/plugins/zscore_reversion.py) | an indicator wrapped into a vectorized algorithm: [Indicators as algorithms](indicators-as-algorithms.md) |

## Algorithm

```python
from tradebot.algorithms.base import BaseAlgorithm
from tradebot.core import ops
from tradebot.core.registry import IntParam, Param
from tradebot.indicators.builtin import RSI

class RsiDip(BaseAlgorithm):
    name = "rsi_dip"
    params = {"period": IntParam(14, 2, 50), "level": Param(25, 5, 45)}

    def signals(self, o, p):
        rsi = RSI.compute(o, period=p["period"])["value"]
        return ops.latch(rsi < p["level"], rsi > 50)
```

`signals` returns the target position per candle (`1` long, `0` flat, `-1` short with `allow_short = True`). If you only
use `tradebot.core.ops` and arithmetic, the same code runs in backtests, the live bot and the batched GPU optimizer; the
optimizer passes float parameters as `[population, 1]` arrays, and integer parameters are always plain `int`.

Class attributes (all optional but `name`):

| Attribute | Meaning |
|---|---|
| `description` | one line, shown in `tradebot list` and tooltips |
| `params` | `{name: Param(default, low, high, description)}`; `IntParam` for whole numbers. The range is what the optimizer searches |
| `allow_short` | positions may go to -1 |
| `vectorized` | `False` for loop-based algorithms: the optimizer evaluates them one by one on the CPU |
| `period_batched` | `True` when integer window lengths may be `[population, 1]` arrays |
| `poolable` | `False` keeps it out of the optimizer's automatic pool |
| `chart_indicators()` | method returning indicator instances the chart should draw with this strategy |

`BaseAlgorithm.__doc__` (in `src/tradebot/algorithms/base.py`) is the reference. The full story, including how to wrap an
indicator, avoid look-ahead and stay vectorized: [Indicators as algorithms](indicators-as-algorithms.md).

## Indicator

Subclass `BaseIndicator`: set `name`, `params` and `outputs`, set `overlay = False` for its own pane, add `levels` for
reference lines, and implement `compute(cls, o, **params) -> dict`:

```python
from tradebot.core import ops
from tradebot.core.registry import IntParam
from tradebot.indicators.base import BaseIndicator

class Momentum(BaseIndicator):
    name = "momentum"
    params = {"period": IntParam(10, 2, 100)}
    overlay = False

    @classmethod
    def compute(cls, o, period):
        return {"value": o.close - ops.shift(o.close, period)}
```

`o` is an `OHLC` with arrays `o.open`, `o.high`, `o.low`, `o.close`, `o.time` and `o.volume` (NaN for symbols without
volume; set `uses_volume = True` for such indicators). Every output is one value per candle and may only use that candle
and earlier ones.

| Attribute | Meaning |
|---|---|
| `outputs` | names of the arrays `compute` returns (default `("value",)`) |
| `overlay` | `True`: drawn on the price pane; `False`: in its own pane |
| `levels` | horizontal reference lines for non-overlay panes (RSI 30/70) |
| `styles` | `{output: Style(color, width, dash, bars)}`, how each output is drawn |
| `level_styles` | colours of individual `levels` |
| `fills` | shaded areas between two outputs: `[(a, b, Style(color="rgba(...)"))]` |
| `labels`, `label_anchor` | outputs drawn as text labels instead of lines |
| `hidden` | outputs computed (for algorithms and screeners) but not drawn |
| `alerts` | `{output: description}` of 0/1 conditions; hidden from the chart, used by the Scanner (first two become filter templates). Build them with `ops.flag(cond, *inputs)` |
| `units` | what each output measures, for the evolver: `"price"`, `"osc"`, `"ratio"`, `"dist"`, `"volume"`, `"money"` or your own name |
| `uses_volume` | outputs are NaN on symbols without volume data |
| `screenable` | `False` keeps it out of the Scanner (chart patterns whose labels change with new candles) |

**Styles.** `styles = {"fast": Style(color="#ffd166", width=2), "slow": Style(color="up", dash="dash")}`. Colours are
any CSS colour (`rgba(r, g, b, a)` too) or a theme colour that follows the theme: `up`, `down`, `accent`, `muted`, `text`
and `series[N]` (the palette's N-th colour). `dash` is `"dash"` or `"dot"`. `Style(bars=True)` draws an output as bars
from 0 (merged bars add up, like volume), and a `*_hist` output draws as bars in its colour. Outputs without a colour take
the theme palette and keep it when other indicators are added or removed. The user can override colours per indicator
from the chart's **Indicators** menu.

**Evolver units.** The strategy [evolver](evolve.md) compares only like with like, so it needs to know what an output
measures. Unset outputs count as prices on overlays and as oscillators when all `levels` lie within 0 to 100; otherwise as
the indicator's own unit.

The docstring at the top of `src/tradebot/indicators/base.py` is the reference.

## The script editor

The [Scanner](scanner.md#writing-your-own-indicator) has an in-app Python editor with live reload for indicator scripts
(`near_high.py` is the example). A script is an ordinary indicator plugin file in the first plugin folder, so it also loads
at the next start, in the CLI and on every other page.

## Brokers

For any ccxt exchange a broker is two lines:

```python
from tradebot.brokers.crypto import CcxtBroker

class Bitstamp(CcxtBroker):
    name = "bitstamp"
    exchange_id = "bitstamp"
```

For venues without an API, subclass `ManualExecutionBroker` or `HoldingsFileBroker` (holdings from a CSV; orders become
order intents for you to execute by hand). Every broker exposes `get_ticker`, `get_klines`, `place_order`, `cancel_order`
and `get_balance`. See [Brokers and live trading](live-trading.md#brokers).

## Notification channels

```python
from tradebot import notify

@notify.channel("sms", ("url",))      # the name and the options the channel needs
def _sms(options, kind, title, text):
    ...
```

Code raises an event with `notify.notify("risk", title, text)`. Channels are configured as `[notify.<name>]` tables, see
[Brokers and live trading](live-trading.md#notifications).

## Rules for contributions

Don't add third-party tools or broker assets to the repo or the binaries. Integrations call a tool the user installed
(found on `PATH` or via a configured path) and fail with a clear message and a download link if it's missing. Add a row
for the tool to the table in [Getting started](getting-started.md#install). Broker names appear only as plain text, never
as logos.
