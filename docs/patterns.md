# Pattern detection

Three indicators find patterns in the candles. They only use the current and earlier candles: a swing point needs
`strength` candles on each side, so a pattern built on one is reported `strength` candles after it formed and never
changes later. That makes them safe in the [Scanner](scanner.md), as combo conditions and in [evolve](evolve.md)
(unlike the Elliott wave labels).

| Indicator | Finds | Alerts |
|---|---|---|
| `candles` | doji, hammer, shooting star, bullish / bearish engulfing, morning / evening star | `bullish`, `bearish`, `doji`, `hammer`, `shooting_star`, `bull_engulfing`, `bear_engulfing`, `morning_star`, `evening_star` |
| `chart_patterns` | double top / bottom, head and shoulders (and inverse), ascending / descending / symmetrical triangle, bull / bear flag | `bullish`, `bearish`, `double_top`, `double_bottom`, `head_shoulders`, `inverse_head_shoulders`, `ascending_triangle`, `descending_triangle`, `symmetrical_triangle`, `bull_flag`, `bear_flag` |
| `support_resistance` | the nearest support and resistance level from swing points, confirmed by repeated touches or a high-volume candle | `near_support`, `near_resistance`, `breakout_up`, `breakout_down` |

* **Chart**: add the indicator; patterns are marked with short labels (`Eng+`, `W`, `M`, `H&S`, `Asc.Tri`, ...), the
  levels are drawn as two lines. The doji is an alert only (it is frequent, so marking it clutters the chart).
* **Combo filter**: `candles.bull_engulfing & rsi(14) < 40`, `chart_patterns.double_bottom`,
  `support_resistance.breakout_up & close > sma(200)`; parameters in brackets, e.g. `chart_patterns(8).bullish`.
* **Scanner**: pick the indicator and filter on an alert, e.g. `chart_patterns` with `double_bottom` on.
* **Parameters**: `candles` `doji_ratio`, `trend`, `long_ratio`; `chart_patterns` `strength`, `tolerance`, `depth`;
  `support_resistance` `strength`, `zone`, `min_touches`, `lookback`, `proximity`. Without volume data the levels use
  touches only.

Patterns are heuristics on price shape, not predictions; check any rule in a backtest.
