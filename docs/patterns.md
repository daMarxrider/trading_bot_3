# Pattern detection

Four indicators find patterns in the candles. They only use the current and earlier candles: a swing point needs
`strength` candles on each side, so a pattern built on one is reported `strength` candles after it formed and never
changes later. That makes them safe in the [Scanner](scanner.md), as combo conditions and in [evolve](evolve.md)
(unlike the Elliott wave labels).

| Indicator | Finds | Alerts |
|---|---|---|
| `candles` | doji, hammer, shooting star, bullish / bearish engulfing, morning / evening star, three white soldiers / three black crows, bullish / bearish harami, piercing line / dark cloud, tweezer bottom / top | `bullish`, `bearish`, `doji`, `hammer`, `shooting_star`, `bull_engulfing`, `bear_engulfing`, `morning_star`, `evening_star`, `three_white_soldiers`, `three_black_crows`, `bull_harami`, `bear_harami`, `piercing`, `dark_cloud`, `tweezer_bottom`, `tweezer_top` |
| `chart_patterns` | double top / bottom, head and shoulders (and inverse), ascending / descending / symmetrical triangle, rising / falling wedge, bull / bear flag, bull / bear pennant, cup and handle | `bullish`, `bearish`, `double_top`, `double_bottom`, `head_shoulders`, `inverse_head_shoulders`, `ascending_triangle`, `descending_triangle`, `symmetrical_triangle`, `rising_wedge`, `falling_wedge`, `bull_flag`, `bear_flag`, `bull_pennant`, `bear_pennant`, `cup_handle` |
| `support_resistance` | the nearest support and resistance level from swing points, confirmed by repeated touches or a high-volume candle | `near_support`, `near_resistance`, `breakout_up`, `breakout_down` |
| `volume_levels` | the nearest support and resistance at the high-volume nodes of the volume profile of the last `lookback` candles, and the point of control (POC) | `near_support`, `near_resistance`, `breakout_up`, `breakout_down` |

* **Chart**: add the indicator. Patterns are short labels that pair the colour with a triangle: green `▲` under the
  candle for bullish, red `▼` over it for bearish, a grey `◆` for the symmetrical triangle (no direction). Hover a
  label for its full name. Labels of one indicator closer than eight candles on the same side are thinned out so the
  chart stays readable; every pattern is still an alert. The doji is an alert only (it is frequent).
* **Levels** (`support_resistance`, `volume_levels`) are drawn as horizontal bands, one per stretch where the level
  stayed the same, with a thin line at the level and an `S` (green, support) or `R` (red, resistance) at the latest
  one (`POC` in grey for the point of control). The band is `zone` of the price high. They are not lines that slant
  from level to level, because a level does not move gradually: it is replaced.
* **Combo filter**: `candles.bull_engulfing & rsi(14) < 40`, `chart_patterns.double_bottom`,
  `support_resistance.breakout_up & close > sma(200)`; parameters in brackets, e.g. `chart_patterns(8).bullish`.
* **Scanner**: pick the indicator and filter on an alert, e.g. `chart_patterns` with `double_bottom` on.
* **Parameters**: `candles` `doji_ratio`, `trend`, `long_ratio`; `chart_patterns` `strength`, `tolerance`, `depth`;
  `support_resistance` `strength`, `zone`, `min_touches`, `lookback`, `proximity`; `volume_levels` `lookback`, `bins`,
  `node_mult`, `zone`. Without volume data `support_resistance` uses touches only and `volume_levels` is empty.

Patterns are heuristics on price shape, not predictions; check any rule in a backtest.

## Labels

| Label | Pattern |
|---|---|
| `▲ Ham` / `▼ Star` | hammer / shooting star |
| `▲ Eng` / `▼ Eng` | bullish / bearish engulfing |
| `▲ MStar` / `▼ EStar` | morning / evening star |
| `▲ 3WS` / `▼ 3BC` | three white soldiers / three black crows |
| `▲ Har` / `▼ Har` | bullish / bearish harami |
| `▲ Pierce` / `▼ Cloud` | piercing line / dark cloud cover |
| `▲ Twz` / `▼ Twz` | tweezer bottom / top |
| `▲ W` / `▼ M` | double bottom / double top |
| `▲ iH&S` / `▼ H&S` | inverse head and shoulders / head and shoulders |
| `▲ Asc` / `▼ Desc` / `◆ Sym` | ascending / descending / symmetrical triangle |
| `▲ Wedge` / `▼ Wedge` | falling / rising wedge |
| `▲ Flag` / `▼ Flag` | bull / bear flag |
| `▲ Pennant` / `▼ Pennant` | bull / bear pennant (a flag whose range narrows) |
| `▲ Cup` | cup and handle |

## How the new patterns are found

* **Candles**: three soldiers / crows are three long candles in a row, each opening inside the previous body and
  closing beyond it. A harami is a small body inside the previous long, opposite coloured body after a fall / rise.
  Piercing / dark cloud open beyond the previous candle's extreme and close past the middle of its body but not past
  its open. Tweezers are two opposite coloured candles whose lows / highs match within a tenth of the range.
* **Wedges** are triangles whose two edges slope the same way and converge. Triangles, wedges and cups are marked
  once per formation, not again at every further swing point.
* **Pennants** are flags whose second half of the consolidation spans at most 0.6 of the first half; a flag keeps
  its width.
* **Cup and handle** needs two swing-high rims within 5 % of each other at least 20 candles apart, a bottom at
  least 10 % deep that is U-shaped (a V is no cup), and a handle dip of at most half the cup's depth. It is
  reported when the handle's low is confirmed.
* **Volume levels** put each candle's volume into the bin of its typical price over the last `lookback` candles; a
  bin that is a local maximum and holds at least `node_mult` times the average bin is a node. The profile is rebuilt
  every `lookback / 10` candles from earlier candles only, so levels hold still between rebuilds and nothing is
  seen in hindsight. The `volume_levels` outputs are not offered to `evolve` as numbers (they are bands, and move
  with the profile); use the alerts in rules, e.g. `volume_levels.breakout_up & rsi(14) > 50`.
