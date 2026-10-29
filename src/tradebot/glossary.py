# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Plain-language explanations of the trading and optimizer terms the app shows (#70).

One place for the texts, so the backtest lab's tooltips, its results table and any later help page or CLI
output say the same thing. Each entry: what it means in everyday words first, the precise definition second,
1–3 short sentences.
"""

from __future__ import annotations

#: Backtest metrics, keyed like :func:`tradebot.backtest.metrics.compute_metrics`.
METRICS = {
    "total_return": "How much the money grew (or shrank) over the whole test. +50% means 1,000 became 1,500.",
    "cagr": "The total return as an average per year (compound annual growth rate), so tests of different "
            "lengths can be compared.",
    "sharpe": "Return compared with how bumpy the ride was: the average return divided by how much it swings, "
              "per year. Rough guide: above 1 is good, above 2 is very good, below 0 lost money.",
    "sortino": "Like Sharpe, but only the drops count as risk, since swings upwards don't hurt. "
               "Higher is better.",
    "max_drawdown": "The biggest fall from a high point to a later low. It answers: how much would I have been "
                    "down at the worst moment?",
    "trades": "How many trades were completed. With only a handful, the result says more about luck than about "
              "the strategy.",
    "win_rate": "The share of trades that made money. A low win rate can still be profitable if the wins are much "
                "bigger than the losses.",
    "profit_factor": "Money won divided by money lost, over all trades. Above 1 means profitable overall; 2 means "
                     "twice as much won as lost.",
    "exposure": "The share of the time the money was in the market. The rest of the time it sat in cash, "
                "earning nothing and risking nothing.",
}

#: What the optimizer maximises ("Objective").
OBJECTIVES = {
    "sharpe": "Best return for the risk taken (Sharpe ratio). Prefers steady gains over wild swings. "
              "A good default.",
    "calmar": "Best yearly return compared with the worst fall from a peak (Calmar ratio). Prefers strategies "
              "with small drawdowns.",
    "return": "Highest profit, ignoring risk. Tends to pick risky settings that fit the past by luck.",
}

#: Search methods ("Method").
METHODS = {
    "ga": "Genetic algorithm: tries many parameter sets, keeps the best, and mixes and slightly changes them "
          "for the next round, like breeding.",
    "pso": "Particle swarm: a swarm of guesses that keep drifting towards the best settings found so far, by "
           "themselves and by the swarm.",
    "both": "Runs both methods on the same data and budget and compares them. If they disagree a lot, the "
            "\"best\" settings are probably fragile.",
}

#: The built-in strategies ("Algorithm"); a plugin strategy shows its own description instead.
STRATEGIES = {
    "rsi_threshold": "Buys after the price has fallen sharply (the RSI is low) and sells after it has risen sharply "
                     "(the RSI is high). Bets that big moves bounce back.",
    "macd_cross": "In the market while the short-term trend (MACD line) is above its own average (signal line). "
                  "Follows trends.",
    "ichimoku_cloud": "In the market while the price is above the Ichimoku \"cloud\", a band built from recent "
                      "highs and lows. Follows trends.",
    "sma_cross": "In the market while a short moving average of the price is above a long one, i.e. while the "
                 "price is trending up.",
    "bollinger_reversion": "Buys when the price falls unusually far below its average (below the lower Bollinger "
                           "band) and sells once it is back at the average.",
    "range": "Buys at a fixed price and sells at a higher fixed price, with an optional stop loss. For a market "
             "moving sideways in a known range.",
    "vote": "Combines several strategies: buys when most of them say buy and sells when most say sell.",
    "elliott_wave": "Looks for the wave pattern of Elliott wave theory and trades the breakout of its strongest "
                    "wave (wave 3), with a stop loss and a price target.",
}

#: Inputs and labels of the backtest lab.
TERMS = {
    "symbol": "The market to test on, e.g. BTC/USDT or AAPL.",
    "timeframe": "How much time one candle covers: 1d is one candle per day, 1h one per hour. Shorter "
                 "timeframes mean more trades and more fees.",
    "candles": "How much history to test on, in candles. More history gives more trades and a more "
               "trustworthy result.",
    "dates": "Optionally a fixed period to test, instead of the latest candles.",
    "fee": "The cost of each buy and each sell, as a fraction: 0.001 is 0.1%. A full trade (in and out) pays it "
           "twice.",
    "algorithm": "The trading strategy: the rules that decide when to buy and when to sell.",
    "members": "The strategies that vote. Each one says buy or sell, and the vote acts when enough of them "
               "agree (the threshold). All checks every known strategy.",
    "parameters": "The strategy's settings, such as how many days an average covers. Run backtest uses these "
                  "values; Optimize searches for good ones and fills them in.",
    "method": "How the optimizer searches for good parameters.",
    "population": "How many parameter sets are tried in each round. More finds better settings but takes "
                  "longer.",
    "generations": "How many rounds of improvement the optimizer runs. More rounds refine the result but take "
                   "longer.",
    "objective": "What \"best\" means to the optimizer.",
    "backend": "Where the optimizer computes: the processor (CPU) or a graphics card (CUDA for NVIDIA, ROCm for "
               "AMD). It changes only the speed, not the results.",
    "in-sample": "The first 70% of the candles, which the optimizer tuned the parameters on. Results here are "
                 "always flattering.",
    "out-of-sample": "The last 30% of the candles, which the optimizer never saw. This shows whether the "
                     "settings work on new data; a much worse result than in-sample means overfitting.",
    "backtest": "The strategy replayed on past prices with the current parameters.",
    "optimize": "Searches for the parameters that score best on the objective, using only the in-sample part, "
                "then tests the winner on the out-of-sample part and fills it in.",
    "show_on_chart": "Opens the symbol in the chart with this strategy's buy and sell signals.",
    "equity": "What the account would have been worth over time, compared with simply buying at the start and "
              "holding (dashed line).",
    "verdict": "Whether GA and PSO agree. Consistent: same settings, similar score, the optimum is solid. "
               "Fragile: different settings, similar score, the exact values are mostly noise. Divergent: one "
               "method found clearly better settings than the other.",
    "agreement": "How close the two methods' parameters are, from 0% (opposite ends of the range) to 100% "
                 "(identical).",
    "gap": "How much the two methods' in-sample scores differ. Small means they found equally good settings.",
    "param_diff": "How far apart the two methods' values are, as a share of the range the optimizer searched.",
}


def explain(key: str) -> str:
    """The explanation for any term above, or "" if there is none."""
    for table in (TERMS, METRICS, OBJECTIVES, METHODS, STRATEGIES):
        if key in table:
            return table[key]
    return ""
