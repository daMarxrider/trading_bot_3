# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Portfolio analytics (#169): value over time, time- and money-weighted return, benchmark, correlation, risk
shares and rebalancing. Pure functions on pandas data, no network.

The history is rebuilt from the journal's fills (:mod:`tradebot.journal`): the quantity held per symbol is the
running sum of signed fills, valued at each day's close. A buy is money paid into the portfolio, a sell money taken
out (fees count against you). The return measures are the usual two:

* **Time-weighted return (TWR)**: chains the daily returns after removing that day's cash flow, so it does not depend
  on when money went in. This is the figure to compare with a benchmark.
* **Money-weighted return (MWR)**: the annualised internal rate of return of the cash flows plus today's value; it
  rewards good timing of the deposits.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd


def _day(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    return t.normalize()


def cash_flows(fills: list[dict]) -> pd.Series:
    """Money paid in (+) or taken out (-) per day: a buy is ``amount * price + fee``, a sell ``-(proceeds - fee)``."""
    rows: dict = {}
    for f in fills:
        gross = float(f["amount"]) * float(f["price"])
        flow = gross + float(f["fee"]) if f["side"] == "buy" else -(gross - float(f["fee"]))
        d = _day(f["time"])
        rows[d] = rows.get(d, 0.0) + flow
    return pd.Series(rows, dtype=float).sort_index()


def quantities(fills: list[dict], index: pd.DatetimeIndex) -> pd.DataFrame:
    """Quantity held per symbol at the end of every day of ``index`` (UTC days)."""
    cols = {}
    for sym in sorted({f["symbol"] for f in fills}):
        moves: dict = {}
        for f in fills:
            if f["symbol"] == sym:
                d = _day(f["time"])
                moves[d] = moves.get(d, 0.0) + (1 if f["side"] == "buy" else -1) * float(f["amount"])
        s = pd.Series(moves, dtype=float).sort_index().cumsum()
        cols[sym] = s.reindex(s.index.union(index)).ffill().reindex(index).fillna(0.0)
    return pd.DataFrame(cols, index=index)


def _daily(close: pd.Series) -> pd.Series:
    s = close.copy()
    s.index = pd.DatetimeIndex([_day(t) for t in s.index])
    return s[~s.index.duplicated(keep="last")].sort_index()


def value_history(fills: list[dict], closes: dict[str, pd.Series], end=None) -> pd.DataFrame:
    """Daily value per held symbol plus a ``total`` column, from the first fill to ``end`` (default: last close).

    A symbol without a close on a day uses the last one before it; symbols with no price data at all are left out."""
    fills = [f for f in fills if f["symbol"] in closes and len(closes[f["symbol"]])]
    if not fills:
        return pd.DataFrame(columns=["total"], dtype=float)
    start = min(_day(f["time"]) for f in fills)
    last = max(_daily(closes[s]).index[-1] for s in {f["symbol"] for f in fills})
    index = pd.date_range(start, _day(end) if end is not None else max(last, start), freq="D", tz="UTC")
    qty = quantities(fills, index)
    px = {}
    for s in qty.columns:
        c = _daily(closes[s])
        px[s] = c.reindex(index.union(c.index)).ffill().reindex(index)
    val = pd.DataFrame({s: qty[s] * px[s] for s in qty.columns})
    val["total"] = val.sum(axis=1, skipna=True)
    return val


def time_weighted_return(value: pd.Series, flows: pd.Series) -> pd.Series:
    """Cumulative TWR per day (0.10 = +10%). Day ``t`` returns ``(V_t - F_t) / V_{t-1} - 1`` with ``F_t`` the net
    money paid in that day, so deposits and withdrawals do not count as performance."""
    f = flows.reindex(value.index).fillna(0.0)
    prev = value.shift(1)
    r = ((value - f) / prev - 1).where(prev > 0, 0.0).fillna(0.0)
    return (1 + r).cumprod() - 1


def money_weighted_return(flows: pd.Series, end_value: float, end=None) -> tuple[float | None, float | None]:
    """``(total, annualised)`` money-weighted return: ``total`` is profit over money paid in, ``annualised`` the
    internal rate of return (None when there is no solution, e.g. no flows or under a day of history)."""
    if flows.empty:
        return None, None
    paid = float(flows[flows > 0].sum())
    total = (end_value + float(-flows[flows < 0].sum()) - paid) / paid if paid else None
    end_t = _day(end) if end is not None else max(flows.index[-1], _day(datetime.now(timezone.utc)))
    years = np.array([(end_t - t).days / 365.0 for t in flows.index])
    amounts = -flows.to_numpy(dtype=float)  # the investor's view: paying in is negative

    def npv(r: float) -> float:
        return float((amounts * (1 + r) ** years).sum() + end_value)

    lo, hi = -0.999, 100.0
    if years.max() <= 0 or npv(lo) * npv(hi) > 0:
        return total, None
    for _ in range(200):
        mid = (lo + hi) / 2
        if npv(lo) * npv(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return total, (lo + hi) / 2


def benchmark_return(close: pd.Series, start, end=None) -> pd.Series:
    """Cumulative return of a benchmark's close from ``start`` (the portfolio's first day)."""
    c = _daily(close)
    s = c[c.index >= _day(start)]
    if s.empty:
        return pd.Series(dtype=float)
    prior = c[c.index <= _day(start)]
    base = prior.iloc[-1] if len(prior) else s.iloc[0]
    out = s / base - 1
    return out[out.index <= _day(end)] if end is not None else out


def performance(fills: list[dict], closes: dict[str, pd.Series], benchmark: pd.Series | None = None) -> dict:
    """Everything the page and CLI show: ``value`` (frame), ``flows``, ``twr`` and ``bench`` (cumulative series),
    ``summary`` (dict: value, invested, profit, twr, mwr, mwr_annual, benchmark, excess)."""
    value = value_history(fills, closes)
    if value.empty:
        return {"value": value, "flows": pd.Series(dtype=float), "twr": pd.Series(dtype=float),
                "bench": pd.Series(dtype=float), "summary": {}}
    flows = cash_flows([f for f in fills if f["symbol"] in closes and len(closes[f["symbol"]])])
    twr = time_weighted_return(value["total"], flows)
    end_value = float(value["total"].iloc[-1])
    mwr, annual = money_weighted_return(flows, end_value, value.index[-1])
    bench = benchmark_return(benchmark, value.index[0], value.index[-1]) if benchmark is not None and len(benchmark) \
        else pd.Series(dtype=float)
    b = float(bench.iloc[-1]) if len(bench) else None
    summary = {"value": end_value, "invested": float(flows.sum()), "profit": end_value - float(flows.sum()),
               "twr": float(twr.iloc[-1]), "mwr": mwr, "mwr_annual": annual, "benchmark": b,
               "excess": float(twr.iloc[-1]) - b if b is not None else None}
    return {"value": value, "flows": flows, "twr": twr, "bench": bench, "summary": summary}


def returns_frame(closes: dict[str, pd.Series]) -> pd.DataFrame:
    """Daily returns per symbol on the days all of them trade (crypto + stocks: the stock calendar)."""
    df = pd.DataFrame({s: _daily(c) for s, c in closes.items() if len(c)})
    return df.dropna().pct_change().dropna()


def correlation(closes: dict[str, pd.Series]) -> pd.DataFrame:
    """Correlation matrix of the daily returns."""
    return returns_frame(closes).corr()


def risk_contributions(closes: dict[str, pd.Series], weights: dict[str, float]) -> pd.DataFrame:
    """Per holding: ``weight``, annualised ``volatility`` and ``risk_share``, the part of the portfolio's variance
    it causes (``w_i * (Sigma w)_i / w'Sigma w``; the shares add up to 1). A high share next to a small weight
    marks a volatile or highly correlated holding."""
    rets = returns_frame({s: c for s, c in closes.items() if s in weights})
    names = list(rets.columns)
    w = np.array([weights[s] for s in names], dtype=float)
    w = w / w.sum() if w.sum() else w
    cov = rets.cov().to_numpy() * 252
    var = float(w @ cov @ w)
    share = (w * (cov @ w)) / var if var > 0 else np.full(len(names), np.nan)
    return pd.DataFrame({"weight": w, "volatility": np.sqrt(np.diag(cov)), "risk_share": share}, index=names)


def parse_targets(text: str) -> dict[str, float]:
    """``"AAPL=40, MSFT=60"`` (or ``AAPL:40``) -> weights that add up to 1 (any scale works)."""
    out: dict[str, float] = {}
    for part in text.replace(";", ",").split(","):
        if not part.strip():
            continue
        sym, sep, val = part.replace(":", "=").partition("=")
        if not sep:
            raise ValueError(f"target {part.strip()!r}: expected SYMBOL=percent")
        out[sym.strip()] = float(val.strip().rstrip("%"))
    total = sum(out.values())
    if not out or total <= 0 or any(v < 0 for v in out.values()):
        raise ValueError("targets must be positive numbers")
    return {s: v / total for s, v in out.items()}


def rebalance(values: dict[str, float], prices: dict[str, float], targets: dict[str, float],
              tolerance: float = 0.0) -> pd.DataFrame:
    """What to buy or sell to reach the target weights, with the money kept constant (sells fund the buys).

    ``values`` is the current value per holding, ``prices`` the price per unit (needed for every target and
    holding), ``targets`` weights summing to 1; a holding without a target gets 0 and is sold. Drifts of at most
    ``tolerance`` (in weight) are left alone. Columns: weight, target, trade_value (+ buy / - sell), quantity,
    action."""
    total = sum(values.values())
    t = {s: w / sum(targets.values()) for s, w in targets.items()}
    rows = []
    for sym in sorted(set(values) | set(t), key=lambda s: -values.get(s, 0.0)):
        cur, tgt = values.get(sym, 0.0), t.get(sym, 0.0)
        diff = tgt * total - cur
        px = prices.get(sym)
        skip = abs(diff / total) <= tolerance if total else True
        trade = 0.0 if skip else diff
        rows.append({"symbol": sym, "value": cur, "weight": cur / total if total else 0.0, "target": tgt,
                     "trade_value": trade, "quantity": trade / px if px else float("nan"),
                     "action": "hold" if trade == 0 else "buy" if trade > 0 else "sell"})
    return pd.DataFrame(rows)


def analyse(get_close, summary, fills: list[dict], benchmark: str = "", targets: dict[str, float] | None = None,
            tolerance: float = 0.0) -> dict:
    """The whole report for the page and the CLI. ``get_close(symbol)`` gives a daily close series (it may raise,
    that symbol is then skipped), ``summary`` is a :class:`~tradebot.portfolio.PortfolioSummary`, ``fills`` the
    journal fills. Keys: ``performance``, ``correlation``, ``risk``, ``rebalance`` (None when not computable).
    Holdings are named as the journal does: stocks bare, crypto ``BTC/USDT``."""
    cache: dict[str, pd.Series | None] = {}

    def close(sym: str):
        if sym not in cache:
            try:
                cache[sym] = get_close(sym)
            except Exception:
                cache[sym] = None
        return cache[sym]

    def sym_of(h) -> str:
        return f"{h.symbol}/USDT" if h.asset_class.value == "crypto" and "/" not in h.symbol else h.symbol

    out: dict = {"performance": None, "correlation": None, "risk": None, "rebalance": None}
    closes = {s: c for s in {f["symbol"] for f in fills} if (c := close(s)) is not None}
    bench = close(benchmark) if benchmark else None
    if fills and closes:
        out["performance"] = performance(fills, closes, bench)
    held = {sym_of(h): h for h in summary.holdings if h.value > 0 and h.symbol not in {"USD", "USDT", "USDC"}}
    hc = {s: c for s in held if (c := close(s)) is not None}
    if len(hc) >= 2:
        out["correlation"] = correlation(hc)
        total = sum(held[s].value for s in hc)
        out["risk"] = risk_contributions(hc, {s: held[s].value / total for s in hc})
    if targets:
        values = {sym_of(h): h.value for h in summary.holdings}
        prices = {sym_of(h): h.price for h in summary.holdings}
        for s in targets:
            if s not in prices and (c := close(s)) is not None and len(c):
                prices[s] = float(c.iloc[-1])
        out["rebalance"] = rebalance(values, prices, targets, tolerance)
    return out
