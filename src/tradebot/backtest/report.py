# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Backtest report (#152): expectancy and streaks, monthly returns, the underwater curve, CSV and HTML export."""

from __future__ import annotations

import html
from pathlib import Path

import numpy as np
import pandas as pd

from tradebot.backtest import metrics as m
from tradebot.core.series import OHLC

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
_DAY = pd.Timedelta(days=1)


def drawdown_series(equity: pd.Series) -> pd.Series:
    """Fall from the running peak as a fraction (0 at a new high, -0.2 at 20% below it)."""
    return equity / equity.cummax() - 1.0


def monthly_returns(returns: pd.Series) -> pd.DataFrame:
    """Compounded return per calendar month: one row per year, columns Jan..Dec and ``Year``."""
    if returns.empty:
        return pd.DataFrame(columns=[*MONTHS, "Year"], dtype=float)
    growth = (1 + returns).groupby([returns.index.year, returns.index.month]).prod() - 1
    table = growth.unstack().reindex(columns=range(1, 13))
    table.columns = MONTHS
    table["Year"] = (1 + returns).groupby(returns.index.year).prod() - 1
    table.index.name = "year"
    return table


def _longest_run(flags: np.ndarray) -> int:
    best = run = 0
    for f in flags:
        run = run + 1 if f else 0
        best = max(best, run)
    return best


def report_stats(equity: pd.Series, trades: pd.DataFrame) -> dict:
    """The numbers :func:`compute_metrics` lacks. Trade figures are fractions per trade; times are in days."""
    ret = trades["return"] if len(trades) else pd.Series(dtype=float)
    wins, losses = ret[ret > 0], ret[ret <= 0]
    dd = drawdown_series(equity) if len(equity) else pd.Series(dtype=float)
    below = (dd < 0).to_numpy()
    longest_uw = 0.0
    if below.any():  # longest stretch from a peak until equity is back at it (or the end)
        start = None
        for i, b in enumerate(below):
            if b and start is None:
                start = i
            if (not b or i == len(below) - 1) and start is not None:
                longest_uw = max(longest_uw, (equity.index[i] - equity.index[start - 1 if start else 0]) / _DAY)
                start = None
    hold = (trades["exit"] - trades["entry"]) / _DAY if len(trades) else pd.Series(dtype=float)
    return {
        "expectancy": float(ret.mean()) if len(ret) else 0.0,
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "longest_losing_streak": _longest_run((ret <= 0).to_numpy()),
        "time_under_water": float(below.mean()) if len(below) else 0.0,
        "longest_under_water_days": float(longest_uw),
        "avg_holding_days": float(hold.mean()) if len(hold) else 0.0,
    }


def regime_table(res, df: pd.DataFrame, **params) -> pd.DataFrame:
    """Per-regime metrics (#168): how the strategy did in bull / bear / sideways markets and in high / low volatility.

    The regime of each bar comes from the ``regime`` indicator on ``df`` (``params`` override its defaults) and is
    the one known when the position was taken, i.e. the bar before the return is earned. Columns: share of bars,
    compounded return, mean return and Sharpe ratio per bar (annualised), share of bars in the market, and the
    trades that were opened in the regime with their win rate.
    """
    from tradebot.indicators.regime import NAMES, Regime

    out = Regime.compute(OHLC.from_frame(df), **{**{k: v.default for k, v in Regime.params.items()}, **params})
    ret = res.returns
    idx = ret.index
    ppy = m.infer_periods_per_year(idx)
    pos = res.positions.shift(1).reindex(idx).fillna(0.0).abs().to_numpy()  # held while the return is earned
    trades = res.trades
    rows = {}
    for key, name in NAMES.items():
        at = pd.Series(np.asarray(out[key], dtype=float), index=df.index).reindex(idx)  # regime on each bar
        sel = (at.shift(1) == 1).to_numpy()  # the regime known when the position of the bar was taken
        r = ret[sel]
        n = int(sel.sum())
        std = float(r.std(ddof=1)) if n > 1 else 0.0
        entered = trades[trades["entry"].isin(idx[(at == 1).to_numpy()])]
        rows[name] = {
            "bars": n,
            "share": n / len(idx) if len(idx) else 0.0,
            "return": float((1 + r).prod() - 1) if n else 0.0,
            "mean_return": float(r.mean()) if n else 0.0,
            "sharpe": float(r.mean() / std * np.sqrt(ppy)) if std > 0 else 0.0,
            "exposure": float(pos[sel].mean()) if n else 0.0,
            "trades": len(entered),
            "win_rate": float((entered["return"] > 0).mean()) if len(entered) else 0.0,
        }
    return pd.DataFrame(rows).T.rename_axis("regime")


def fmt_regime(table: pd.DataFrame) -> pd.DataFrame:
    """:func:`regime_table` as display strings."""
    t = table.copy()
    for c in ("share", "return", "mean_return", "exposure", "win_rate"):
        t[c] = [f"{v * 100:.2f}%" for v in table[c]]
    t["sharpe"] = [f"{v:.2f}" for v in table["sharpe"]]
    for c in ("bars", "trades"):
        t[c] = [str(int(v)) for v in table[c]]
    return t


def fmt_stat(key: str, value) -> str:
    if key in ("longest_losing_streak",):
        return str(int(value))
    if key.endswith("_days"):
        return f"{value:.1f} d"
    return f"{value * 100:.2f}%"


def trades_frame(res) -> pd.DataFrame:
    """The trade list with holding time in days."""
    t = res.trades.copy()
    t["holding_days"] = (t["exit"] - t["entry"]) / _DAY
    return t


def export_trades_csv(res, path) -> Path:
    path = Path(path)
    trades_frame(res).to_csv(path, index=False)
    return path


def _svg_line(values: np.ndarray, w: int, h: int, color: str, fill: bool = False, lo=None, hi=None) -> str:
    if len(values) < 2:
        return ""
    lo = float(np.nanmin(values)) if lo is None else lo
    hi = float(np.nanmax(values)) if hi is None else hi
    span = (hi - lo) or 1.0
    xs = np.linspace(0, w, len(values))
    ys = h - (values - lo) / span * h
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    if fill:
        return f'<polygon points="0,{h - (0 - lo) / span * h:.1f} {pts} {w},{h - (0 - lo) / span * h:.1f}" fill="{color}" opacity="0.5"/>'
    return f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.5"/>'


def _heat_color(v: float, scale: float) -> str:
    if not np.isfinite(v):
        return "#2a2f3a"
    a = min(abs(v) / scale, 1.0)
    base = (38, 166, 91) if v >= 0 else (214, 69, 65)
    mix = tuple(int(42 + (c - 42) * a) for c in base)
    return "#%02x%02x%02x" % mix


def monthly_heatmap_svg(table: pd.DataFrame) -> str:
    cw, ch, left, top = 56, 24, 50, 22
    cols = list(table.columns)
    w, h = left + cw * len(cols), top + ch * len(table)
    scale = float(np.nanpercentile(np.abs(table[MONTHS].to_numpy(dtype=float)), 90)) if table[MONTHS].notna().any().any() else 1.0
    scale = scale or 1.0
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" font-family="sans-serif" font-size="11">']
    for j, c in enumerate(cols):
        out.append(f'<text x="{left + j * cw + cw / 2}" y="14" fill="#aab" text-anchor="middle">{c}</text>')
    for i, (year, row) in enumerate(table.iterrows()):
        y = top + i * ch
        out.append(f'<text x="{left - 6}" y="{y + 16}" fill="#aab" text-anchor="end">{year}</text>')
        for j, c in enumerate(cols):
            v = row[c]
            label = f"{v * 100:.1f}" if np.isfinite(v) else ""
            out.append(f'<rect x="{left + j * cw}" y="{y}" width="{cw - 2}" height="{ch - 2}" fill="{_heat_color(v, scale)}"/>'
                       f'<text x="{left + j * cw + cw / 2 - 1}" y="{y + 15}" fill="#fff" text-anchor="middle">{label}</text>')
    out.append("</svg>")
    return "".join(out)


def html_report(res, title: str = "", regimes: pd.DataFrame | None = None) -> str:
    """A self-contained HTML page (inline SVG charts, no external files) with metrics, charts and the trade list."""
    eq, dd = res.equity, drawdown_series(res.equity)
    stats = report_stats(eq, res.trades)
    monthly = monthly_returns(res.returns)
    title = html.escape(title or res.algorithm)
    rows = "".join(f"<tr><td>{html.escape(k.replace('_', ' '))}</td><td>{html.escape(m.fmt(k, v))}</td></tr>"
                   for k, v in res.metrics.items())
    rows += "".join(f"<tr><td>{html.escape(k.replace('_', ' '))}</td><td>{fmt_stat(k, v)}</td></tr>"
                    for k, v in stats.items())
    span = f"{eq.index[0]:%Y-%m-%d} to {eq.index[-1]:%Y-%m-%d}" if len(eq) else ""
    trades = trades_frame(res)
    trade_rows = "".join(
        f"<tr><td>{a:%Y-%m-%d}</td><td>{b:%Y-%m-%d}</td><td>{s}</td><td>{r * 100:.2f}%</td><td>{d:.1f}</td></tr>"
        for a, b, s, r, d in trades[["entry", "exit", "side", "return", "holding_days"]].itertuples(index=False))
    regime_html = ""
    if regimes is not None and len(regimes):
        f = fmt_regime(regimes)
        head = "".join(f"<th>{html.escape(c.replace('_', ' '))}</th>" for c in f.columns)
        body = "".join(f"<tr><td>{html.escape(str(n))}</td>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>"
                       for n, row in zip(f.index, f.itertuples(index=False)))
        regime_html = f"<h2>By market regime</h2><table><tr><th>regime</th>{head}</tr>{body}</table>"
    equity_svg = _svg_line(eq.to_numpy(dtype=float), 800, 200, "#4aa3ff")
    dd_svg = _svg_line(dd.to_numpy(dtype=float), 800, 120, "#d64541", fill=True, lo=min(float(dd.min()), -1e-9), hi=0.0)
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Backtest report: {title}</title>
<style>body{{background:#161a22;color:#dde;font-family:sans-serif;max-width:900px;margin:2em auto}}
table{{border-collapse:collapse}}td,th{{padding:3px 10px;border-bottom:1px solid #333;text-align:right}}
td:first-child{{text-align:left}}svg{{background:#1d222c}}</style></head><body>
<h1>{title}</h1><p>{span}</p>
<h2>Metrics</h2><table>{rows}</table>
<h2>Equity</h2><svg width="800" height="200">{equity_svg}</svg>
<h2>Underwater (drawdown, worst {dd.min() * 100:.1f}%)</h2><svg width="800" height="120">{dd_svg}</svg>
<h2>Monthly returns (%)</h2>{monthly_heatmap_svg(monthly)}
{regime_html}
<h2>Trades ({len(trades)})</h2><table><tr><th>entry</th><th>exit</th><th>side</th><th>return</th><th>days held</th></tr>{trade_rows}</table>
</body></html>"""


def export_html(res, path, title: str = "", regimes: pd.DataFrame | None = None) -> Path:
    path = Path(path)
    path.write_text(html_report(res, title, regimes), encoding="utf-8")
    return path
