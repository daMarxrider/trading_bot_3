# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Read-only terminal charts with plotext (works in fish, bash, zsh, ...)."""

from __future__ import annotations

import shutil

import numpy as np
import pandas as pd
import plotext as plt

from tradebot.algorithms.base import BaseAlgorithm
from tradebot.indicators.base import BaseIndicator

THEME = "dark"


def render_chart(
    df: pd.DataFrame,
    title: str,
    indicators: list[BaseIndicator] = (),
    algorithm: BaseAlgorithm | None = None,
    line: bool = False,
    equity: pd.Series | None = None,
    width: int | None = None,
    height: int | None = None,
    window: int | None = None,
) -> str:
    """Build the chart and return it as an ANSI string.

    Indicators and signals are computed on the full ``df`` (so they are warmed
    up) and only the last ``window`` candles are drawn. Overlay indicators and
    buy/sell markers share the price pane; other indicators and the equity
    curve get panes of their own.
    """
    computed = {id(i): i(df) for i in indicators}
    positions = algorithm.run(df) if algorithm is not None else None
    if window:
        df = df.tail(window)
        computed = {k: v.loc[df.index] for k, v in computed.items()}
        positions = positions.loc[df.index] if positions is not None else None
        equity = equity.loc[df.index] if equity is not None else None
    panes = [i for i in indicators if not i.overlay]
    rows = 1 + len(panes) + (equity is not None)

    term = shutil.get_terminal_size((120, 40))
    fig = plt.figure
    fig.clear()
    fig.theme(THEME)
    fig.plot_size(width or term.columns, height or max(20, min(term.lines - 2, 14 + 10 * (rows - 1))))
    if rows > 1:
        fig.subplots(rows, 1)
    dates = list(df.index.tz_convert(None))

    main = _pane(fig, 1, rows)
    if line:
        main.draw(main.signal(dates, df["close"].tolist()).lines().label("close"))
    else:
        main.draw(main.candlestick({"date": dates, "open": df["open"].tolist(), "high": df["high"].tolist(),
                                    "low": df["low"].tolist(), "close": df["close"].tolist()}))
    for ind in indicators:
        if ind.overlay:
            _draw_frame(main, dates, computed[id(ind)])
    if positions is not None:
        change = np.diff(np.r_[0.0, positions.to_numpy()])
        closes = df["close"].to_numpy()
        for mask, symbol, color, label in ((change > 0, "▲", "green", "buy"), (change < 0, "▼", "red", "sell")):
            idx = np.flatnonzero(mask)
            if len(idx):
                sig = main.signal([dates[i] for i in idx], closes[idx].tolist(), marker=plt.marker(symbol, pixel=color))
                main.draw(sig.label(label))
    main.title(title)

    for n, ind in enumerate(panes, start=2):
        pane = _pane(fig, n, rows)
        _draw_frame(pane, dates, computed[id(ind)])
        for level in ind.levels:
            pane.line(level, pixel="gray")
        pane.title(ind.label)
    if equity is not None:
        pane = _pane(fig, rows, rows)
        pane.draw(pane.signal(list(equity.index.tz_convert(None)), equity.tolist()).lines().label("equity"))
        pane.title("equity")
    return str(fig.build())


def _pane(fig, row: int, rows: int):
    pane = fig.subplot(row, 1) if rows > 1 else fig
    pane.theme(THEME)
    pane.date().activate()
    return pane


def _draw_frame(pane, dates, frame: pd.DataFrame) -> None:
    for name, series in frame.items():
        mask = series.notna().to_numpy()
        if mask.any():
            pane.draw(pane.signal([d for d, m in zip(dates, mask) if m], series[mask].tolist()).lines().label(name))
