# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""The chart's Live button (#158): the forming candle follows the market while it is on.

Prices come from a :class:`tradebot.data.stream.MarketStream` (the exchange's websocket for crypto, Yahoo's
price stream for stocks, polling as the fallback) on its own thread; they reach the GUI thread through a Qt
signal and the chart redraws at most once a second. The button's icon shows the state: the stream icon while
it streams, the alert icon while it polls or reconnects (the tooltip says why).
"""

from __future__ import annotations

import pandas as pd
from PyQt6.QtCore import QObject, QTimer, Qt, pyqtSignal
from PyQt6.QtWidgets import QToolButton

from tradebot.core.models import split_pair
from tradebot.core.timeframes import to_timedelta
from tradebot.data import stream as streams
from tradebot.gui.icons import bind_icon

IDLE_TIP = "Live: keep the last candle moving with the market (websocket, else polling every few seconds)"
TIPS = {
    "connecting": "Live: connecting to the price stream …",
    "streaming": "Live: streaming over a websocket",
    "reconnecting": "Live: the connection dropped, reconnecting",
    "polling": "Live: polling for prices (no websocket right now)",
}


def merge_candle(df: pd.DataFrame, candle: tuple, step: pd.Timedelta) -> pd.DataFrame | None:
    """``df`` with the forming ``candle`` ``(start, open, high, low, close, volume)`` merged into its last row, or
    appended when it is the next candle; None when it is older than the last row (nothing to change)."""
    if df is None or df.empty:
        return None
    start, o, h, lo, c, v = candle
    last = df.index[-1]
    if start > last and start < last + step:  # the source's grid is offset from ours: the same candle
        start = last
    out = df.copy()
    if start in out.index:
        if start < last and start != out.index[-2]:
            return None  # far behind: a late update from before the last load
        row = out.loc[start]
        out.loc[start, ["high", "low", "close"]] = [max(row["high"], h), min(row["low"], lo), c]
        if v and "volume" in out.columns:
            out.loc[start, "volume"] = v
        return out
    if start < last:
        return None
    new = pd.DataFrame({"open": [o], "high": [h], "low": [lo], "close": [c]}, index=pd.DatetimeIndex([start]))
    if "volume" in out.columns:
        new["volume"] = v
    return pd.concat([out, new.reindex(columns=out.columns)])


class LiveCandles(QObject):
    """The Live button and the stream behind it, for one :class:`~tradebot.gui.chart.ChartView`."""

    tick = pyqtSignal(object)  #: (generation, Tick), from the stream's thread
    state = pyqtSignal(object)  #: (generation, state, note)

    def __init__(self, chart, themes, open_stream=None, redraw_ms: int = 1000):
        super().__init__(chart)
        self.chart, self.themes = chart, themes
        self._open = open_stream or streams.open_stream
        self.button = QToolButton(text="Live", checkable=True, toolTip=IDLE_TIP)
        self.button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        bind_icon(self.button, "stream", themes, off="muted", on="accent_ink", size=14)
        self.button.toggled.connect(self._toggled)
        self.stream = None
        self.key: tuple[str, str] | None = None
        self.current = "stopped"
        self._gen = 0
        self.tick.connect(self._on_tick)
        self.state.connect(self._on_state)
        self._redraw = QTimer(self, singleShot=True, interval=redraw_ms)
        self._redraw.timeout.connect(lambda: self.chart.redraw())

    @property
    def on(self) -> bool:
        return self.button.isChecked()

    def follow(self) -> None:
        """The chart loaded data: stream what it now shows (when Live is on and that changed)."""
        if self.on and self.key != self._wanted():
            self._start()

    def stop(self) -> None:
        self._gen += 1
        if self.stream is not None:
            self.stream.stop(timeout=2.0)
        self.stream, self.key, self.current = None, None, "stopped"
        self._redraw.stop()
        self._show("stopped", "")

    def _wanted(self) -> tuple[str, str]:
        return self.chart._loaded_symbol, self.chart.tf.currentText()

    def _toggled(self, on: bool) -> None:
        if on:
            self._start()
        else:
            self.stop()
            self.chart.status.emit("live updates off")

    def _start(self) -> None:
        self.stop()
        symbol, tf = self._wanted()
        if not symbol or self.chart.df is None:
            return  # starts with the next load
        if split_pair(symbol):
            self.chart.status.emit("Live follows a single symbol; a ratio chart updates with Load")
            self._uncheck()
            return
        try:
            s = self._open(symbol, tf, engine=self.chart.engine)
        except Exception as exc:
            self.chart.status.emit(f"no live prices for {symbol}: {exc}")
            self._uncheck()
            return
        gen = self._gen
        s.on_tick(lambda t, g=gen: self.tick.emit((g, t)))
        s.on_state(lambda st, note, g=gen: self.state.emit((g, st, note)))
        self.stream, self.key = s, (symbol, tf)
        s.start()

    def _uncheck(self) -> None:
        self.button.blockSignals(True)
        self.button.setChecked(False)
        self.button.blockSignals(False)
        self._show("stopped", "")

    def _on_tick(self, item) -> None:
        gen, tick = item
        if gen != self._gen or self.key is None or tick.symbol != self.chart._loaded_symbol:
            return
        df = merge_candle(self.chart.df, tick.candle, to_timedelta(self.key[1]))
        if df is None:
            return
        self.chart.df = df
        if not self._redraw.isActive():
            self._redraw.start()

    def _on_state(self, item) -> None:
        gen, state, note = item
        if gen != self._gen:
            return
        before, self.current = self.current, state
        self._show(state, note)
        if state != before and state in ("streaming", "polling", "reconnecting"):
            symbol = self.key[0] if self.key else ""
            self.chart.status.emit(f"{symbol}: live, {TIPS[state].split(': ', 1)[1]}" + (f" ({note})" if note else ""))

    def _show(self, state: str, note: str) -> None:
        tip = TIPS.get(state, IDLE_TIP)
        self.button.setToolTip(tip + (f"\n{note}" if note else ""))
        degraded = state in ("polling", "reconnecting")
        bind_icon(self.button, "alert" if degraded else "stream", self.themes, off="muted", on="accent_ink", size=14)
