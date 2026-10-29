# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Chart view benchmark (#68): redraw and pan times for growing candle counts.

    uv run python scripts/bench_chart.py [--sizes 500 5000 50000] [--repeat 3]

Offscreen, 1400x800, dark theme. A *redraw* is ``ChartView.redraw()`` plus painting the canvas once; a *pan
frame* is one scroll step (a tenth of the visible span, auto-fit on) plus painting. The view shows half of the
candles while panning, like a user scrolled into the history.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from conftest import make_ohlc  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from tradebot.gui.chart import ChartView  # noqa: E402
from tradebot.gui.theme import ThemeManager  # noqa: E402
from tradebot.indicators.builtin import MACD, RSI, Bollinger  # noqa: E402


def best_ms(fn, repeat: int) -> float:
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times) * 1000


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sizes", type=int, nargs="+", default=[500, 5000, 50000])
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--pan-steps", type=int, default=20)
    args = ap.parse_args()

    app = QApplication.instance() or QApplication([])
    themes = ThemeManager(user_dir=Path(tempfile.mkdtemp()))
    themes.apply("dark")
    chart = ChartView(None, themes, ["TEST"])
    chart.resize(1400, 800)
    chart.show()
    app.processEvents()

    def paint():
        chart.canvas.grab()

    def redraw():
        chart.redraw()
        paint()

    def pan_frame_ms():
        chart.redraw(autorange=True)
        x0, x1 = chart.price.vb.viewRange()[0]
        mid, half = (x0 + x1) / 2, (x1 - x0) / 4
        chart.price.setXRange(mid - half, mid + half, padding=0)
        paint()
        step = half / 5
        t0 = time.perf_counter()
        for k in range(args.pan_steps):
            d = step if k % 2 == 0 else -step  # back and forth, so the view stays inside the data
            a, b = chart.price.vb.viewRange()[0]
            chart.price.setXRange(a + d, b + d, padding=0)
            paint()
        return (time.perf_counter() - t0) / args.pan_steps * 1000

    rows = []
    for n in args.sizes:
        chart.df = make_ohlc(n)
        chart.indicators = []
        candles = best_ms(redraw, args.repeat)
        pan_plain = pan_frame_ms()
        chart.indicators = [Bollinger(), RSI(), MACD()]
        full = best_ms(redraw, args.repeat)
        pan_full = pan_frame_ms()
        rows.append((n, candles, full, pan_plain, pan_full))

    print(f"{'candles':>8} | {'redraw':>9} | {'redraw +3 ind':>13} | {'pan frame':>9} | {'pan +3 ind':>10}")
    for n, candles, full, pan_plain, pan_full in rows:
        print(f"{n:>8,} | {candles:>6.0f} ms | {full:>10.0f} ms | {pan_plain:>6.0f} ms | {pan_full:>7.0f} ms")


if __name__ == "__main__":
    main()
