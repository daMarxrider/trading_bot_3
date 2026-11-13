# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Chart view benchmark (#68): redraw and pan times for growing candle counts.

    uv run python scripts/bench_chart.py [--sizes 500 5000 50000] [--repeat 3] [--onscreen] [--opengl]

Offscreen, 1400x800, dark theme. A *redraw* is ``ChartView.redraw()`` plus painting the canvas once; a *pan
frame* is one scroll step (a tenth of the visible span, auto-fit on) plus painting. The view shows half of the
candles while panning, like a user scrolled into the history.

``--onscreen`` uses the real display and repaints the visible window instead of grabbing it offscreen;
``--opengl`` (implies ``--onscreen``: it needs a GL context) gives the canvas the OpenGL viewport of the
"GPU rendering" setting (#71). Compare ``--onscreen`` with ``--opengl`` for the pan fps with the setting off/on.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

if not {"--onscreen", "--opengl"} & set(sys.argv):
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
    ap.add_argument("--onscreen", action="store_true", help="paint on the real display, not offscreen")
    ap.add_argument("--opengl", action="store_true", help="OpenGL viewport for the canvas (needs a display)")
    args = ap.parse_args()

    app = QApplication.instance() or QApplication([])
    themes = ThemeManager(user_dir=Path(tempfile.mkdtemp()))
    themes.apply("dark")
    chart = ChartView(None, themes, ["TEST"], opengl=args.opengl)
    if args.opengl and not chart.opengl:
        sys.exit(chart.opengl_note)
    chart.resize(1400, 800)
    chart.show()
    app.processEvents()
    onscreen = args.onscreen or args.opengl
    print(f"renderer: {'OpenGL' if chart.opengl else 'raster'}, {'onscreen' if onscreen else 'offscreen'}")

    def paint():
        if onscreen:
            chart.canvas.viewport().repaint()
            app.processEvents()
        else:
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

    print(f"{'candles':>8} | {'redraw':>9} | {'redraw +3 ind':>13} | {'pan frame':>9} | {'pan +3 ind':>10} | {'pan fps +3 ind':>14}")
    for n, candles, full, pan_plain, pan_full in rows:
        print(f"{n:>8,} | {candles:>6.0f} ms | {full:>10.0f} ms | {pan_plain:>6.0f} ms | {pan_full:>7.0f} ms | "
              f"{1000 / pan_full:>14.0f}")


if __name__ == "__main__":
    main()
