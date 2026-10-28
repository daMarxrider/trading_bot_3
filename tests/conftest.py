# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
import gc

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _collect_garbage_between_gui_tests(request):
    """Free reference cycles between GUI tests, never in the middle of one.

    GUI views form cycles (signal lambdas), so the cyclic GC destroys them at an arbitrary
    later moment. When that falls inside Qt's event processing in a later test, Qt can touch
    a half-destroyed widget and segfault. Pausing automatic collection during a GUI test and
    collecting afterwards makes destruction happen at a safe, deterministic point.
    """
    if not hasattr(request.module, "QApplication"):  # only GUI test modules run Qt's event loop
        yield
        return
    gc.disable()
    try:
        yield
    finally:
        gc.enable()
        gc.collect()


@pytest.fixture(autouse=True)
def _no_symbol_name_lookups(monkeypatch):
    """The chart looks up the instrument's full name online after loading; tests stay offline."""
    import tradebot.symbols

    def offline(*args, **kwargs):
        raise ConnectionError("network disabled in tests")

    monkeypatch.setattr(tradebot.symbols, "yahoo_search_http", offline)


def make_ohlc(n: int = 400, seed: int = 0, freq: str = "D") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    open_ = np.r_[close[0], close[:-1]]
    spread = np.abs(rng.normal(0, 0.01, n)) * close
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    idx = pd.date_range("2020-01-01", periods=n, freq=freq, tz="UTC")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": rng.uniform(1e3, 1e4, n)}, index=idx
    )


@pytest.fixture
def ohlc() -> pd.DataFrame:
    return make_ohlc()
