# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Slippage settings: how much worse than the close an order fills, in basis points per side.

The backtester charges it like the fee (see :func:`tradebot.backtest.engine.simulate`). It is 0 unless the
user sets it, so existing results do not change. Where the setting comes from, first match wins:

* an explicit value (``--slippage 5``, the spin box in the app),
* ``--slippage auto`` / the app's suggestion: :data:`SUGGESTED_BPS` for the symbol's asset class,
* ``config.toml``: ``[backtest] slippage_bps = 3`` for every symbol, or per asset class,
  ``slippage_bps = { crypto = 5, stock = 2 }``,
* 0.
"""

from __future__ import annotations

#: half spread plus a small move on liquid symbols; widen it for illiquid ones
SUGGESTED_BPS = {"crypto": 5.0, "stock": 2.0, "fx": 1.0}


def asset_class(symbol: str | None) -> str:
    """``crypto`` for pairs like ``BTC/USDT``, ``fx`` for ``EURUSD=X``, otherwise ``stock``."""
    symbol = (symbol or "").upper()
    if "/" in symbol:
        return "crypto"
    return "fx" if symbol.endswith("=X") else "stock"


def suggested_bps(symbol: str | None = None) -> float:
    """A sensible slippage for the symbol's asset class, in basis points."""
    return SUGGESTED_BPS[asset_class(symbol)]


def configured_bps(symbol: str | None = None, config: dict | None = None) -> float:
    """The slippage from ``config.toml``'s ``[backtest] slippage_bps`` (a number, or a table per asset class); else 0."""
    if config is None:
        from tradebot.config import load_config

        config = load_config()
    value = (config.get("backtest") or {}).get("slippage_bps", 0.0)
    if isinstance(value, dict):
        value = value.get(asset_class(symbol), 0.0)
    return max(0.0, float(value))


def resolve_bps(text: str | float | None, symbol: str | None = None) -> float:
    """An explicit number of basis points, ``"auto"`` (suggested for the asset class) or None (the config, else 0)."""
    if text is None:
        return configured_bps(symbol)
    if isinstance(text, str) and text.strip().lower() == "auto":
        return suggested_bps(symbol)
    value = float(text)
    if value < 0:
        raise ValueError("slippage cannot be negative")
    return value
