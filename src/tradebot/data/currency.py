# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Which currency a symbol is quoted in, and how to convert it (#117)."""

from __future__ import annotations

STABLECOINS = {"USDT", "USDC", "BUSD", "TUSD", "FDUSD"}
#: Yahoo quotes some exchanges in a subunit: (currency, factor to the main unit)
SUBUNITS = {"GBp": ("GBP", 0.01), "GBX": ("GBP", 0.01), "ZAc": ("ZAR", 0.01), "ILA": ("ILS", 0.01)}


def yahoo_currency(symbol: str) -> str:
    """Currency Yahoo reports for a stock/ETF ticker (may be a subunit such as ``GBp``). Needs the network."""
    import yfinance as yf

    cur = yf.Ticker(symbol).fast_info.get("currency")
    if not cur:
        raise LookupError(f"no currency known for {symbol}")
    return str(cur)


def split_subunit(currency: str) -> tuple[str, float]:
    return SUBUNITS.get(currency, (currency.upper(), 1.0))


def quote_currency(symbol: str, lookup=yahoo_currency) -> tuple[str, float]:
    """``(currency, factor)``: prices of ``symbol`` times ``factor`` are in ``currency``.
    A crypto pair is quoted in its quote asset (stablecoins count as USD), a stock in what Yahoo says."""
    if "/" in symbol:
        quote = symbol.split("/")[1].split(":")[0]
        return ("USD" if quote in STABLECOINS else quote), 1.0
    return split_subunit(lookup(symbol))
