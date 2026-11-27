# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Neo-brokers without an official trading API (semi-automated execution).

Holdings come from a CSV file (``symbol,quantity,avg_cost[,currency]``,
e.g. converted from the broker's own export); market data comes from the
shared data engine; orders become manual :class:`OrderIntent` objects.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

from tradebot.brokers.base import ManualExecutionBroker
from tradebot.core.models import AssetClass, Balance, Position, guess_asset_class


def _number(text: str, column: str, path: Path, line: int) -> float:
    try:
        value = float(text)
    except ValueError:
        value = math.nan
    if not math.isfinite(value):
        raise ValueError(f"{path}, line {line}: {column} {text!r} is not a number (use a dot as the decimal point)")
    return value


class HoldingsFileBroker(ManualExecutionBroker):
    """Semi-automated broker whose holdings are read from ``credentials["holdings_csv"]``."""

    credential_keys = ("holdings_csv",)

    def _holdings_path(self) -> Path:
        path = self.credentials.get("holdings_csv")
        if not path:
            raise NotImplementedError(
                f"{self.name}: no API for holdings; set brokers.{self.name}.holdings_csv in the config "
                "(columns: symbol,quantity,avg_cost)"
            )
        return Path(path).expanduser()

    def get_positions(self) -> list[Position]:
        """Rows of ``symbol,quantity[,avg_cost]`` (any case, comma or semicolon separated, UTF-8 with or without
        the BOM Excel writes); rows without a symbol or quantity are skipped. A file without those columns or
        with a value that isn't a number raises ValueError naming the line, so holdings never vanish silently."""
        path = self._holdings_path()
        if not path.is_file():
            raise FileNotFoundError(f"{self.name}: the holdings file {path} does not exist; fix "
                                    f"brokers.{self.name}.holdings_csv in the config")
        out = []
        with path.open(newline="", encoding="utf-8-sig") as fh:
            header = fh.readline()
            fh.seek(0)
            delimiter = ";" if header.count(";") > header.count(",") else ","
            reader = csv.DictReader(fh, delimiter=delimiter)
            columns = {(k or "").strip().lower() for k in reader.fieldnames or []}
            if header.strip() and not {"symbol", "quantity"} <= columns:
                raise ValueError(f"{path}: needs the columns symbol and quantity (optional avg_cost); "
                                 f"found {', '.join(sorted(c for c in columns if c)) or 'none'}")
            for row in reader:
                row = {k.strip().lower(): (v or "").strip() for k, v in row.items() if isinstance(k, str)}
                if not row.get("symbol") or not row.get("quantity"):
                    continue
                cost = row.get("avg_cost")
                out.append(Position(row["symbol"], _number(row["quantity"], "quantity", path, reader.line_num),
                                    _number(cost, "avg_cost", path, reader.line_num) if cost else None,
                                    guess_asset_class(row["symbol"])))
        return out

    def get_balance(self) -> dict[str, Balance]:
        return {p.symbol: Balance(p.symbol, p.quantity) for p in self.get_positions()}


class ScalableCapitalBroker(HoldingsFileBroker):
    """Scalable Capital: no public trading API -> tracking + manual order intents."""

    name = "scalable"
    title = "Scalable Capital"
    asset_classes = frozenset({AssetClass.STOCK})


class TradeRepublicBroker(HoldingsFileBroker):
    """Trade Republic extension point built on `pytr` (``pip install tradebot[traderepublic]``).

    :meth:`connect` performs the pytr web login (phone number + PIN + 2FA
    code on first use). Live portfolio sync and order placement are not wired
    up yet, so holdings still come from the CSV and orders stay manual.
    """

    name = "traderepublic"
    title = "Trade Republic"
    asset_classes = frozenset({AssetClass.STOCK, AssetClass.CRYPTO})
    credential_keys = ("phone_no", "pin", "holdings_csv")

    def connect(self):
        try:
            from pytr.account import login
        except ImportError as exc:
            raise ImportError("Trade Republic support needs `pytr`: pip install 'tradebot[traderepublic]'") from exc
        self.api = login(phone_no=self.credentials.get("phone_no"), pin=self.credentials.get("pin"))
        return self.api
