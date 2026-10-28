# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Neo-brokers without a usable trading API (semi-automated execution).

Scalable Capital lives in :mod:`tradebot.brokers.scalable` (official CLI).
Holdings here come from a CSV file (``symbol,quantity,avg_cost[,currency]``,
e.g. converted from the broker's own export); market data comes from the
shared data engine; orders become manual :class:`OrderIntent` objects.
"""

from __future__ import annotations

import csv
from pathlib import Path

from tradebot.brokers.base import ManualExecutionBroker
from tradebot.core.models import AssetClass, Balance, Position, guess_asset_class


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
        out = []
        with self._holdings_path().open(newline="") as fh:
            for row in csv.DictReader(fh):
                row = {k.strip().lower(): (v or "").strip() for k, v in row.items() if k}
                if not row.get("symbol") or not row.get("quantity"):
                    continue
                cost = row.get("avg_cost")
                out.append(Position(row["symbol"], float(row["quantity"]), float(cost) if cost else None,
                                    guess_asset_class(row["symbol"])))
        return out

    def get_balance(self) -> dict[str, Balance]:
        return {p.symbol: Balance(p.symbol, p.quantity) for p in self.get_positions()}


class TradeRepublicBroker(HoldingsFileBroker):
    """Trade Republic extension point built on `pytr` (``pip install tradebot[traderepublic]``).

    :meth:`connect` performs the pytr web login (phone number + PIN + 2FA
    code on first use). Live portfolio sync and order placement are not wired
    up yet, so holdings still come from the CSV and orders stay manual.
    """

    name = "traderepublic"
    asset_classes = frozenset({AssetClass.STOCK, AssetClass.CRYPTO})
    credential_keys = ("phone_no", "pin", "holdings_csv")

    def connect(self):
        try:
            from pytr.account import login
        except ImportError as exc:
            raise ImportError("Trade Republic support needs `pytr`: pip install 'tradebot[traderepublic]'") from exc
        self.api = login(phone_no=self.credentials.get("phone_no"), pin=self.credentials.get("pin"))
        return self.api
