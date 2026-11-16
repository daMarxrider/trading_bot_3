# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Trade journal (#159): every order the bots place or simulate, kept in a local SQLite database.

Each fill is one row (broker, mode ``live``/``paper``/``dry``, symbol, side, amount, price, fee, a note and
tags). Realised P&L comes from matching fills first-in-first-out per symbol and mode (:func:`fifo_lots`):
every closed piece is a *tax lot* with its open and close time, cost, proceeds, fees and profit. A sell
without holdings opens a short lot, so futures legs work too. Unrealised P&L is the open pieces valued at
given prices. Fees are split over the lots in proportion to the amount and are part of the profit. The
database is ``journal.sqlite`` next to the app's other state (``$TRADEBOT_JOURNAL`` overrides the path).
"""

from __future__ import annotations

import csv
import logging
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

LONG_TERM_DAYS = 365  #: a lot held longer than this counts as long-term in the export
FILL_COLUMNS = ("id", "time", "mode", "broker", "bot", "symbol", "side", "amount", "price", "fee", "order_id",
                "status", "note", "tags")
LOT_COLUMNS = ("year", "symbol", "mode", "direction", "amount", "open_time", "close_time", "open_price",
               "close_price", "cost", "proceeds", "fees", "pnl", "days", "term")
_SCHEMA = """CREATE TABLE IF NOT EXISTS fills (
    id INTEGER PRIMARY KEY AUTOINCREMENT, time TEXT NOT NULL, mode TEXT NOT NULL, broker TEXT NOT NULL,
    bot TEXT NOT NULL DEFAULT '', symbol TEXT NOT NULL, side TEXT NOT NULL, amount REAL NOT NULL,
    price REAL NOT NULL, fee REAL NOT NULL DEFAULT 0, order_id TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT '', note TEXT NOT NULL DEFAULT '', tags TEXT NOT NULL DEFAULT '')"""


def default_journal_path() -> Path:
    if os.environ.get("TRADEBOT_JOURNAL"):
        return Path(os.environ["TRADEBOT_JOURNAL"]).expanduser()
    from tradebot.config import os_config_dir

    return os_config_dir() / "journal.sqlite"


def order_fee(broker, order) -> float:
    """The fee of a placed order in the quote currency: what the exchange reports, else the paper account's
    book, else 0 (unknown)."""
    raw_fee = (getattr(order, "raw", None) or {}).get("fee")
    if isinstance(raw_fee, dict) and raw_fee.get("cost") is not None:
        return float(raw_fee["cost"])
    for o in reversed(getattr(broker, "orders", None) or []):
        if isinstance(o, dict) and o.get("id") == order.id:
            return float(o.get("fee") or 0.0)
    return 0.0


@dataclass
class Lot:
    """A closed (matched) piece of a position."""

    symbol: str
    mode: str
    direction: str  #: "long" or "short"
    amount: float
    open_time: datetime
    close_time: datetime
    open_price: float
    close_price: float
    fees: float
    close_id: int = 0  #: id of the fill that closed it

    @property
    def cost(self) -> float:
        return self.amount * (self.open_price if self.direction == "long" else self.close_price)

    @property
    def proceeds(self) -> float:
        return self.amount * (self.close_price if self.direction == "long" else self.open_price)

    @property
    def pnl(self) -> float:
        return self.proceeds - self.cost - self.fees

    @property
    def days(self) -> float:
        return (self.close_time - self.open_time).total_seconds() / 86400

    @property
    def term(self) -> str:
        return "long" if self.days > LONG_TERM_DAYS else "short"

    @property
    def year(self) -> int:
        return self.close_time.year

    def row(self) -> dict:
        return {"year": self.year, "symbol": self.symbol, "mode": self.mode, "direction": self.direction,
                "amount": self.amount, "open_time": self.open_time.isoformat(),
                "close_time": self.close_time.isoformat(), "open_price": self.open_price,
                "close_price": self.close_price, "cost": self.cost, "proceeds": self.proceeds, "fees": self.fees,
                "pnl": self.pnl, "days": round(self.days, 2), "term": self.term}


def _parse(ts: str) -> datetime:
    t = datetime.fromisoformat(ts)
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def fifo_lots(fills: list[dict]) -> tuple[list[Lot], list[dict]]:
    """Match fills first-in-first-out per (symbol, mode). Returns the closed lots (oldest close first) and the
    still open pieces ``{symbol, mode, direction, amount, price, time, fee}``; ``fee`` is what is still to be
    charged for that remainder."""
    books: dict[tuple, list[dict]] = {}
    lots: list[Lot] = []
    for f in sorted(fills, key=lambda f: (f["time"], f["id"])):
        book = books.setdefault((f["symbol"], f["mode"]), [])
        sign = 1 if f["side"] == "buy" else -1
        left, total, t = float(f["amount"]), float(f["amount"]), _parse(f["time"])
        fee_total = float(f["fee"])
        while left > 1e-12 and book and book[0]["sign"] != sign:
            o = book[0]
            q = min(left, o["amount"])
            share = q / o["amount"]
            fee = fee_total * q / total + o["fee"] * share
            lots.append(Lot(f["symbol"], f["mode"], "long" if o["sign"] > 0 else "short", q, o["time"], t,
                            o["price"], float(f["price"]), fee, f["id"]))
            o["fee"] -= o["fee"] * share
            o["amount"] -= q
            left -= q
            if o["amount"] <= 1e-12:
                book.pop(0)
        if left > 1e-12:
            book.append({"sign": sign, "amount": left, "price": float(f["price"]), "time": t,
                         "fee": fee_total * left / total})
    opens = [{"symbol": s, "mode": m, "direction": "long" if o["sign"] > 0 else "short", "amount": o["amount"],
              "price": o["price"], "time": o["time"], "fee": o["fee"]}
             for (s, m), book in books.items() for o in book]
    lots.sort(key=lambda lot: lot.close_time)
    return lots, opens


def unrealised(opens: list[dict], prices: dict[str, float]) -> dict[str, float]:
    """Unrealised P&L per symbol of the open pieces at ``prices`` (symbols without a price are left out)."""
    out: dict[str, float] = {}
    for o in opens:
        px = prices.get(o["symbol"])
        if px is None:
            continue
        gain = (px - o["price"]) * o["amount"] * (1 if o["direction"] == "long" else -1) - o["fee"]
        out[o["symbol"]] = out.get(o["symbol"], 0.0) + gain
    return out


def yearly_summary(lots: list[Lot]) -> list[dict]:
    """Per year of closing: lots, proceeds, cost, fees, gains, losses, net, and the net split short/long term."""
    years: dict[int, dict] = {}
    for lot in lots:
        y = years.setdefault(lot.year, {"year": lot.year, "lots": 0, "proceeds": 0.0, "cost": 0.0, "fees": 0.0,
                                         "gains": 0.0, "losses": 0.0, "net": 0.0, "short_term": 0.0,
                                         "long_term": 0.0})
        y["lots"] += 1
        y["proceeds"] += lot.proceeds
        y["cost"] += lot.cost
        y["fees"] += lot.fees
        y["gains" if lot.pnl > 0 else "losses"] += lot.pnl
        y["net"] += lot.pnl
        y[lot.term + "_term"] += lot.pnl
    return [years[k] for k in sorted(years)]


class Journal:
    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path else default_journal_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute(_SCHEMA)
        self._db.commit()

    def close(self) -> None:
        self._db.close()

    def record(self, symbol: str, side, amount: float, price: float, *, fee: float = 0.0, mode: str = "paper",
               broker: str = "", bot: str = "", order_id: str = "", status: str = "", note: str = "",
               tags: str = "", time: datetime | None = None) -> int:
        """Store one fill; returns its id."""
        side = getattr(side, "value", side)
        t = (time or datetime.now(timezone.utc)).isoformat()
        cur = self._db.execute(
            "INSERT INTO fills (time, mode, broker, bot, symbol, side, amount, price, fee, order_id, status, note, tags)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (t, mode, broker, bot, symbol, side, float(amount), float(price), float(fee), order_id, status, note, tags))
        self._db.commit()
        return int(cur.lastrowid)

    def record_order(self, order, broker, *, mode: str, bot: str = "", note: str = "") -> int:
        """Store a placed :class:`~tradebot.core.models.Order` (price and fee as the broker reports them)."""
        return self.record(order.symbol, order.side, order.filled or order.amount, order.price or 0.0,
                           fee=order_fee(broker, order), mode=mode, broker=getattr(broker, "name", ""), bot=bot,
                           order_id=order.id, status=getattr(order.status, "value", str(order.status)), note=note,
                           time=order.timestamp)

    def fills(self, mode: str | None = None, symbol: str | None = None, tag: str | None = None) -> list[dict]:
        """Fills oldest first. ``mode``: live, paper, dry, or "real" for live and paper (everything but dry runs)."""
        clauses = {"mode = ?": mode, "symbol = ?": symbol}
        if mode == "real":
            clauses = {"mode != ?": "dry", "symbol = ?": symbol}
        where = [(w, a) for w, a in clauses.items() if a]
        sql = "SELECT * FROM fills" + (" WHERE " + " AND ".join(w for w, _ in where) if where else "")
        rows = [dict(r) for r in self._db.execute(sql + " ORDER BY time, id", [a for _, a in where])]
        return [r for r in rows if tag in self.tag_list(r["tags"])] if tag else rows

    @staticmethod
    def tag_list(tags: str) -> list[str]:
        return [t.strip() for t in tags.split(",") if t.strip()]

    def annotate(self, fill_id: int, note: str | None = None, tags: str | None = None) -> bool:
        """Set the note and/or the comma-separated tags of a fill. False when there is no such fill."""
        if self._db.execute("SELECT 1 FROM fills WHERE id = ?", (fill_id,)).fetchone() is None:
            return False
        if note is not None:
            self._db.execute("UPDATE fills SET note = ? WHERE id = ?", (note, fill_id))
        if tags is not None:
            self._db.execute("UPDATE fills SET tags = ? WHERE id = ?", (", ".join(self.tag_list(tags)), fill_id))
        self._db.commit()
        return True

    def lots(self, mode: str | None = None, year: int | None = None) -> list[Lot]:
        return [lot for lot in fifo_lots(self.fills(mode))[0] if year is None or lot.year == year]

    def trades(self, mode: str | None = None, symbol: str | None = None, tag: str | None = None) -> list[dict]:
        """The fills, each with the realised P&L of the lots it closed (``pnl``, None for a fill that only opened)."""
        realised: dict[int, float] = {}
        for lot in fifo_lots(self.fills(mode))[0]:
            realised[lot.close_id] = realised.get(lot.close_id, 0.0) + lot.pnl
        return [{**f, "pnl": realised.get(f["id"])} for f in self.fills(mode, symbol, tag)]

    def summary(self, mode: str | None = None, prices: dict[str, float] | None = None) -> dict:
        fills = self.fills(mode)
        lots, opens = fifo_lots(fills)
        wins = sum(1 for lot in lots if lot.pnl > 0)
        return {"fills": len(fills), "lots": len(lots), "realised": sum(lot.pnl for lot in lots),
                "fees": sum(f["fee"] for f in fills), "win_rate": wins / len(lots) if lots else None,
                "unrealised": sum(unrealised(opens, prices or {}).values()), "open": len(opens)}

    def export_trades(self, path: Path | str, mode: str | None = None) -> int:
        """All fills with notes, tags and realised P&L as CSV; returns the number of rows."""
        rows = self.trades(mode)
        _write(path, list(FILL_COLUMNS) + ["pnl"], rows)
        return len(rows)

    def export_lots(self, path: Path | str, mode: str | None = None, year: int | None = None) -> int:
        """FIFO tax lots, one row per closed piece with its closing year, as CSV; ``year`` keeps one year."""
        rows = [lot.row() for lot in self.lots(mode, year)]
        _write(path, LOT_COLUMNS, rows)
        return len(rows)


def _write(path: Path | str, columns, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(columns), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
