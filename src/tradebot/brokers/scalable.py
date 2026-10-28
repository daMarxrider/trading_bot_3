# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Scalable Capital via the official Scalable CLI (``sc``).

https://github.com/ScalableCapital/scalable-cli is Scalable's own command line
on top of the Scalable API (Apache-2.0). This adapter shells out to
``sc ... --json`` and maps its machine-readable envelopes onto the broker
interface:

* holdings (incl. FIFO cost basis), cash, quotes -> ``get_positions`` /
  ``get_balance`` / ``get_ticker``
* orders use Scalable's mandatory two-step flow: ``place_order`` only runs
  **phase 1** (a binding pre-trade preview with costs and disclosures) and
  returns an :class:`OrderIntent` holding the confirmation id. Phase 2 runs
  in :meth:`ScalableCapitalBroker.confirm`, which must only be triggered by
  an explicit user action after the full preview was shown - the CLI forbids
  automatic phase-2 execution, so the broker stays ``SEMI_AUTO``.

Setup (once, by a human): enable *Profile → Security → Agentic Investing* on
the Scalable web platform, install ``sc`` and run ``sc login``. Sessions last
up to seven days. Without ``sc`` the broker falls back to ``holdings_csv``.

Config (``[brokers.scalable]``)::

    sc_path = "sc"                    # binary, if not on PATH
    portfolio_id = "..."              # only if you have several portfolios
    venue = "gettex"                  # optional execution venue
    holdings_csv = "~/scalable.csv"   # fallback when sc is unavailable
    [brokers.scalable.symbols]        # optional ticker -> ISIN overrides
    AAPL = "US0378331005"
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import time
from typing import Callable

from tradebot.brokers.base import OrderIntent
from tradebot.brokers.stocks import HoldingsFileBroker
from tradebot.core.models import AssetClass, Balance, Order, OrderStatus, OrderType, Position, Side, Ticker

log = logging.getLogger(__name__)
ISIN_RE = re.compile(r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$")


class ScalableCLIError(RuntimeError):
    def __init__(self, code: str, message: str, command: str = ""):
        hint = " (run `sc login` in a terminal)" if "login" in code or "session" in code or "auth" in code else ""
        super().__init__(f"sc {command}: {code}: {message}{hint}")
        self.code, self.command = code, command


def _num(value) -> float | None:
    """The CLI emits decimals as numbers or strings; ``null`` means unavailable, not zero."""
    return None if value is None else float(value)


def yahoo_symbol_for_isin(isin: str) -> str | None:
    """Best-effort ISIN -> Yahoo ticker lookup (for charts, prices and dividends)."""
    try:
        import yfinance as yf

        quotes = yf.Search(isin, max_results=1).quotes
        return quotes[0]["symbol"] if quotes else None
    except Exception as exc:
        log.debug("ISIN lookup failed for %s: %s", isin, exc)
        return None


def format_preview(details: dict) -> str:
    """Human-readable rendering of *every* field of a phase-1 preview.

    Scalable requires all pre-trade information to be shown unchanged before
    confirmation; values are printed exactly as delivered, ``null`` literally.
    """
    pres = details.get("presentation") or {}
    lines = []
    for key in pres.get("section_order", []):
        section = pres.get("sections", {}).get(key, {})
        lines.append(f"── {section.get('title', key)} ──")
        for f in section.get("fields", []):
            value = f.get("value")
            text = "null" if value is None else json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
            lines.append(f"  {f.get('label', f.get('path'))}: {text}")
    if not lines:  # unknown presentation format: show the raw preview rather than hiding anything
        lines.append(json.dumps(details.get("raw", details), indent=2, ensure_ascii=False))
    return "\n".join(lines)


class ScalableCapitalBroker(HoldingsFileBroker):
    """Scalable Capital through the official ``sc`` CLI (semi-automated execution)."""

    name = "scalable"
    asset_classes = frozenset({AssetClass.STOCK, AssetClass.CRYPTO})
    credential_keys = ("sc_path", "portfolio_id", "venue", "symbols", "holdings_csv")

    def __init__(self, credentials=None, data=None, on_intent=None,
                 runner: Callable[[list[str]], dict] | None = None,
                 symbol_resolver: Callable[[str], str | None] = yahoo_symbol_for_isin):
        super().__init__(credentials, data, on_intent)
        self._runner = runner
        self._resolve_symbol = symbol_resolver
        overrides = dict(self.credentials.get("symbols") or {})
        self._isin_by_symbol: dict[str, str] = {k.upper(): v for k, v in overrides.items()}
        self._symbol_by_isin: dict[str, str] = {v: k for k, v in overrides.items()}

    # -- sc plumbing ---------------------------------------------------------------------------------
    @property
    def available(self) -> bool:
        return self._runner is not None or shutil.which(self.credentials.get("sc_path", "sc")) is not None

    def run(self, *args: str) -> dict:
        """Run ``sc <args> --json`` and return the envelope's ``data`` (raises on ``ok: false``)."""
        argv = [*args]
        if self.credentials.get("portfolio_id") and args[0] == "broker" and args[1] not in ("portfolios", "context"):
            argv += ["--portfolio-id", self.credentials["portfolio_id"]]
        argv.append("--json")
        envelope = self._runner(argv) if self._runner else self._exec(argv)
        if not envelope.get("ok"):
            err = envelope.get("error") or {}
            raise ScalableCLIError(err.get("code", "unknown_error"), err.get("message", str(envelope)),
                                   envelope.get("command", " ".join(args)))
        return envelope.get("data") or {}

    def _exec(self, argv: list[str]) -> dict:
        binary = self.credentials.get("sc_path", "sc")
        try:
            proc = subprocess.run([binary, *argv], capture_output=True, text=True, timeout=120)
        except FileNotFoundError:
            raise ScalableCLIError("sc_not_installed", "install the Scalable CLI: "
                                   "https://github.com/ScalableCapital/scalable-cli/releases") from None
        try:
            return json.loads(proc.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError):
            raise ScalableCLIError(f"exit_{proc.returncode}", (proc.stderr or proc.stdout).strip()[:500]) from None

    @staticmethod
    def _result(data: dict) -> dict:
        return data.get("result", data)

    # -- symbol mapping ------------------------------------------------------------------------------
    def isin_for(self, symbol: str) -> str:
        sym = symbol.upper()
        if ISIN_RE.match(sym):
            return sym
        if sym not in self._isin_by_symbol:
            items = self._result(self.run("broker", "search", symbol)).get("items") or []
            if not items:
                raise ScalableCLIError("security_not_found", f"no Scalable security matches {symbol!r}")
            self._isin_by_symbol[sym] = items[0]["isin"]
        return self._isin_by_symbol[sym]

    def symbol_for(self, isin: str) -> str:
        if isin not in self._symbol_by_isin:
            self._symbol_by_isin[isin] = self._resolve_symbol(isin) or isin
            self._isin_by_symbol.setdefault(self._symbol_by_isin[isin].upper(), isin)
        return self._symbol_by_isin[isin]

    # -- read side -----------------------------------------------------------------------------------
    def get_positions(self) -> list[Position]:
        if not self.available and self.credentials.get("holdings_csv"):
            return super().get_positions()
        result = self._result(self.run("broker", "holdings"))
        out = []
        for item in result.get("items", []):
            qty = _num(item.get("quantity")) or 0.0
            if qty > 0:
                out.append(Position(self.symbol_for(item["isin"]), qty, _num(item.get("fifo_price")), AssetClass.STOCK))
        for coin in result.get("crypto_items", []):
            qty = _num((coin.get("position") or {}).get("filled")) or 0.0
            if qty > 0:
                out.append(Position(coin["ticker"], qty, None, AssetClass.CRYPTO))
        return out

    def get_balance(self) -> dict[str, Balance]:
        balances = {p.symbol: Balance(p.symbol, p.quantity) for p in self.get_positions()}
        if self.available:
            cash = self._result(self.run("broker", "cash-breakdown"))
            if cash.get("cash_balance") is not None:
                balances["EUR"] = Balance("EUR", _num(cash["cash_balance"]))
        return balances

    def get_ticker(self, symbol: str) -> Ticker:
        q = self._result(self.run("broker", "quote", "--isin", self.isin_for(symbol)))
        return Ticker(symbol, _num(q.get("quote_mid_price")), _num(q.get("quote_bid_price")), _num(q.get("quote_ask_price")))

    # -- orders (two-step) ---------------------------------------------------------------------------
    def _trade_args(self, isin: str, side: Side, amount: float, type: OrderType, price: float | None) -> list[str]:
        args = ["broker", "trade", side.value, "--isin", isin]
        if side is Side.BUY and amount != int(amount):
            # buy-side share input is whole shares only; size fractional buys by EUR amount instead
            ref = price or self.get_ticker(isin).last
            args += ["--amount", f"{amount * ref:.2f}"]
        else:
            args += ["--shares", f"{amount:g}"]
        args += ["--order-type", type.value]
        if type is OrderType.LIMIT:
            args += ["--limit-price", f"{price:g}"]
        if self.credentials.get("venue"):
            args += ["--venue", self.credentials["venue"]]
        return args

    def place_order(self, symbol, side, amount, type=OrderType.MARKET, price=None) -> Order:
        """Phase 1 only: request Scalable's binding preview and park it as an intent."""
        side, type = Side(side), OrderType(type)
        if type is OrderType.LIMIT and price is None:
            raise ValueError("limit orders need a price")
        isin = self.isin_for(symbol)
        args = self._trade_args(isin, side, amount, type, price)
        data = self.run(*args)
        confirmation = data.get("confirmation") or {}
        if not confirmation.get("id"):
            raise ScalableCLIError("preview_without_confirmation", "phase 1 returned no confirmation id", "broker trade")
        intent = OrderIntent(f"{self.name}-{confirmation['id']}", self.name, symbol, side, amount, type, price, details={
            "isin": isin,
            "args": args,
            "confirmation_id": confirmation["id"],
            "expires_at_epoch": confirmation.get("expires_at_epoch"),
            "requires_appropriateness_ack": bool(confirmation.get("requires_appropriateness_warning_acknowledgement")),
            "presentation": data.get("presentation"),
            "raw": data,
        })
        self.pending[intent.id] = intent
        log.warning("Scalable order preview ready, awaiting user confirmation: %s", intent.describe())
        if self.on_intent:
            self.on_intent(intent)
        return Order(intent.id, symbol, side, type, amount, price, OrderStatus.PENDING_MANUAL, timestamp=intent.created)

    def confirm(self, intent_id: str, acknowledge_appropriateness_warning: bool = False) -> Order:
        """Phase 2: submit a previewed order. Call only after explicit user approval of the full preview."""
        intent = self.pending.get(intent_id)
        if intent is None:
            raise KeyError(f"no pending Scalable intent {intent_id!r}")
        d = intent.details
        if d.get("expires_at_epoch") and time.time() > d["expires_at_epoch"]:
            self.pending.pop(intent_id)
            raise ScalableCLIError("confirmation_expired", "preview expired (15 min); place the order again")
        if d["requires_appropriateness_ack"] and not acknowledge_appropriateness_warning:
            raise ScalableCLIError("appropriateness_ack_required",
                                   "this order requires acknowledging the appropriateness warning")
        args = [*d["args"], "--confirm", d["confirmation_id"]]
        if acknowledge_appropriateness_warning:
            args.append("--acknowledge-appropriateness-warning")
        data = self.run(*args)
        # the confirmation is consumed even if submission failed; never resubmit blindly
        self.pending.pop(intent_id, None)
        submission = self._result(data).get("order_submission") or {}
        if not submission.get("submitted"):
            raise ScalableCLIError("not_submitted", f"order was not submitted: {submission}", "broker trade")
        return Order(str(submission["order_id"]), intent.symbol, intent.side, intent.type, intent.amount, intent.price,
                     OrderStatus.OPEN, raw=data)

    def cancel_order(self, order_id: str, symbol: str | None = None) -> bool:
        if order_id in self.pending:  # an unconfirmed preview is not an order yet
            return super().cancel_order(order_id, symbol)
        self.run("broker", "trade", "cancel", "--order-id", order_id)
        return True
