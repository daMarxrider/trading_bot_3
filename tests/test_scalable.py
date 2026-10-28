# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Scalable Capital adapter against a fake `sc` CLI (envelopes mirror scalable-cli 1.2.0 JSON output)."""

import time

import pytest

from tradebot.brokers import BROKERS
from tradebot.brokers.scalable import ScalableCapitalBroker, ScalableCLIError, format_preview
from tradebot.core.models import AssetClass, ExecutionMode, OrderStatus, OrderType

ISIN = "US0378331005"


def ok(command, result, **extra):
    return {"ok": True, "command": command, "data": {"account_id": "a1", "portfolio_id": "p1", "result": result, **extra}}


class FakeSC:
    def __init__(self):
        self.calls = []
        self.confirm_expires = time.time() + 900
        self.ack_required = False

    def __call__(self, argv):
        self.calls.append(argv)
        cmd = argv[:3]
        if cmd[:2] == ["broker", "holdings"]:
            return ok("broker.holdings", {
                "items": [
                    {"isin": ISIN, "name": "Apple Inc.", "quantity": "3", "fifo_price": "150.5",
                     "quote_mid_price": "180.1", "quote_currency": "EUR"},
                    {"isin": "IE00B4L5Y983", "name": "iShares Core MSCI World", "quantity": 0, "fifo_price": None},
                ],
                "crypto_items": [{"kind": "crypto", "ticker": "BTC", "name": "Bitcoin",
                                  "position": {"filled": "0.01", "sellable": "0.01", "pending": "0"}}],
            })
        if cmd[:2] == ["broker", "cash-breakdown"]:
            return ok("broker.cash-breakdown", {"cash_balance": "1234.56", "buying_power": "2000",
                                                "buying_power_without_credit": "1200"})
        if cmd[:2] == ["broker", "quote"]:
            return ok("broker.quote", {"isin": ISIN, "quote_mid_price": "180", "quote_bid_price": "179.9",
                                       "quote_ask_price": "180.1"})
        if cmd[:2] == ["broker", "search"]:
            return ok("broker.search", {"items": [{"isin": ISIN, "name": "Apple Inc."}]} if argv[2] == "AAPL" else {"items": []})
        if cmd == ["broker", "trade", "cancel"]:
            return ok("broker.trade.cancel", {"order_id": argv[4], "cancelled": True})
        if cmd[:2] == ["broker", "trade"]:
            if "--confirm" in argv:
                return ok("broker.trade." + argv[2], {"order_submission": {"submitted": True, "order_id": "ord-77"}},
                          confirmation={"id": argv[argv.index("--confirm") + 1], "consumed": True},
                          next_step="completed")
            return ok("broker.trade." + argv[2], {"intent": {"side": argv[2], "isin": ISIN},
                                                   "order_submission": {"submitted": False, "reason": "phase_1_preview_only"}},
                      confirmation={"id": "scb1_abc", "expires_at_epoch": self.confirm_expires,
                                    "requires_appropriateness_warning_acknowledgement": self.ack_required},
                      presentation={"section_order": ["order", "costs"], "sections": {
                          "order": {"title": "Order", "fields": [{"label": "Side", "value": argv[2]},
                                                                 {"label": "Limit price", "value": None}]},
                          "costs": {"title": "Ex-ante costs", "fields": [{"label": "Total costs", "value": "0.99"}]}}},
                      next_step="confirm_with_id")
        return {"ok": False, "command": " ".join(argv), "error": {"code": "unsupported", "message": "fake"}}


@pytest.fixture
def sc():
    return FakeSC()


@pytest.fixture
def broker(sc):
    return ScalableCapitalBroker({"portfolio_id": "p1"}, runner=sc, symbol_resolver={ISIN: "AAPL"}.get)


def test_registered_semi_auto():
    assert BROKERS.get("scalable") is ScalableCapitalBroker
    assert ScalableCapitalBroker.execution_mode is ExecutionMode.SEMI_AUTO


def test_positions_and_balance(broker, sc):
    pos = broker.get_positions()
    assert [(p.symbol, p.quantity, p.avg_cost, p.asset_class) for p in pos] == [
        ("AAPL", 3.0, 150.5, AssetClass.STOCK), ("BTC", 0.01, None, AssetClass.CRYPTO)]
    assert sc.calls[0] == ["broker", "holdings", "--portfolio-id", "p1", "--json"]
    bal = broker.get_balance()
    assert bal["EUR"].free == pytest.approx(1234.56) and bal["AAPL"].total == 3


def test_ticker_and_symbol_mapping(broker):
    t = broker.get_ticker("AAPL")
    assert (t.last, t.bid, t.ask) == (180.0, 179.9, 180.1)
    assert broker.isin_for(ISIN) == ISIN
    with pytest.raises(ScalableCLIError, match="security_not_found"):
        broker.isin_for("NOPE")


def test_place_order_only_previews(broker, sc):
    seen = []
    broker.on_intent = seen.append
    order = broker.place_order("AAPL", "buy", 2)
    assert order.status is OrderStatus.PENDING_MANUAL
    assert not any("--confirm" in c for c in sc.calls), "phase 2 must never run automatically"
    assert sc.calls[-1][:9] == ["broker", "trade", "buy", "--isin", ISIN, "--shares", "2", "--order-type", "market"]
    intent = seen[0]
    assert intent.details["confirmation_id"] == "scb1_abc"
    text = format_preview(intent.details)
    assert "Side: buy" in text and "Limit price: null" in text and "Total costs: 0.99" in text


def test_fractional_buy_is_sized_in_eur(broker, sc):
    broker.place_order("AAPL", "buy", 0.5)
    assert "--amount" in sc.calls[-1] and sc.calls[-1][sc.calls[-1].index("--amount") + 1] == "90.00"


def test_limit_sell(broker, sc):
    broker.place_order("AAPL", "sell", 1.5, OrderType.LIMIT, 200)
    argv = sc.calls[-1]
    assert argv[argv.index("--shares") + 1] == "1.5" and argv[argv.index("--limit-price") + 1] == "200"


def test_confirm_submits_exact_preview_args(broker, sc):
    order = broker.place_order("AAPL", "buy", 2)
    preview_args = sc.calls[-1][:-3]  # strip --portfolio-id p1 --json
    placed = broker.confirm(order.id)
    phase2 = sc.calls[-1]
    assert phase2[: len(preview_args)] == preview_args
    assert phase2[phase2.index("--confirm") + 1] == "scb1_abc"
    assert placed.id == "ord-77" and placed.status is OrderStatus.OPEN
    assert order.id not in broker.pending
    with pytest.raises(KeyError):
        broker.confirm(order.id)  # a confirmation id is single-use


def test_confirm_refuses_expired_and_unacknowledged(broker, sc):
    sc.confirm_expires = time.time() - 1
    expired = broker.place_order("AAPL", "buy", 1)
    with pytest.raises(ScalableCLIError, match="expired"):
        broker.confirm(expired.id)
    sc.confirm_expires, sc.ack_required = time.time() + 900, True
    risky = broker.place_order("AAPL", "buy", 1)
    with pytest.raises(ScalableCLIError, match="appropriateness"):
        broker.confirm(risky.id)
    broker.confirm(risky.id, acknowledge_appropriateness_warning=True)
    assert "--acknowledge-appropriateness-warning" in sc.calls[-1]


def test_cancel_preview_vs_order(broker, sc):
    order = broker.place_order("AAPL", "buy", 1)
    n = len(sc.calls)
    assert broker.cancel_order(order.id) and len(sc.calls) == n  # preview: local only
    assert broker.cancel_order("ord-77")
    assert sc.calls[-1][:5] == ["broker", "trade", "cancel", "--order-id", "ord-77"]


def test_cli_error_envelope_raises(broker):
    broker._runner = lambda argv: {"ok": False, "command": "broker.holdings",
                                   "error": {"code": "login_required", "message": "session expired"}}
    with pytest.raises(ScalableCLIError, match="sc login"):
        broker.get_positions()


def test_csv_fallback_without_sc(tmp_path):
    csv = tmp_path / "h.csv"
    csv.write_text("symbol,quantity,avg_cost\nAAPL,2,100\n")
    b = ScalableCapitalBroker({"holdings_csv": str(csv), "sc_path": "/nonexistent/sc"})
    assert not b.available
    assert [(p.symbol, p.quantity) for p in b.get_positions()] == [("AAPL", 2.0)]


@pytest.mark.parametrize("answer, submitted", [("y\n", True), ("n\n", False), ("\n", False)])
def test_cli_trade_requires_explicit_confirmation(monkeypatch, sc, answer, submitted):
    from typer.testing import CliRunner

    from tradebot.cli import main as cli

    def fake_create(name, **kwargs):
        return ScalableCapitalBroker({}, runner=sc, symbol_resolver={ISIN: "AAPL"}.get, **kwargs)

    monkeypatch.setattr(cli, "create_broker", fake_create)
    res = CliRunner().invoke(cli.app, ["trade", "buy", "AAPL", "2"], input=answer)
    assert res.exit_code == 0, res.output
    assert "Total costs: 0.99" in res.output  # full preview shown before the question
    assert any("--confirm" in c for c in sc.calls) is submitted
    assert ("ord-77" in res.output) is submitted
