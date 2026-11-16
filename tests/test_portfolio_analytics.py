# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Portfolio analytics (#169), offline."""

import numpy as np
import pandas as pd
import pytest

from tradebot import portfolio_analytics as pa

_ids = iter(range(1, 1000))


def fill(day, sym, side, amount, price, fee=0.0):
    return {"id": next(_ids), "time": f"2025-01-{day:02d}T12:00:00+00:00", "symbol": sym, "side": side,
            "amount": amount, "price": price, "fee": fee, "mode": "paper"}


def closes(values, start="2025-01-01"):
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="D", tz="UTC"), dtype=float)


def test_twr_ignores_deposits():
    # 1 unit bought at 100 on day 1, 1 more at 110 on day 2; price 100,110,121 -> +21% from the first day's price
    fills = [fill(1, "A", "buy", 1, 100), fill(2, "A", "buy", 1, 110)]
    perf = pa.performance(fills, {"A": closes([100, 110, 121])})
    assert perf["twr"].iloc[-1] == pytest.approx(0.21)
    s = perf["summary"]
    assert s["value"] == pytest.approx(242) and s["invested"] == pytest.approx(210)
    assert s["mwr"] == pytest.approx(32 / 210)


def test_twr_after_withdrawal_and_flows():
    fills = [fill(1, "A", "buy", 2, 100), fill(2, "A", "sell", 1, 100, fee=1)]
    perf = pa.performance(fills, {"A": closes([100, 100, 200])})
    assert perf["flows"].iloc[1] == pytest.approx(-99)
    assert perf["twr"].iloc[-1] > 0.9  # the sale does not count as a loss


def test_mwr_annualised_matches_known_rate():
    flows = pd.Series([100.0], index=[pd.Timestamp("2024-01-01", tz="UTC")])
    total, annual = pa.money_weighted_return(flows, 110.0, "2025-01-01")
    assert total == pytest.approx(0.10)
    assert annual == pytest.approx(0.10, abs=0.01)


def test_benchmark_and_excess():
    fills = [fill(1, "A", "buy", 1, 100)]
    bench = closes([50, 55, 60])
    perf = pa.performance(fills, {"A": closes([100, 110, 120])}, bench)
    assert perf["summary"]["benchmark"] == pytest.approx(0.2)
    assert perf["summary"]["excess"] == pytest.approx(0.0)


def test_empty_history():
    assert pa.performance([], {})["summary"] == {}


def test_correlation_and_risk_share():
    rng = np.random.default_rng(1)
    a = rng.normal(0, 0.01, 200)
    cl = {"A": closes(100 * np.cumprod(1 + a)), "B": closes(100 * np.cumprod(1 + a)),
          "C": closes(100 * np.cumprod(1 + rng.normal(0, 0.03, 200)))}
    corr = pa.correlation(cl)
    assert corr.loc["A", "B"] == pytest.approx(1.0)
    assert abs(corr.loc["A", "C"]) < 0.3
    risk = pa.risk_contributions(cl, {"A": 0.3, "B": 0.3, "C": 0.4})
    assert risk["risk_share"].sum() == pytest.approx(1.0)
    assert risk.loc["C", "risk_share"] > risk.loc["C", "weight"]


def test_parse_targets():
    assert pa.parse_targets("AAPL=40, MSFT:60%") == {"AAPL": 0.4, "MSFT": 0.6}
    with pytest.raises(ValueError):
        pa.parse_targets("AAPL")


def test_rebalance_buys_and_sells_net_to_zero():
    df = pa.rebalance({"A": 700.0, "B": 300.0}, {"A": 10.0, "B": 5.0, "C": 20.0}, {"A": 0.4, "B": 0.4, "C": 0.2})
    t = df.set_index("symbol")
    assert t.loc["A", "trade_value"] == pytest.approx(-300) and t.loc["A", "action"] == "sell"
    assert t.loc["A", "quantity"] == pytest.approx(-30)
    assert t.loc["B", "trade_value"] == pytest.approx(100)
    assert t.loc["C", "trade_value"] == pytest.approx(200) and t.loc["C", "quantity"] == pytest.approx(10)
    assert df["trade_value"].sum() == pytest.approx(0)


def test_rebalance_tolerance_and_unlisted_sold():
    df = pa.rebalance({"A": 510.0, "B": 490.0}, {"A": 1, "B": 1}, {"A": 0.5, "B": 0.5}, tolerance=0.02)
    assert set(df["action"]) == {"hold"}
    df = pa.rebalance({"A": 500.0, "X": 500.0}, {"A": 1, "X": 1}, {"A": 1.0})
    assert df.set_index("symbol").loc["X", "action"] == "sell"
