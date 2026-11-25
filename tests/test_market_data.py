# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Crypto market data (#166): sources, alignment without look-ahead, caching, failures, indicators, rules, order book.
Nothing here touches the network: exchanges and HTTP are fakes."""

import numpy as np
import pandas as pd
import pytest

from tradebot.algorithms import build_algorithm
from tradebot.core.series import OHLC
from tradebot.data import market
from tradebot.data.market import MarketData, MarketDataError, align, fetch_order_book
from tradebot.indicators import INDICATORS
from tests.conftest import make_ohlc

T0 = pd.Timestamp("2024-01-01", tz="UTC")


class FakeExchange:
    """A ccxt-like perpetuals exchange; counts calls so caching can be checked."""

    id = "fake"

    def __init__(self, fail=False):
        self.has = {"fetchFundingRateHistory": True, "fetchOpenInterestHistory": True}
        self.fail, self.calls = fail, []

    def fetch_funding_rate_history(self, symbol, since=None, limit=None, params=None):
        self.calls.append(("funding", symbol, since))
        if self.fail:
            raise ConnectionError("offline")
        ts = [int((T0 + pd.Timedelta(hours=8 * i)).timestamp() * 1000) for i in range(60)]
        return [{"timestamp": t, "fundingRate": 0.0001 * (i % 5 - 1)} for i, t in enumerate(ts) if t >= since][:limit]

    def fetch_open_interest_history(self, symbol, timeframe, since=None, limit=None, params=None):
        self.calls.append(("oi", symbol, timeframe))
        if self.fail:
            raise ConnectionError("offline")
        ts = [int((T0 + pd.Timedelta(hours=i)).timestamp() * 1000) for i in range(200)]
        return [{"timestamp": t, "openInterestAmount": 1.0, "openInterestValue": 1000.0 + i} for i, t in enumerate(ts)
                if t >= since][:limit]

    def fetch_order_book(self, symbol, limit=None):
        if self.fail:
            raise ConnectionError("offline")
        return {"timestamp": 1_700_000_000_000, "bids": [[99.0, 2.0], [98.0, 3.0], [90.0, 50.0]],
                "asks": [[101.0, 1.0], [102.0, 4.0], [120.0, 40.0]]}


def fake_json(url):
    if "alternative.me" in url:
        return {"data": [{"value": str(v), "timestamp": str(int((T0 + pd.Timedelta(days=i)).timestamp()))}
                         for i, v in reversed(list(enumerate([10, 20, 30, 40, 50, 60, 70, 80, 90, 95])))]}
    assert "coinmetrics" in url and "assets=btc" in url
    metrics = [m for m in ("CapMVRVCur", "AdrActCnt", "FlowInExUSD", "FlowOutExUSD") if m in url]
    return {"data": [{"asset": "btc", "time": (T0 + pd.Timedelta(days=i)).strftime("%Y-%m-%dT00:00:00.000000000Z"),
                      **{m: str((1.0 + i) * (0.5 if m == "FlowOutExUSD" else 1)) for m in metrics}}
                     for i in range(10)]}


def frame(n=40, freq="D", start=T0):
    df = make_ohlc(n, freq=freq)
    df.index = pd.date_range(start, periods=n, freq=freq)
    return df


def md(tmp_path=None, **kw):
    kw.setdefault("exchange", FakeExchange())
    kw.setdefault("get_json", fake_json)
    kw.setdefault("exchange_id", "fake")
    return MarketData(cache=tmp_path, **kw)


# -- alignment --------------------------------------------------------------------------------------
def test_align_sees_only_what_was_known_at_the_candle_open():
    s = pd.Series([1.0, 2.0], index=pd.DatetimeIndex([T0 + pd.Timedelta(hours=8), T0 + pd.Timedelta(hours=16)]))
    idx = pd.date_range(T0, periods=4, freq="8h")  # 00:00, 08:00, 16:00, 24:00
    assert np.allclose(align(s, idx), [np.nan, 1.0, 2.0, 2.0], equal_nan=True)  # a value stamped at the open counts
    assert np.allclose(align(s, idx, lag=pd.Timedelta(hours=8)), [np.nan, np.nan, 1.0, 2.0], equal_nan=True)


def test_align_drops_stale_values_and_handles_naive_and_empty():
    s = pd.Series([5.0], index=pd.DatetimeIndex([T0]))
    idx = pd.date_range(T0.tz_localize(None), periods=5, freq="D")  # naive = UTC
    out = align(s, idx, max_age=pd.Timedelta(days=2))
    assert out[:3].tolist() == [5.0, 5.0, 5.0] and np.isnan(out[3:]).all()
    assert np.isnan(align(s.iloc[:0], idx)).all()


def test_align_never_changes_when_later_values_arrive():
    full = pd.Series(np.arange(1.0, 21.0), index=pd.date_range(T0, periods=20, freq="8h"))
    idx = pd.date_range(T0, periods=15, freq="D")
    a = align(full, idx, max_age=pd.Timedelta(days=2))
    b = align(full.iloc[:10], idx, max_age=pd.Timedelta(days=2))
    known = idx <= full.index[9]
    assert np.allclose(a[known], b[known], equal_nan=True)


# -- attaching ---------------------------------------------------------------------------------------
def test_attach_adds_every_column_lined_up(tmp_path):
    df = frame(8)
    out = md(tmp_path).attach(df, "BTC/USDT")
    assert {"funding_rate", "open_interest", "fear_greed", "mvrv", "active_addresses"} <= set(out.columns)
    assert out.attrs["market_errors"] == {} and "funding_rate" in out.attrs["market_data"]
    assert out.index.equals(df.index) and "funding_rate" not in df.columns  # the original is untouched
    # fear & greed of day i is stamped 00:00 of that day: known at that candle's open
    assert out["fear_greed"].iloc[:4].tolist() == [10.0, 20.0, 30.0, 40.0]
    # on-chain values of a day are known only once the day is over
    assert np.isnan(out["mvrv"].iloc[0]) and out["mvrv"].iloc[1] == 1.0 and out["mvrv"].iloc[2] == 2.0


def test_open_interest_is_known_when_its_bucket_ends(tmp_path):
    out = md(tmp_path).attach(frame(48, "h"), "BTC/USDT", ["open_interest"])
    assert np.isnan(out["open_interest"].iloc[0]) and out["open_interest"].iloc[1] == 1000.0  # bucket 0 ended at 01:00
    assert out["open_interest"].iloc[10] == 1009.0


def test_funding_rate_per_settlement(tmp_path):
    out = md(tmp_path).attach(frame(10, "8h"), "BTC/USDT", ["funding_rate"])
    assert np.allclose(out["funding_rate"].iloc[:5], [-0.0001, 0.0, 0.0001, 0.0002, 0.0003])


def test_stocks_and_pairs_get_no_market_data_and_say_so():
    out = md().attach(frame(5), "AAPL")
    assert "funding_rate" not in out.columns and "not a crypto pair" in out.attrs["market_errors"]["funding_rate"]
    assert "funding_rate" not in md().attach(frame(5), "BTC/USDT:ETH/USDT").columns


def test_failures_leave_the_column_out_without_raising():
    def limited(url):
        raise OSError("rate limited")

    out = md(exchange=FakeExchange(fail=True), get_json=limited).attach(frame(5), "BTC/USDT")
    assert not {"funding_rate", "open_interest", "fear_greed", "mvrv"} & set(out.columns)
    assert set(out.attrs["market_errors"]) == set(market.SOURCES)
    assert "offline" in out.attrs["market_errors"]["funding_rate"]


def test_unknown_coin_on_chain_is_reported():
    def get(url):
        return {"error": {"type": "bad_parameter", "message": "unsupported asset zzz"}}

    out = md(get_json=get).attach(frame(5), "ZZZ/USDT", ["mvrv"])
    assert "unsupported asset" in out.attrs["market_errors"]["mvrv"]


# -- caching -----------------------------------------------------------------------------------------
def test_cache_serves_fresh_data_without_the_network_and_offline_afterwards(tmp_path, monkeypatch):
    ex = FakeExchange()
    md(tmp_path, exchange=ex).attach(frame(10, "8h"), "BTC/USDT", ["funding_rate"])
    n = len(ex.calls)
    again = md(tmp_path, exchange=ex).attach(frame(10, "8h"), "BTC/USDT", ["funding_rate"])
    assert len(ex.calls) == n and again["funding_rate"].notna().any()  # fresh: no request
    monkeypatch.setattr(market, "FRESH_FOR", 0.0)
    off = md(tmp_path, exchange=FakeExchange(fail=True)).attach(frame(10, "8h"), "BTC/USDT", ["funding_rate"])
    assert off.attrs.get("offline") is True
    assert np.allclose(off["funding_rate"], again["funding_rate"], equal_nan=True)


def test_cache_accumulates_history_and_only_fetches_the_tail(tmp_path, monkeypatch):
    monkeypatch.setattr(market, "FRESH_FOR", 0.0)
    ex = FakeExchange()
    md(tmp_path, exchange=ex).attach(frame(6, "8h"), "BTC/USDT", ["funding_rate"])
    first = ex.calls[0][2]
    md(tmp_path, exchange=ex).attach(frame(12, "8h"), "BTC/USDT", ["funding_rate"])
    assert ex.calls[-1][2] > first  # the second request started after what was cached


def test_fear_greed_is_one_series_for_every_symbol(tmp_path):
    seen = []

    def get(url):
        seen.append(url)
        return fake_json(url)

    md(tmp_path, get_json=get).attach(frame(5), "BTC/USDT", ["fear_greed"])
    md(tmp_path, get_json=get).attach(frame(5), "ETH/USDT", ["fear_greed"])
    assert len(seen) == 1


def test_no_disk_cache_still_works():
    out = MarketData(cache=None, exchange=FakeExchange(), get_json=fake_json).attach(frame(5), "BTC/USDT",
                                                                                    ["fear_greed"])
    assert out["fear_greed"].notna().all()


def test_open_interest_bucket_follows_the_candle():
    assert market.oi_timeframe(pd.Timedelta(minutes=1)) == "5m"
    assert market.oi_timeframe(pd.Timedelta(hours=1)) == "1h"
    assert market.oi_timeframe(pd.Timedelta(hours=3)) == "2h"
    assert market.oi_timeframe(pd.Timedelta(days=7)) == "1d"


# -- indicators --------------------------------------------------------------------------------------
def test_indicators_are_registered_and_flagged():
    for name in ("funding", "open_interest", "oi_change", "fear_greed", "mvrv", "active_addresses", "exchange_netflow"):
        cls = INDICATORS.get(name)
        assert cls.market_data and not cls.overlay
    from tradebot.indicators.base import is_screenable

    ok, why = is_screenable(INDICATORS.get("funding"))
    assert not ok and "market data" in why
    with pytest.raises(ValueError, match="own timeframe"):
        INDICATORS.get("funding")(tf="1d")


def test_indicators_read_the_attached_columns_and_are_nan_without():
    df = md().attach(frame(40), "BTC/USDT", ["fear_greed"])
    fg = INDICATORS.get("fear_greed")(smooth=2)(df)["fear_greed_value"]
    assert fg.iloc[10] == pytest.approx(95.0) and fg.iloc[0] != fg.iloc[0]
    assert INDICATORS.get("funding")()(frame(10))["funding_value"].isna().all()
    ch = INDICATORS.get("oi_change")(period=1)(md().attach(frame(48, "h"), "BTC/USDT", ["open_interest"]))
    assert ch["oi_change_value"].iloc[5] == pytest.approx((1004 / 1003 - 1) * 100)


def test_funding_in_percent():
    df = frame(10, "8h")
    df["funding_rate"] = 0.0003
    assert INDICATORS.get("funding")()(df)["funding_value"].iloc[-1] == pytest.approx(0.03)


def test_ohlc_carries_data_through_slicing_and_without():
    df = frame(12)
    df["fear_greed"] = np.arange(12.0)
    o = OHLC.from_frame(df)
    assert o.has_data("fear_greed") and not o.has_data("mvrv") and np.isnan(o.data("mvrv")).all()
    part = o.apply(lambda a: a[:5])
    assert part.data("fear_greed").tolist() == [0, 1, 2, 3, 4] and len(part) == 5


def test_indicator_values_do_not_change_when_later_candles_arrive():
    df = md().attach(frame(60), "BTC/USDT", ["fear_greed", "mvrv", "open_interest"])
    for name in ("fear_greed", "mvrv", "oi_change"):
        col = f"{name}_value"
        ind = INDICATORS.get(name)()
        assert np.allclose(ind(df)[col].iloc[:30], ind(df.iloc[:30])[col], equal_nan=True)


# -- rules and evolve ---------------------------------------------------------------------------------
def test_combinator_rules_use_market_data():
    df = md().attach(frame(40), "BTC/USDT", ["fear_greed"])
    pos = build_algorithm("combo", entry="fear_greed > 50", exit="fear_greed < 30").run(df)
    assert pos.iloc[:5].eq(0).all() and pos.iloc[6:10].eq(1).all()  # greed above 50 from day 5, acted on next candle
    assert (build_algorithm("combo", entry="fear_greed > 50").run(frame(40)) == 0).all()  # no data: never fires


def test_needed_columns_from_rule_text():
    assert market.needed("close > sma(200) & funding > 0.05") == ["funding_rate"]
    assert sorted(market.needed("oi_change(24) > 5 | fear_greed < 20")) == ["fear_greed", "open_interest"]
    assert market.needed("rsi < 30") == []


def test_evolve_offers_market_data_only_when_asked():
    from tradebot.evolve.primitives import PrimitiveSet, indicator_outputs

    plain = {c.name for c, _, _ in indicator_outputs()}
    assert not plain & {"funding", "fear_greed", "open_interest"}
    assert {c.name for c, _, _ in indicator_outputs(["funding", "fear_greed"])} == {"funding", "fear_greed"}
    prims = PrimitiveSet.default(["fear_greed", "rsi"]).primitives
    assert any(p.indicator and p.indicator[0] == "fear_greed" for p in prims)


# -- order book --------------------------------------------------------------------------------------
def test_order_book_depth_and_imbalance():
    book = fetch_order_book("BTC/USDT", exchange=FakeExchange())
    assert book.mid == 100.0 and book.spread == pytest.approx(0.02)
    bp, bc, ap, ac = book.depth()
    assert bp.tolist() == [90, 98, 99] and bc.tolist() == [55, 5, 2]  # cumulative towards the spread
    assert ap.tolist() == [101, 102, 120] and ac.tolist() == [1, 5, 45]
    bp, bc, ap, ac = book.depth(within=0.05)
    assert bp.tolist() == [98, 99] and ap.tolist() == [101, 102]
    assert book.imbalance(0.05) == pytest.approx(0.0)
    assert book.imbalance(0.015) == pytest.approx((2 - 1) / 3)


def test_order_book_failures_are_market_data_errors():
    with pytest.raises(MarketDataError, match="offline"):
        fetch_order_book("BTC/USDT", exchange=FakeExchange(fail=True))
    ex = FakeExchange()
    ex.fetch_order_book = lambda s, limit=None: {"bids": [], "asks": []}
    with pytest.raises(MarketDataError, match="empty"):
        fetch_order_book("BTC/USDT", exchange=ex)
    with pytest.raises(MarketDataError):
        md(spot=FakeExchange(fail=True)).order_book("BTC/USDT")


# -- CLI ----------------------------------------------------------------------------------------------
def _cli(monkeypatch, *args, rows=60):
    from typer.testing import CliRunner

    import tradebot.cli.main as cli

    monkeypatch.setattr(cli, "load_data", lambda s, tf, st, en, bars=None, market=None: (
        md().attach(frame(rows), s, market) if market else frame(rows)))
    return CliRunner().invoke(cli.app, list(args), env={"COLUMNS": "200"})


def test_cli_backtest_fetches_the_market_data_its_rules_use(monkeypatch):
    res = _cli(monkeypatch, "backtest", "BTC/USDT", "combo", "--entry", "fear_greed > 50", "--exit", "fear_greed < 30",
               "--no-plot")
    assert res.exit_code == 0, res.output
    plain = _cli(monkeypatch, "backtest", "BTC/USDT", "combo", "--entry", "rsi < 30", "--no-plot")
    assert plain.exit_code == 0, plain.output


def test_cli_load_data_attaches_and_reports_what_is_missing(monkeypatch):
    import tradebot.cli.main as cli
    from tradebot.data.engine import MarketDataEngine

    monkeypatch.setattr(MarketDataEngine, "get_ohlc", lambda self, *a, **k: frame(30))
    monkeypatch.setattr(market, "MarketData", lambda: md(exchange=FakeExchange(fail=True)))
    df = cli.load_data("BTC/USDT", "1d", None, None, 30, ["funding_rate", "fear_greed"])
    assert "fear_greed" in df.columns and "funding_rate" not in df.columns
    assert "funding_rate" in df.attrs["market_errors"]


def test_cli_market_command_and_evolve_flag(monkeypatch):
    from typer.testing import CliRunner

    import tradebot.cli.main as cli

    monkeypatch.setattr(market, "MarketData", lambda: md(spot=FakeExchange()))
    out = CliRunner().invoke(cli.app, ["market", "BTC/USDT", "--depth"], env={"COLUMNS": "200"})
    assert out.exit_code == 0, out.output
    for text in ("Funding rate", "Open interest", "Fear & Greed", "MVRV", "order book", "imbalance"):
        assert text in out.output
    assert CliRunner().invoke(cli.app, ["market", "AAPL"]).exit_code != 0
    bad = CliRunner().invoke(cli.app, ["evolve", "BTC/USDT", "--market-data", "--from-scratch"])
    assert bad.exit_code != 0


def test_exchange_netflow_is_inflow_minus_outflow():
    from tradebot.data.market import fetch_onchain
    out = fetch_onchain("exchange_netflow", "BTC/USDT", T0, T0 + pd.Timedelta(days=9), fake_json)
    assert out.iloc[0] == 0.5 and out.iloc[3] == 2.0
