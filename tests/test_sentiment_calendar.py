# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""News sentiment and the economic calendar (#167). Offline: the news fetch and the calendar feed are fakes."""

import os
import time
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest

from tradebot import economic, sentiment
from tradebot.backtest import backtest
from tradebot.data.market import MarketData, MarketDataError
from tradebot.indicators import INDICATORS
from tradebot.news import NewsItem
from tests.conftest import make_ohlc


def item(title, when, summary=""):
    return NewsItem("BTC/USDT", title, "pub", "http://x/" + title, when, summary)


# -- sentiment ---------------------------------------------------------------------------------------
def test_scores_have_the_right_sign_and_range():
    assert sentiment.score_text("Bitcoin surges to a record high as ETF inflows rise") > 0.5
    assert sentiment.score_text("Exchange hacked, users lose funds in a crash") < -0.5
    assert sentiment.score_text("Company holds its annual meeting") == 0.0
    assert -1 < sentiment.score_text("collapse " * 20) < 0


def test_negation_and_intensifier():
    assert sentiment.score_text("profits rise") > 0
    assert sentiment.score_text("profits do not rise") < sentiment.score_text("profits rise")
    assert sentiment.score_text("shares sharply fall") < sentiment.score_text("shares fall")


def test_labels_and_summary():
    assert (sentiment.label(0.5), sentiment.label(-0.5), sentiment.label(0.05)) == ("positive", "negative", "neutral")
    assert sentiment.arrow(0.5) == "▲" and sentiment.arrow(-0.5) == "▼"
    s = sentiment.summarize([0.5, -0.5, 0.0, 0.9])
    assert (s.count, s.positive, s.negative, s.neutral) == (4, 2, 1, 1)
    assert sentiment.summarize([]).mean == 0.0


def test_hourly_and_daily_buckets():
    t = datetime(2026, 1, 5, 10, 20, tzinfo=timezone.utc)
    items = [item("stocks surge", t), item("stocks crash", t.replace(minute=40)),
             item("coin rally", t.replace(hour=11)), item("coin slump", t.replace(day=6))]
    h = sentiment.hourly(items)
    assert list(h.index) == [pd.Timestamp("2026-01-05 10:00", tz="UTC"), pd.Timestamp("2026-01-05 11:00", tz="UTC"),
                             pd.Timestamp("2026-01-06 10:00", tz="UTC")]
    d = sentiment.daily(items)
    assert list(d["count"]) == [3, 1]
    assert sentiment.hourly([]).empty and sentiment.daily([]).empty


def test_history_drops_the_possibly_cut_first_hour():
    t = datetime(2026, 1, 5, 10, 20, tzinfo=timezone.utc)
    s = sentiment.history("BTC/USDT", lambda syms, limit: [item("a surge", t), item("b surge", t.replace(hour=12))])
    assert len(s) == 1 and s.index[0].hour == 12


# -- sentiment as market data ------------------------------------------------------------------------
def feed(symbols, limit=50):
    now = pd.Timestamp.now(tz="UTC").floor("h")
    return [item(f"coin surge {i}", (now - pd.Timedelta(hours=i)).to_pydatetime()) for i in range(1, 6)]


def test_news_sentiment_is_attached_to_any_symbol_without_look_ahead(tmp_path):
    now = pd.Timestamp.now(tz="UTC").floor("h")
    df = make_ohlc(48, freq="h")
    df.index = pd.date_range(now - pd.Timedelta(hours=47), periods=48, freq="h")
    out = MarketData(cache=tmp_path, news=feed).attach(df, "AAPL", ["news_sentiment"])
    assert "news_sentiment" in out.columns and out.attrs["market_errors"] == {}
    col = out["news_sentiment"]
    assert col.iloc[-1] > 0
    # an hour's score counts from the end of that hour: the oldest kept hour (now-4h) is known from now-3h
    assert col[col.index < now - pd.Timedelta(hours=3)].isna().all()
    assert col[col.index >= now - pd.Timedelta(hours=3)].notna().all()
    stock = MarketData(cache=tmp_path, news=feed).attach(df, "AAPL")  # not asked for by name: not attached
    assert "news_sentiment" not in stock.columns and "fear_greed" in stock.attrs["market_errors"]


def test_no_headlines_is_an_error_not_a_crash(tmp_path):
    df = make_ohlc(10, freq="h")
    empty = lambda s, n: []
    out = MarketData(cache=tmp_path, news=empty).attach(df, "AAPL", ["news_sentiment"])
    assert "news_sentiment" not in out.columns and "no recent headlines" in out.attrs["market_errors"]["news_sentiment"]
    with pytest.raises(MarketDataError):
        MarketData(cache=tmp_path, news=empty).series("news_sentiment", "AAPL", df.index[0], df.index[-1])


def test_the_cache_accumulates_hours(tmp_path):
    now = pd.Timestamp.now(tz="UTC").floor("h")
    first = lambda s, n: [item("a surge", (now - pd.Timedelta(hours=h)).to_pydatetime()) for h in (30, 29, 28)]
    later = lambda s, n: [item("b crash", (now - pd.Timedelta(hours=h)).to_pydatetime()) for h in (3, 2, 1)]
    a = MarketData(cache=tmp_path, news=first).series("news_sentiment", "AAPL", now - pd.Timedelta(days=3), now)[0]
    for p in (tmp_path / "market").rglob("*.npz"):  # make the cache look old
        os.utime(p, (time.time() - 4000, time.time() - 4000))
    b = MarketData(cache=tmp_path, news=later).series("news_sentiment", "AAPL", now - pd.Timedelta(days=3), now)[0]
    assert len(b) > len(a) and b.iloc[0] > 0 > b.iloc[-1]


def test_indicator_reads_the_column():
    cls = INDICATORS.get("news_sentiment")
    assert cls.market_data == ("news_sentiment",)
    df = make_ohlc(20)
    assert np.isnan(cls()(df)["news_sentiment_value"]).all()
    df["news_sentiment"] = 0.4
    assert cls()(df)["news_sentiment_value"].iloc[-1] == pytest.approx(0.4)


# -- calendar ----------------------------------------------------------------------------------------
RAW = [
    {"title": "CPI m/m", "country": "USD", "date": "2026-11-12T08:30:00-05:00", "impact": "High",
     "forecast": "0.3%", "previous": "0.2%"},
    {"title": "Bank Holiday", "country": "USD", "date": "2026-11-26T00:00:00-05:00", "impact": "Holiday"},
    {"title": "Retail Sales", "country": "EUR", "date": "2026-11-13T05:00:00-05:00", "impact": "Low"},
    {"nonsense": 1},
]


def test_parse_feed_and_classify():
    ev = economic.parse_feed(RAW)
    assert [e.title for e in ev] == ["CPI m/m", "Retail Sales"]
    assert ev[0].time == datetime(2026, 11, 12, 13, 30, tzinfo=timezone.utc)
    assert (ev[0].kind, ev[0].impact, ev[0].forecast) == ("cpi", "high", "0.3%")
    assert economic.parse_feed("garbage") == []
    assert economic.classify("FOMC Statement") == "rates" and economic.classify("Non-Farm Employment Change") == "jobs"
    assert economic.classify("Retail Sales m/m") == "other"


def test_jobs_report_day_rule():
    assert economic.jobs_report_day(2026, 11) == date(2026, 12, 4)
    assert economic.jobs_report_day(2025, 9) == date(2025, 10, 3)
    assert all(economic.jobs_report_day(2026, m).weekday() == 4 for m in range(1, 13))


def test_load_merges_schedule_feed_and_cache():
    ev = economic.load("2026-12-01", "2026-12-31", fetch=lambda: [])
    assert [(e.kind, e.time.day) for e in ev] == [("jobs", 4), ("rates", 9)]
    assert all(e.approx for e in ev) and ev[1].time.hour == 19  # 14:00 New York in winter
    feed_ = lambda: economic.parse_feed(RAW)
    got = economic.load("2026-11-01", "2026-11-30", fetch=feed_, refresh=True, min_impact="high")
    assert [e.title for e in got if e.kind == "cpi"] == ["CPI m/m"] and all(e.impact == "high" for e in got)

    def boom():
        raise ConnectionError("offline")

    again = economic.load("2026-11-01", "2026-11-30", fetch=boom, refresh=True)  # the cache keeps the week
    assert any(e.title == "CPI m/m" for e in again)


def test_feed_replaces_the_schedule_for_the_same_day():
    real = economic.EconEvent(datetime(2026, 12, 4, 13, 30, tzinfo=timezone.utc), "Non-Farm Employment Change", "USD",
                              "high", "jobs", "150K", "140K")
    ev = economic.load("2026-12-04", "2026-12-04 23:00", fetch=lambda: [real], refresh=True)
    assert ev == [real]


def test_flat_mask_covers_candles_touching_the_window():
    idx = pd.date_range("2026-01-01", periods=48, freq="h", tz="UTC")
    e = economic.EconEvent(datetime(2026, 1, 1, 12, 30, tzinfo=timezone.utc), "CPI", "USD")
    m = economic.flat_mask(idx, [e], 2)
    assert list(idx[m.to_numpy()]) == list(pd.date_range("2026-01-01 10:00", "2026-01-01 14:00", freq="h", tz="UTC"))
    assert not economic.flat_mask(idx, [e], 0).any() and not economic.flat_mask(idx, [], 2).any()


def test_apply_flat_closes_and_blocks():
    pos = pd.Series([0, 1, 1, 1, 1, 0, 1.0])
    flat = pd.Series([False, False, True, True, False, False, False])
    assert list(economic.apply_flat(pos, flat)) == [0, 1, 0, 0, 1, 0, 1]


class Hold:
    name = "hold"
    values: dict = {}
    allow_short = False

    def run_fills(self, df):
        return pd.Series(1.0, index=df.index), None


def test_backtest_flat_around_events_closes_and_reopens():
    df = make_ohlc(200, freq="h")
    e = economic.EconEvent(df.index[100].to_pydatetime(), "CPI", "USD")
    flat = economic.flat_mask(df.index, [e], 3)
    base, cut = backtest(df, Hold()), backtest(df, Hold(), flat=flat)
    assert len(cut.trades) == len(base.trades) + 1  # out before the event, back in after it


def test_event_flat_uses_events_and_off_switch():
    df = make_ohlc(100, freq="h")
    assert economic.event_flat(df, 0) is None
    ev = economic.EconEvent(df.index[50].to_pydatetime(), "CPI", "USD")
    mask = economic.event_flat(df, 2, fetch=lambda: [ev], impact="low")
    assert mask is not None and mask.iloc[48:53].all() and not mask.iloc[:40].any()
