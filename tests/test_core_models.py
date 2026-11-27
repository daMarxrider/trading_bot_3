# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Core data helpers every data path goes through: OHLC normalisation, timeframes, symbol pairs."""

import numpy as np
import pandas as pd
import pytest

from tradebot.core.models import AssetClass, Balance, guess_asset_class, normalize_ohlc, split_pair, to_ohlc_frame
from tradebot.core.timeframes import month_start, resample, to_timedelta


def ohlc(index, **columns):
    """A frame with OHLC columns all equal to ``close`` unless given."""
    close = columns.pop("close")
    data = {"open": close, "high": close, "low": close, "close": close, **columns}
    return pd.DataFrame(data, index=pd.DatetimeIndex(index))


# -- split_pair: edge cases beyond tests/test_pairs.py ---------------------------------------------------
@pytest.mark.parametrize("symbol", ["", ":", ":PEP", "KO:", "BTC/USDT:PEP", "KO:ETH/USDT", "A:B:C"])
def test_malformed_pairs_are_not_pairs(symbol):
    assert split_pair(symbol) is None


def test_a_stock_pair_with_exchange_suffixes_splits():
    assert split_pair("SAP.DE:SIE.DE") == ("SAP.DE", "SIE.DE")


# -- guess_asset_class ------------------------------------------------------------------------------------
@pytest.mark.parametrize("symbol, expected", [
    ("BTC/USDT", AssetClass.CRYPTO),
    ("AAPL", AssetClass.STOCK),
    ("SAP.DE", AssetClass.STOCK),
    ("EURUSD=X", AssetClass.STOCK),  # anything without a slash is treated as a stock/ETF ticker
    ("", AssetClass.STOCK),
])
def test_asset_class_is_guessed_from_the_slash(symbol, expected):
    assert guess_asset_class(symbol) is expected


def test_balance_total_is_free_plus_used():
    assert Balance("USDT", free=70.0, used=30.0).total == 100.0


# -- normalize_ohlc ----------------------------------------------------------------------------------------
def test_upper_case_columns_are_accepted():
    raw = pd.DataFrame({"Open": [1.0], "High": [2.0], "Low": [0.5], "Close": [1.5]},
                       index=pd.DatetimeIndex(["2024-01-01"]))
    assert list(normalize_ohlc(raw).columns) == ["open", "high", "low", "close"]


def test_a_missing_ohlc_column_is_named_in_the_error():
    raw = pd.DataFrame({"open": [1.0], "high": [2.0], "close": [1.5]}, index=pd.DatetimeIndex(["2024-01-01"]))
    with pytest.raises(ValueError, match=r"lacks columns: \['low'\]"):
        normalize_ohlc(raw)


def test_non_numeric_prices_are_rejected():
    raw = ohlc(["2024-01-01"], close=["n/a"])
    with pytest.raises(ValueError):
        normalize_ohlc(raw)


def test_a_naive_index_is_taken_as_utc():
    out = normalize_ohlc(ohlc(["2024-01-01 09:00"], close=[1.0]))
    assert out.index[0] == pd.Timestamp("2024-01-01 09:00", tz="UTC")


def test_a_local_time_index_is_converted_to_utc():
    # 10:00 in Berlin in winter (UTC+1) is 09:00 UTC
    idx = pd.DatetimeIndex(["2024-01-15 10:00"]).tz_localize("Europe/Berlin")
    out = normalize_ohlc(ohlc(idx, close=[1.0]))
    assert out.index[0] == pd.Timestamp("2024-01-15 09:00", tz="UTC")


def test_duplicate_timestamps_keep_the_last_row():
    out = normalize_ohlc(ohlc(["2024-01-01", "2024-01-01"], close=[1.0, 2.0]))
    assert out["close"].tolist() == [2.0]


def test_rows_are_sorted_by_time():
    out = normalize_ohlc(ohlc(["2024-01-03", "2024-01-01", "2024-01-02"], close=[3.0, 1.0, 2.0]))
    assert out["close"].tolist() == [1.0, 2.0, 3.0]


def test_rows_with_a_missing_price_are_dropped():
    raw = ohlc(["2024-01-01", "2024-01-02"], close=[1.0, 2.0])
    raw.loc[raw.index[0], "high"] = np.nan
    assert normalize_ohlc(raw)["close"].tolist() == [2.0]


def test_extra_columns_are_dropped_and_volume_is_kept():
    out = normalize_ohlc(ohlc(["2024-01-01"], close=[1.0], volume=[5.0], vwap=[1.0]))
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]


def test_an_empty_frame_stays_empty():
    empty = pd.DataFrame(columns=["open", "high", "low", "close"], index=pd.DatetimeIndex([]))
    assert normalize_ohlc(empty).empty


# -- to_ohlc_frame -----------------------------------------------------------------------------------------
def test_exchange_rows_become_a_utc_frame():
    # 86_400_000 ms = 1970-01-02 00:00 UTC
    out = to_ohlc_frame([[86_400_000, 1, 2, 0.5, 1.5, 10]])
    assert out.index[0] == pd.Timestamp("1970-01-02", tz="UTC")
    assert out.iloc[0].tolist() == [1.0, 2.0, 0.5, 1.5, 10.0]


def test_rows_without_volume_have_no_volume_column():
    assert "volume" not in to_ohlc_frame([[0, 1, 2, 0.5, 1.5]]).columns


def test_no_rows_give_an_empty_frame():
    assert to_ohlc_frame([]).empty


# -- timeframes ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("timeframe, expected", [
    ("1m", pd.Timedelta(minutes=1)),
    ("4h", pd.Timedelta(hours=4)),
    ("1w", pd.Timedelta(days=7)),
    ("1mo", pd.Timedelta(days=31)),  # the longest month, so a gap check never mistakes a month for a hole
])
def test_known_timeframes(timeframe, expected):
    assert to_timedelta(timeframe) == expected


@pytest.mark.parametrize("typo", ["", "1H", "1 h", " 1h", "60m", "1y", "-1h", "1d1"])
def test_unknown_timeframes_are_rejected_listing_the_valid_ones(typo):
    with pytest.raises(ValueError, match="unknown timeframe.*1m, 5m"):
        to_timedelta(typo)


def test_month_start_keeps_the_timezone():
    assert month_start(pd.Timestamp("2024-02-29 15:30", tz="UTC")) == pd.Timestamp("2024-02-01", tz="UTC")


def test_resample_hours_to_four_hours():
    idx = pd.date_range("2024-01-01", periods=8, freq="h", tz="UTC")
    hourly = pd.DataFrame({"open": [1, 2, 3, 4, 5, 6, 7, 8], "high": [2, 3, 9, 5, 6, 7, 8, 9],
                           "low": [0, 1, 2, 3, 4, 1, 6, 7], "close": [2, 3, 4, 5, 6, 7, 8, 10],
                           "volume": [1, 1, 1, 1, 2, 2, 2, 2]}, index=idx, dtype=float)
    out = resample(hourly, "4h")
    # 00-03h: open 1, high max(2,3,9,5)=9, low 0, close 5, volume 4; 04-07h: 5, 9, 1, 10, 8
    assert out.to_numpy().tolist() == [[1, 9, 0, 5, 4], [5, 9, 1, 10, 8]]


def test_resample_to_months_follows_the_calendar():
    # 31 January days + 2 February days -> two monthly candles, labelled on the 1st
    idx = pd.date_range("2024-01-01", periods=33, freq="D", tz="UTC")
    daily = pd.DataFrame({c: np.arange(33.0) for c in ["open", "high", "low", "close"]}, index=idx)
    out = resample(daily, "1mo")
    assert out.index.tolist() == [pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-02-01", tz="UTC")]
    assert out["close"].tolist() == [30.0, 32.0]


def test_resample_skips_empty_periods():
    # 00:00 and 09:00 only: the 4h buckets 04:00 holds nothing and must not appear as a NaN candle
    hourly = ohlc(pd.DatetimeIndex(["2024-01-01 00:00", "2024-01-01 09:00"], tz="UTC"), close=[1.0, 2.0])
    out = resample(hourly, "4h")
    assert out.index.tolist() == [pd.Timestamp("2024-01-01 00:00", tz="UTC"),
                                  pd.Timestamp("2024-01-01 08:00", tz="UTC")]


def test_resample_to_an_unknown_timeframe_is_rejected():
    with pytest.raises(ValueError, match="unknown timeframe"):
        resample(ohlc(["2024-01-01"], close=[1.0]), "3h")
