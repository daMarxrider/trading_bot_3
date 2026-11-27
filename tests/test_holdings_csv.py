# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""The holdings CSV of the brokers without an API (Scalable Capital, Trade Republic), as users export it."""

import pytest

from tradebot.brokers.stocks import ScalableCapitalBroker
from tradebot.core.models import AssetClass


def holdings(tmp_path, text: str, encoding: str = "utf-8"):
    path = tmp_path / "holdings.csv"
    path.write_bytes(text.encode(encoding))
    return [(p.symbol, p.quantity, p.avg_cost) for p in ScalableCapitalBroker({"holdings_csv": str(path)}).get_positions()]


def test_a_plain_file(tmp_path):
    assert holdings(tmp_path, "symbol,quantity,avg_cost\nAAPL,10,150.5\n") == [("AAPL", 10.0, 150.5)]


def test_an_excel_utf8_file_with_a_byte_order_mark(tmp_path):
    assert holdings(tmp_path, "﻿symbol,quantity,avg_cost\nAAPL,10,150\n") == [("AAPL", 10.0, 150.0)]


def test_a_semicolon_separated_file(tmp_path):
    # what a German Excel writes for "CSV"
    assert holdings(tmp_path, "symbol;quantity;avg_cost\nSAP.DE;5;120.5\n") == [("SAP.DE", 5.0, 120.5)]


def test_windows_line_endings(tmp_path):
    assert holdings(tmp_path, "symbol,quantity\r\nAAPL,1\r\nMSFT,2\r\n") == [("AAPL", 1.0, None), ("MSFT", 2.0, None)]


def test_header_case_spaces_and_extra_columns_are_tolerated(tmp_path):
    text = " Symbol , QUANTITY ,Avg_Cost,currency\n AAPL , 3 , ,USD\n"
    assert holdings(tmp_path, text) == [("AAPL", 3.0, None)]


def test_rows_without_symbol_or_quantity_are_skipped(tmp_path):
    text = "symbol,quantity,avg_cost\n,5,1\nAAPL,,1\n\nMSFT\nNVDA,2,3\n"
    assert holdings(tmp_path, text) == [("NVDA", 2.0, 3.0)]


def test_a_row_with_more_cells_than_the_header_keeps_its_known_columns(tmp_path):
    assert holdings(tmp_path, "symbol,quantity\nAAPL,1,surplus,cells\n") == [("AAPL", 1.0, None)]


def test_an_empty_file_has_no_holdings(tmp_path):
    assert holdings(tmp_path, "") == []


def test_a_header_only_file_has_no_holdings(tmp_path):
    assert holdings(tmp_path, "symbol,quantity,avg_cost\n") == []


def test_crypto_symbols_are_crypto(tmp_path):
    path = tmp_path / "holdings.csv"
    path.write_text("symbol,quantity\nBTC/EUR,0.5\n")
    (pos,) = ScalableCapitalBroker({"holdings_csv": str(path)}).get_positions()
    assert pos.asset_class is AssetClass.CRYPTO


@pytest.mark.parametrize("text", [
    "ticker,amount\nAAPL,1\n",  # other column names
    "symbol\tquantity\nAAPL\t1\n",  # tab separated: one unknown column
])
def test_a_file_without_the_needed_columns_names_what_it_found(tmp_path, text):
    with pytest.raises(ValueError, match="needs the columns symbol and quantity"):
        holdings(tmp_path, text)


@pytest.mark.parametrize("cell, column", [
    ("ten", "quantity"),
    ("nan", "quantity"),
    ("inf", "quantity"),
])
def test_a_quantity_that_is_not_a_number_names_its_line(tmp_path, cell, column):
    with pytest.raises(ValueError, match=rf"line 3: {column} '{cell}' is not a number"):
        holdings(tmp_path, f"symbol,quantity\nAAPL,1\nMSFT,{cell}\n")


def test_a_decimal_comma_is_rejected_with_a_hint(tmp_path):
    with pytest.raises(ValueError, match="line 2: avg_cost '120,5' is not a number.*dot as the decimal point"):
        holdings(tmp_path, "symbol;quantity;avg_cost\nSAP.DE;5;120,5\n")


def test_a_missing_file_is_reported(tmp_path):
    broker = ScalableCapitalBroker({"holdings_csv": str(tmp_path / "missing.csv")})
    with pytest.raises(FileNotFoundError, match="does not exist; fix brokers.scalable.holdings_csv"):
        broker.get_positions()


def test_without_a_configured_file_it_says_what_to_set():
    with pytest.raises(NotImplementedError, match="brokers.scalable.holdings_csv"):
        ScalableCapitalBroker({}).get_positions()


def test_the_balance_lists_each_holding(tmp_path):
    path = tmp_path / "holdings.csv"
    path.write_text("symbol,quantity\nAAPL,2\n")
    assert ScalableCapitalBroker({"holdings_csv": str(path)}).get_balance()["AAPL"].free == 2.0
