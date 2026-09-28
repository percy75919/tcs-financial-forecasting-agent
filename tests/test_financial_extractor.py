from app.tools.financial_data_extractor import _first_float, _last_float_row


def test_first_float_matches_common_tcs_formats():
    text = "Revenue at US$ 7,624 million; Operating Margin at 24.0%; TCV: US$ 9.5 billion"
    assert _first_float([r"Revenue at US\$\s*([\d,]+)\s*million"], text) == 7624
    assert _first_float([r"Operating Margin at\s*([\d.]+)%"], text) == 24.0
    assert _first_float([r"TCV[^$]{0,40}US\$\s*([\d.]+)\s*billion"], text) == 9.5


def test_quarterly_table_uses_final_period_value():
    text = """
    For the three-month periods ended Mar 31, 2025, Dec 31, 2025 and Mar 31, 2026
    Revenue | 7,465 | 7,509 | 7,621
    Net income | 1,418 | 1,503 | 1,479
    """
    assert _last_float_row(
        [r"Revenue\s*\|\s*([\d,]+)\s*\|\s*([\d,]+)\s*\|\s*([\d,]+)"], text
    ) == 7621
    assert _last_float_row(
        [r"Net income\s*\|\s*([\d,]+)\s*\|\s*([\d,]+)\s*\|\s*([\d,]+)"], text
    ) == 1479
