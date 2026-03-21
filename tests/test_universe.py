import pandas as pd

from universe import apply_security_exclusions, build_universe


def test_build_universe_applies_filters() -> None:
    dates = pd.date_range("2022-01-01", periods=252, freq="B")
    idx = pd.MultiIndex.from_product([dates, ["A", "B"]], names=["date", "symbol"])
    bars = pd.DataFrame({"close": [10.0, 2.0] * len(dates), "volume": [1_000_000, 1_000_000] * len(dates)}, index=idx)
    meta = pd.DataFrame({"market_cap": [1_000_000_000.0, 1_000_000_000.0], "security_type": ["COMMON", "COMMON"], "is_primary_listing": [True, True]}, index=["A", "B"])
    assert build_universe(bars, meta, dates[-1]) == ["A"]


def test_build_universe_excludes_ineligible_security_types() -> None:
    meta = pd.DataFrame({"security_type": ["ETF", "COMMON"], "is_primary_listing": [True, True]}, index=["X", "Y"])
    assert apply_security_exclusions(meta, pd.Timestamp("2022-12-30")) == ["Y"]
