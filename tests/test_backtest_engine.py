import pandas as pd

from quant_platform.backtest.engine import BacktestConfig, run_backtest, simulate_tranches
from quant_platform.backtest.rebalance import generate_orders


def _market_frame(dates: pd.DatetimeIndex, symbols: list[str], sectors: dict[str, str] | None = None, betas: dict[str, float] | None = None, advs: dict[str, float] | None = None) -> pd.DataFrame:
    idx = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
    rows = []
    for _date in dates:
        for _symbol in symbols:
            rows.append(
                {
                    "open": 100.0,
                    "close": 101.0,
                    "adv": (advs or {}).get(_symbol, 10_000.0),
                    "daily_volatility": 0.02,
                    "sector": (sectors or {}).get(_symbol, "UNKNOWN"),
                    "beta": (betas or {}).get(_symbol, 0.0),
                }
            )
    return pd.DataFrame(rows, index=idx)


def test_run_backtest_generates_overlapping_tranches() -> None:
    dates = pd.date_range("2022-01-03", periods=6, freq="B")
    signals = {date: pd.Series({"A": 0.1, "B": -0.1}) for date in dates}
    market_data = _market_frame(dates, ["A", "B"])
    result = run_backtest(None, {"signals": signals, "market_data": market_data, "config": BacktestConfig(min_longs=1, min_shorts=1, max_sector_weight=1.0)})
    assert len(result.net_returns) == 5
    assert result.diagnostics["status"] == "implemented_core_backtest"
    tranches = simulate_tranches(signals, 5)
    assert len(tranches.columns) == len(dates)
    assert isinstance(tranches.loc[dates[-1], dates[-1].strftime("%Y-%m-%d")], pd.Series)


def test_generate_orders_matches_target_weights() -> None:
    orders = generate_orders(pd.Series({"A": 0.0}), pd.Series({"A": 0.1}))
    assert orders == [{"symbol": "A", "target_delta": 0.1}]


def test_tranche_aggregation_and_internal_crossing_reduce_turnover() -> None:
    dates = pd.date_range("2022-01-03", periods=7, freq="B")
    signals = {
        dates[0]: pd.Series({"A": 0.01, "B": -0.01}),
        dates[1]: pd.Series({"A": -0.01, "B": 0.01}),
        dates[2]: pd.Series({"A": 0.01, "B": -0.01}),
        dates[3]: pd.Series({"A": -0.01, "B": 0.01}),
        dates[4]: pd.Series({"A": 0.01, "B": -0.01}),
        dates[5]: pd.Series({"A": -0.01, "B": 0.01}),
        dates[6]: pd.Series({"A": 0.01, "B": -0.01}),
    }
    market_data = _market_frame(dates, ["A", "B"])
    result = run_backtest(None, {"signals": signals, "market_data": market_data, "config": BacktestConfig(max_name_weight=0.02, min_longs=1, min_shorts=1, max_sector_weight=1.0)})
    assert float(result.turnover.max()) <= 0.04


def test_delayed_execution_uses_next_day() -> None:
    dates = pd.date_range("2022-01-03", periods=4, freq="B")
    signals = {date: pd.Series({"A": 0.01, "B": -0.01}) for date in dates}
    market_data = _market_frame(dates, ["A", "B"])
    result = run_backtest(None, {"signals": signals, "market_data": market_data, "config": BacktestConfig(min_longs=1, min_shorts=1, max_sector_weight=1.0)})
    assert result.net_returns.index[0] == dates[1]


def test_blocked_names_hold_previous_book() -> None:
    dates = pd.date_range("2022-01-03", periods=4, freq="B")
    signals = {
        dates[0]: pd.Series({"A": 0.01, "B": -0.01}),
        dates[1]: pd.Series({"A": 0.0, "B": 0.0}),
        dates[2]: pd.Series({"A": 0.0, "B": 0.0}),
        dates[3]: pd.Series({"A": 0.0, "B": 0.0}),
    }
    market_data = _market_frame(dates, ["A", "B"])
    blocked = pd.DataFrame(False, index=dates, columns=["A", "B"])
    blocked.loc[dates[2], "A"] = True
    result = run_backtest(None, {"signals": signals, "market_data": market_data, "blocked": blocked, "config": BacktestConfig(min_longs=1, min_shorts=1, max_sector_weight=1.0)})
    assert result.turnover.iloc[0] > 0.0
    assert result.turnover.iloc[1] == 0.0


def test_sector_cap_breach_is_recorded() -> None:
    dates = pd.date_range("2022-01-03", periods=3, freq="B")
    signals = {date: pd.Series({"A": 0.04, "B": -0.04}) for date in dates}
    market_data = _market_frame(dates, ["A", "B"], sectors={"A": "Tech", "B": "Tech"})
    result = run_backtest(None, {"signals": signals, "market_data": market_data, "config": BacktestConfig(min_longs=1, min_shorts=1, max_sector_weight=0.01)})
    assert any(event.get("constraint") == "sector_cap" for event in result.diagnostics["constraint_events"])


def test_insufficient_breadth_yields_failure_state() -> None:
    dates = pd.date_range("2022-01-03", periods=3, freq="B")
    signals = {date: pd.Series({"A": 0.02, "B": -0.02}) for date in dates}
    market_data = _market_frame(dates, ["A", "B"])
    result = run_backtest(None, {"signals": signals, "market_data": market_data, "config": BacktestConfig(min_longs=2, min_shorts=2, max_sector_weight=1.0)})
    assert result.diagnostics["status"] == "constraint_failures_present"
    assert result.diagnostics["failure_days"]


def test_beta_neutralization_behavior_is_recorded() -> None:
    dates = pd.date_range("2022-01-03", periods=3, freq="B")
    signals = {date: pd.Series({"A": 0.01, "B": -0.005, "C": -0.005}) for date in dates}
    market_data = _market_frame(dates, ["A", "B", "C"], betas={"A": 2.0, "B": 0.5, "C": 0.5}, sectors={"A": "Tech", "B": "Health", "C": "Energy"})
    result = run_backtest(None, {"signals": signals, "market_data": market_data, "config": BacktestConfig(min_longs=1, min_shorts=2, max_sector_weight=1.0)})
    assert any(event.get("constraint") == "beta_neutralization" for event in result.diagnostics["constraint_events"])


def test_liquidity_cap_breach_is_recorded() -> None:
    dates = pd.date_range("2022-01-03", periods=3, freq="B")
    signals = {date: pd.Series({"A": 0.05, "B": -0.05}) for date in dates}
    market_data = _market_frame(dates, ["A", "B"], advs={"A": 1.0, "B": 1.0}, sectors={"A": "Tech", "B": "Health"})
    result = run_backtest(None, {"signals": signals, "market_data": market_data, "config": BacktestConfig(min_longs=1, min_shorts=1, max_sector_weight=1.0, participation_cap=0.01)})
    assert any(event.get("constraint") == "liquidity_cap" for event in result.diagnostics["constraint_events"])


def test_gross_net_limit_enforcement_after_netting_is_recorded() -> None:
    dates = pd.date_range("2022-01-03", periods=3, freq="B")
    signals = {date: pd.Series({"A": 0.08, "B": 0.08, "C": -0.02}) for date in dates}
    market_data = _market_frame(dates, ["A", "B", "C"], sectors={"A": "Tech", "B": "Health", "C": "Energy"})
    result = run_backtest(None, {"signals": signals, "market_data": market_data, "config": BacktestConfig(gross_limit=0.03, net_limit=0.0, min_longs=1, min_shorts=1, max_sector_weight=1.0, max_name_weight=0.08)})
    constraints = {event.get("constraint") for event in result.diagnostics["constraint_events"]}
    assert "gross_limit" in constraints or "net_limit" in constraints


def test_unsatisfiable_portfolio_case_is_explicit() -> None:
    dates = pd.date_range("2022-01-03", periods=3, freq="B")
    signals = {date: pd.Series({"A": 0.05}) for date in dates}
    market_data = _market_frame(dates, ["A"], sectors={"A": "Tech"}, betas={"A": 2.0})
    result = run_backtest(None, {"signals": signals, "market_data": market_data, "config": BacktestConfig(min_longs=1, min_shorts=1, max_sector_weight=0.001, net_limit=0.0)})
    assert result.diagnostics["status"] == "constraint_failures_present"
    assert any(event.get("action") == "failure" for event in result.diagnostics["constraint_events"])
