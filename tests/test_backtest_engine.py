import pandas as pd

from quant_platform.backtest.engine import BacktestConfig, run_backtest, simulate_tranches
from quant_platform.backtest.rebalance import generate_orders


def _market_frame(dates: pd.DatetimeIndex, symbols: list[str]) -> pd.DataFrame:
    idx = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
    rows = []
    for _date in dates:
        for _symbol in symbols:
            rows.append({"open": 100.0, "close": 101.0, "adv": 10_000.0, "daily_volatility": 0.02})
    return pd.DataFrame(rows, index=idx)


def test_run_backtest_generates_overlapping_tranches() -> None:
    dates = pd.date_range("2022-01-03", periods=6, freq="B")
    signals = {date: pd.Series({"A": 0.1, "B": -0.1}) for date in dates}
    market_data = _market_frame(dates, ["A", "B"])
    result = run_backtest(None, {"signals": signals, "market_data": market_data, "config": BacktestConfig()})
    assert len(result.net_returns) == 5
    assert result.diagnostics["status"] == "implemented_core_backtest"
    tranches = simulate_tranches(signals, 5)
    assert len(tranches.columns) == 5


def test_generate_orders_matches_target_weights() -> None:
    orders = generate_orders(pd.Series({"A": 0.0}), pd.Series({"A": 0.1}))
    assert orders == [{"symbol": "A", "target_delta": 0.1}]


def test_tranche_aggregation_and_internal_crossing_reduce_turnover() -> None:
    dates = pd.date_range("2022-01-03", periods=7, freq="B")
    signals = {
        dates[0]: pd.Series({"A": 0.01}),
        dates[1]: pd.Series({"A": -0.01}),
        dates[2]: pd.Series({"A": 0.01}),
        dates[3]: pd.Series({"A": -0.01}),
        dates[4]: pd.Series({"A": 0.01}),
        dates[5]: pd.Series({"A": -0.01}),
        dates[6]: pd.Series({"A": 0.01}),
    }
    market_data = _market_frame(dates, ["A"])
    result = run_backtest(None, {"signals": signals, "market_data": market_data, "config": BacktestConfig(max_name_weight=0.02)})
    assert float(result.turnover.max()) <= 0.02


def test_delayed_execution_uses_next_day() -> None:
    dates = pd.date_range("2022-01-03", periods=4, freq="B")
    signals = {date: pd.Series({"A": 0.01}) for date in dates}
    market_data = _market_frame(dates, ["A"])
    result = run_backtest(None, {"signals": signals, "market_data": market_data, "config": BacktestConfig()})
    assert result.net_returns.index[0] == dates[1]


def test_blocked_names_hold_previous_book() -> None:
    dates = pd.date_range("2022-01-03", periods=4, freq="B")
    signals = {
        dates[0]: pd.Series({"A": 0.01}),
        dates[1]: pd.Series({"A": 0.0}),
        dates[2]: pd.Series({"A": 0.0}),
        dates[3]: pd.Series({"A": 0.0}),
    }
    market_data = _market_frame(dates, ["A"])
    blocked = pd.DataFrame(False, index=dates, columns=["A"])
    blocked.loc[dates[2], "A"] = True
    result = run_backtest(None, {"signals": signals, "market_data": market_data, "blocked": blocked, "config": BacktestConfig()})
    assert result.turnover.iloc[0] > 0.0
    assert result.turnover.iloc[1] == 0.0
