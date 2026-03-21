import pandas as pd

from costs import estimate_costs, stress_costs


def test_estimate_costs_increases_with_participation() -> None:
    trades = pd.DataFrame({"trade_notional": [1_000_000.0, 2_000_000.0], "short_notional": [0.0, 0.0]}, index=["A", "B"])
    market = pd.DataFrame({"adv": [10_000_000.0, 10_000_000.0], "daily_volatility": [0.02, 0.02]}, index=["A", "B"])
    costs = estimate_costs(trades, market)
    assert costs.per_name_cost["B"] > costs.per_name_cost["A"]


def test_stress_costs_scales_costs() -> None:
    trades = pd.DataFrame({"trade_notional": [1_000_000.0], "short_notional": [0.0]}, index=["A"])
    market = pd.DataFrame({"adv": [10_000_000.0], "daily_volatility": [0.02]}, index=["A"])
    costs = estimate_costs(trades, market)
    stressed = stress_costs(costs, 2.0)
    assert stressed.total_cost == costs.total_cost * 2.0
