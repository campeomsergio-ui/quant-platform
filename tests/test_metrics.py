import pandas as pd

from data_contracts import BacktestResult, CandidateConfig
from metrics import compute_diagnostics, compute_primary_metrics


def test_compute_primary_metrics_contains_required_fields() -> None:
    returns = pd.Series([0.01, -0.005, 0.002])
    costs = pd.Series([0.001, 0.001, 0.001])
    metrics = compute_primary_metrics(returns, costs)
    assert hasattr(metrics, "net_sharpe")
    assert hasattr(metrics, "annualized_return")


def test_long_vs_short_attribution() -> None:
    series = pd.Series([0.0, 0.0])
    result = BacktestResult(
        candidate=CandidateConfig(candidate_id="x", params={}),
        gross_returns=series,
        net_returns=series,
        turnover=series,
        costs=series,
        diagnostics={"leg_pnl": {"2022-01-01": {"long_contribution": 1.0, "short_contribution": -0.5}}},
    )
    diagnostics = compute_diagnostics(result)
    assert diagnostics.items["long_short_attribution"]["long_total"] == 1.0


def test_turnover_decomposition_consistency() -> None:
    series = pd.Series([0.0, 0.0])
    result = BacktestResult(
        candidate=CandidateConfig(candidate_id="x", params={}),
        gross_returns=series,
        net_returns=series,
        turnover=series,
        costs=series,
        diagnostics={"turnover_decomposition": {"2022-01-01": {"signal_change": 1.0, "tranche_rolloff": 2.0, "constraint_repair": 0.5, "internal_netting_effect": 0.25}}},
    )
    diagnostics = compute_diagnostics(result)
    assert diagnostics.items["turnover_decomposition"]["tranche_rolloff"] == 2.0


def test_exposure_summary_generation() -> None:
    series = pd.Series([0.0, 0.0])
    result = BacktestResult(
        candidate=CandidateConfig(candidate_id="x", params={}),
        gross_returns=series,
        net_returns=series,
        turnover=series,
        costs=series,
        diagnostics={"exposure_summary": {"2022-01-01": {"beta": 0.1, "sector_exposures": {"Tech": 0.2}, "size_proxy": 1.0, "concentration": 0.3, "effective_breadth": 4.0}}},
    )
    diagnostics = compute_diagnostics(result)
    assert diagnostics.items["exposure_summary"]["beta"] == 0.1


def test_capacity_stress_reporting() -> None:
    series = pd.Series([0.0, 0.0])
    result = BacktestResult(
        candidate=CandidateConfig(candidate_id="x", params={}),
        gross_returns=series,
        net_returns=series,
        turnover=series,
        costs=series,
        diagnostics={"capacity_stress": {"2022-01-01": {"baseline_cost": 1.0, "cost_1_5x": 1.5, "cost_2_0x": 2.0, "avg_adv_participation": 0.01, "max_adv_participation": 0.02}}},
    )
    diagnostics = compute_diagnostics(result)
    assert diagnostics.items["capacity_stress"]["cost_2_0x"] == 2.0


def test_delay_cost_sensitivity_reporting() -> None:
    series = pd.Series([0.0, 0.0])
    result = BacktestResult(
        candidate=CandidateConfig(candidate_id="x", params={}),
        gross_returns=series,
        net_returns=series,
        turnover=series,
        costs=series,
        diagnostics={"sensitivity": {"execution_delay": {"baseline_days": 1}, "cost_stress": {"baseline_total_cost": 1.0}}},
    )
    diagnostics = compute_diagnostics(result)
    assert diagnostics.items["delay_cost_sensitivity"]["execution_delay"]["baseline_days"] == 1
