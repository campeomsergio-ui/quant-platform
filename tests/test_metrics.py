import pandas as pd

from data_contracts import BacktestResult, CandidateConfig
from metrics import compute_diagnostics, compute_primary_metrics


def test_compute_primary_metrics_contains_required_fields() -> None:
    returns = pd.Series([0.01, -0.005, 0.002])
    costs = pd.Series([0.001, 0.001, 0.001])
    metrics = compute_primary_metrics(returns, costs)
    assert hasattr(metrics, "net_sharpe")
    assert hasattr(metrics, "annualized_return")


def test_compute_diagnostics_reports_factor_exposure_placeholders() -> None:
    series = pd.Series([0.0, 0.0])
    result = BacktestResult(candidate=CandidateConfig(candidate_id="x", params={}), gross_returns=series, net_returns=series, turnover=series, costs=series)
    diagnostics = compute_diagnostics(result)
    assert diagnostics.items["factor_exposure_status"] == "placeholder"
