from pathlib import Path

from cli import main
from quant_platform.io import load_json
from quant_platform.research_ops import build_verified_sleeve_registry
from quant_platform.sample_data import write_sample_daily_equities_dataset


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_build_verified_sleeve_registry_uses_repo_outputs() -> None:
    registry = build_verified_sleeve_registry(str(REPO_ROOT))
    assert registry["stock_baseline_anchor"]["status"] == "killed"
    assert registry["stock_residual_momentum_single_candidate"]["status"] == "strongest_current_candidate"
    assert registry["etf_trend_baseline"]["status"] == "mechanically_valid_but_poor"
    assert registry["stock_residual_momentum_single_candidate"]["evidence"]["dominates_stock_baseline_on_mean_sharpe"] is True


def test_run_research_orchestrator_cli_exports_control_artifact(tmp_path: Path) -> None:
    data_root = tmp_path / "sample"
    export_path = tmp_path / "control_artifact.json"
    write_sample_daily_equities_dataset(str(data_root))

    rc = main([
        "run-research-orchestrator",
        "--data-root", str(data_root),
        "--preferred-format", "json",
        "--repo-root", str(REPO_ROOT),
        "--export-path", str(export_path),
        "--sleeve", "stock_baseline_anchor",
        "--sleeve", "stock_residual_momentum_single_candidate",
        "--sleeve", "etf_trend_baseline",
    ])
    assert rc == 0

    payload = load_json(str(export_path))
    assert payload["artifact_type"] == "agent_research_ops_control"
    assert payload["sleeve_registry"]["stock_baseline_anchor"]["status"] == "killed"
    assert payload["sleeve_registry"]["stock_residual_momentum_single_candidate"]["status"] == "strongest_current_candidate"
    assert set(payload["run_results"]) == {
        "stock_baseline_anchor",
        "stock_residual_momentum_single_candidate",
        "etf_trend_baseline",
    }
    assert "operator_decision_report" in payload
    assert "decisions" in payload["operator_decision_report"]
