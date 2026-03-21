from pathlib import Path

from cli import main
from quant_platform.sample_data import write_sample_daily_equities_dataset
from quant_platform.data_access import LocalJsonDataAdapter
from quant_platform.research import BaselineResearchConfig, run_baseline_research


def test_run_baseline_local_cli(tmp_path: Path) -> None:
    data_root = tmp_path / "sample"
    write_sample_daily_equities_dataset(str(data_root))
    rc = main(["run-baseline-local", "--data-root", str(data_root), "--residual-model", "industry_beta_log_mcap"])
    assert rc == 0


def test_research_output_contains_diagnostics(tmp_path: Path) -> None:
    data_root = tmp_path / "sample"
    write_sample_daily_equities_dataset(str(data_root))
    bundle = LocalJsonDataAdapter(str(data_root)).load_bundle()
    result = run_baseline_research(bundle, BaselineResearchConfig())
    assert "long_short_attribution" in result.diagnostics
    assert "turnover_decomposition" in result.diagnostics
    assert "capacity_stress" in result.diagnostics
    assert "data_validation" in result.diagnostics
    assert "history_quality" in result.diagnostics
    assert "coverage_quality" in result.diagnostics
