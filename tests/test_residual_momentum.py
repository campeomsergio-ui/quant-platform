from pathlib import Path

from cli import main
from quant_platform.data_access import LocalJsonDataAdapter
from quant_platform.research import BaselineResearchConfig, ResidualMomentumCycleConfig, build_residual_momentum_signals, run_residual_momentum_cycle
from quant_platform.sample_data import write_sample_daily_equities_dataset
from quant_platform.signals.residual_momentum import ResidualMomentumParams, compute_residual_momentum_scores
from quant_platform.validation.multiple_testing import RealityCheckConfig, build_candidate_differentials, run_white_reality_check


def _bundle(tmp_path: Path):
    data_root = tmp_path / "sample"
    write_sample_daily_equities_dataset(str(data_root))
    return LocalJsonDataAdapter(str(data_root)).load_bundle()


def test_signal_construction(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    signals = build_residual_momentum_signals(bundle, 20, 5, "industry_beta_log_mcap", 1)
    assert signals


def test_skip_window_behavior(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    bars = bundle.bars
    meta = bundle.metadata.sort_values(["symbol", "effective_from"]).drop_duplicates("symbol", keep="last").set_index("symbol")
    params_a = ResidualMomentumParams(lookback=5, skip_window=0, residual_model="industry_beta")
    params_b = ResidualMomentumParams(lookback=5, skip_window=3, residual_model="industry_beta")
    scores_a = compute_residual_momentum_scores(bars, meta, params_a)
    scores_b = compute_residual_momentum_scores(bars, meta, params_b)
    assert not scores_a.values.equals(scores_b.values)


def test_residualization_behavior(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    bars = bundle.bars
    meta = bundle.metadata.sort_values(["symbol", "effective_from"]).drop_duplicates("symbol", keep="last").set_index("symbol")
    params = ResidualMomentumParams(lookback=5, skip_window=1, residual_model="industry_beta_log_mcap")
    scores = compute_residual_momentum_scores(bars, meta, params)
    assert len(scores.values) > 0


def test_benchmark_differential_series_construction() -> None:
    benchmark = build_candidate_differentials({"a": __import__("pandas").Series([0.01, 0.02]), "b": __import__("pandas").Series([0.00, 0.01])}, __import__("pandas").Series([0.005, 0.005]))
    assert float(benchmark["a"].iloc[0]) == 0.005


def test_multiple_testing_interface_behavior() -> None:
    import pandas as pd

    result = run_white_reality_check({"a": pd.Series([0.01, 0.02, 0.01]), "b": pd.Series([0.0, 0.0, 0.0])}, pd.Series([0.005, 0.005, 0.005]), RealityCheckConfig(bootstrap_iterations=10, seed=1))
    assert result.differential_summary is not None
    assert "a" in result.differential_summary


def test_research_cycle_candidate_accounting(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    result = run_residual_momentum_cycle(bundle, BaselineResearchConfig(), ResidualMomentumCycleConfig(lookbacks=(20,), skip_windows=(5,), residual_models=("industry_beta",), seed=7), registry={})
    exp = result.registry["experiments"][result.experiment_id]
    assert len(exp["candidates"]) == 1


def test_fold_generation_and_comparison_output_consistency(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    result = run_residual_momentum_cycle(bundle, BaselineResearchConfig(), ResidualMomentumCycleConfig(lookbacks=(20,), skip_windows=(5,), residual_models=("industry_beta",), seed=7), registry={})
    assert result.fold_results
    fold_id = next(iter(result.fold_results.keys()))
    assert "residual_momentum" in result.comparison["by_fold"][fold_id]
    assert "benchmark_differentials" in result.comparison["by_fold"][fold_id]


def test_development_stage_cannot_silently_touch_final_test(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    try:
        run_residual_momentum_cycle(bundle, BaselineResearchConfig(), ResidualMomentumCycleConfig(lookbacks=(20,), skip_windows=(5,), residual_models=("industry_beta",), seed=7, stage="development", touch_final_test=True, final_test_reason="should fail"), registry={})
    except ValueError:
        assert True
    else:
        assert False


def test_explicit_final_test_touch_path_works_and_is_recorded(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    result = run_residual_momentum_cycle(bundle, BaselineResearchConfig(), ResidualMomentumCycleConfig(lookbacks=(20,), skip_windows=(5,), residual_models=("industry_beta",), seed=7, stage="final_test", touch_final_test=True, final_test_reason="explicit_final_evaluation"), registry={})
    exp = result.registry["experiments"][result.experiment_id]
    assert exp["final_test_touched"] is True
    assert exp["final_test_stage"] == "final_test"


def test_repeated_final_test_usage_is_blocked(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    result = run_residual_momentum_cycle(bundle, BaselineResearchConfig(), ResidualMomentumCycleConfig(lookbacks=(20,), skip_windows=(5,), residual_models=("industry_beta",), seed=7, stage="final_test", touch_final_test=True, final_test_reason="explicit_final_evaluation"), registry={})
    try:
        run_residual_momentum_cycle(bundle, BaselineResearchConfig(), ResidualMomentumCycleConfig(lookbacks=(20,), skip_windows=(5,), residual_models=("industry_beta",), seed=7, stage="final_test", touch_final_test=True, final_test_reason="repeat"), registry=result.registry)
    except ValueError:
        assert True
    else:
        assert False


def test_cycle_metadata_remains_separate_from_baseline(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    result = run_residual_momentum_cycle(bundle, BaselineResearchConfig(), ResidualMomentumCycleConfig(lookbacks=(20,), skip_windows=(5,), residual_models=("industry_beta",), seed=7), registry={})
    assert result.comparison["baseline"]["label"] == "frozen_mean_reversion"
    assert result.comparison["cycle_label"] == "new_research_cycle"


def test_comparison_report_outputs(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    result = run_residual_momentum_cycle(bundle, BaselineResearchConfig(), ResidualMomentumCycleConfig(lookbacks=(20,), skip_windows=(5,), residual_models=("industry_beta",), seed=7), registry={})
    assert result.comparison["aggregate"]["benchmark_differentials"]
    assert "final_test_state" in result.comparison["aggregate"]


def test_cli_export_path(tmp_path: Path) -> None:
    data_root = tmp_path / "sample"
    export_path = tmp_path / "rmom_report.json"
    registry_path = tmp_path / "registry.json"
    write_sample_daily_equities_dataset(str(data_root))
    rc = main(["run-residual-momentum-cycle", "--data-root", str(data_root), "--registry", str(registry_path), "--export-path", str(export_path), "--lookback", "20", "--skip-window", "5", "--residual-model", "industry_beta"])
    assert rc == 0
    assert export_path.exists()
    assert registry_path.exists()
