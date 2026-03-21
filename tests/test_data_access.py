from pathlib import Path

import pandas as pd

from quant_platform.data_access import LocalJsonDataAdapter, LocalTableDataAdapter, apply_symbol_mapping, attach_delisting_returns, build_dataset_manifest, ensure_valid_point_in_time_bundle, inspect_local_dataset, validate_point_in_time_bundle
from quant_platform.sample_data import write_sample_daily_equities_dataset


def test_local_adapter_and_pit_validation(tmp_path: Path) -> None:
    write_sample_daily_equities_dataset(str(tmp_path))
    bundle = LocalJsonDataAdapter(str(tmp_path)).load_bundle()
    report = validate_point_in_time_bundle(bundle)
    assert report.ok
    assert not bundle.bars.empty
    assert not bundle.metadata.empty


def test_manifest_helper_output_integrity(tmp_path: Path) -> None:
    write_sample_daily_equities_dataset(str(tmp_path))
    payload = inspect_local_dataset(str(tmp_path))
    assert "manifest" in payload
    assert "validation" in payload
    assert "human_summary" in payload


def test_manifest_generation_and_coverage_summary(tmp_path: Path) -> None:
    write_sample_daily_equities_dataset(str(tmp_path))
    bundle = LocalJsonDataAdapter(str(tmp_path)).load_bundle()
    manifest = build_dataset_manifest(bundle.bars, bundle.metadata, bundle.benchmark, bundle.delistings, bundle.symbol_mapping)
    assert "date_range" in manifest
    assert "symbol_coverage" in manifest
    assert "field_availability" in manifest


def test_local_table_adapter_reads_fixture(tmp_path: Path) -> None:
    write_sample_daily_equities_dataset(str(tmp_path))
    bundle = LocalTableDataAdapter(str(tmp_path)).load_bundle()
    assert "history_quality" in bundle.dataset_manifest
    assert len(bundle.benchmark) > 0


def test_symbol_mapping_handles_symbol_change(tmp_path: Path) -> None:
    write_sample_daily_equities_dataset(str(tmp_path))
    bundle = LocalJsonDataAdapter(str(tmp_path)).load_bundle()
    mapped = apply_symbol_mapping(bundle.bars, bundle.symbol_mapping, pd.Timestamp("2022-01-20"))
    assert "NEW" in mapped.index.get_level_values("symbol")


def test_delisting_hook_overrides_terminal_return() -> None:
    idx = pd.date_range("2022-01-03", periods=3, freq="B")
    returns = pd.DataFrame({"AAA": [0.01, 0.02, 0.03]}, index=idx)
    delistings = pd.DataFrame([{"symbol": "AAA", "delisting_date": idx[-1], "delisting_return": -0.9}])
    updated = attach_delisting_returns(returns, delistings)
    assert updated.loc[idx[-1], "AAA"] == -0.9


def test_pit_validation_rejects_bad_ranges(tmp_path: Path) -> None:
    write_sample_daily_equities_dataset(str(tmp_path))
    bundle = LocalJsonDataAdapter(str(tmp_path)).load_bundle()
    bundle.metadata.loc[0, "effective_to"] = pd.Timestamp("2021-01-01")
    report = validate_point_in_time_bundle(bundle)
    assert not report.ok
    assert any(issue.code == "invalid_metadata_ranges" for issue in report.issues)


def test_benchmark_coverage_check(tmp_path: Path) -> None:
    write_sample_daily_equities_dataset(str(tmp_path))
    bundle = LocalJsonDataAdapter(str(tmp_path)).load_bundle()
    bundle = type(bundle)(
        bars=bundle.bars,
        corporate_actions=bundle.corporate_actions,
        metadata=bundle.metadata,
        benchmark=bundle.benchmark.iloc[:-1],
        delistings=bundle.delistings,
        symbol_mapping=bundle.symbol_mapping,
        dataset_manifest=bundle.dataset_manifest,
        data_quality_metadata=bundle.data_quality_metadata,
    )
    report = validate_point_in_time_bundle(bundle)
    assert not report.ok
    assert any(issue.code in {"missing_benchmark_coverage", "benchmark_gap_ratio"} for issue in report.issues)


def test_symbol_mapping_continuity_check(tmp_path: Path) -> None:
    write_sample_daily_equities_dataset(str(tmp_path))
    bundle = LocalJsonDataAdapter(str(tmp_path)).load_bundle()
    bundle.symbol_mapping.loc[1, "effective_from"] = bundle.symbol_mapping.loc[0, "effective_from"]
    bundle.symbol_mapping.loc[0, "effective_to"] = bundle.symbol_mapping.loc[0, "effective_from"] + pd.Timedelta(days=10)
    report = validate_point_in_time_bundle(bundle)
    assert any(issue.code == "mapping_overlap" for issue in report.issues)


def test_history_sufficiency_tagging(tmp_path: Path) -> None:
    write_sample_daily_equities_dataset(str(tmp_path))
    bundle = LocalJsonDataAdapter(str(tmp_path)).load_bundle()
    report = validate_point_in_time_bundle(bundle)
    assert report.summary["history_quality"] in {"short_history", "adequate_history"}
    assert report.summary["coverage_quality"] in {"narrow_or_sample", "broad", "no_coverage"}


def test_ensure_valid_bundle_raises_on_errors(tmp_path: Path) -> None:
    write_sample_daily_equities_dataset(str(tmp_path))
    bundle = LocalJsonDataAdapter(str(tmp_path)).load_bundle()
    bundle.metadata.loc[0, "effective_to"] = pd.Timestamp("2021-01-01")
    try:
        ensure_valid_point_in_time_bundle(bundle)
    except ValueError:
        assert True
    else:
        assert False
