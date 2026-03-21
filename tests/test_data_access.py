from pathlib import Path

import pandas as pd

from quant_platform.data_access import LocalJsonDataAdapter, apply_symbol_mapping, attach_delisting_returns, validate_point_in_time_bundle
from quant_platform.sample_data import write_sample_daily_equities_dataset


def test_local_adapter_and_pit_validation(tmp_path: Path) -> None:
    write_sample_daily_equities_dataset(str(tmp_path))
    bundle = LocalJsonDataAdapter(str(tmp_path)).load_bundle()
    validate_point_in_time_bundle(bundle)
    assert not bundle.bars.empty
    assert not bundle.metadata.empty


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
    try:
        validate_point_in_time_bundle(bundle)
    except ValueError:
        assert True
    else:
        assert False
