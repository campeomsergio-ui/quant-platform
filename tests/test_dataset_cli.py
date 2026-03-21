from pathlib import Path

from cli import main
from quant_platform.io import load_json
from quant_platform.sample_data import write_sample_daily_equities_dataset


def test_inspect_dataset_cli_export(tmp_path: Path) -> None:
    data_root = tmp_path / "sample"
    export_path = tmp_path / "manifest.json"
    write_sample_daily_equities_dataset(str(data_root))
    rc = main(["inspect-data-local", "--data-root", str(data_root), "--export-path", str(export_path)])
    assert rc == 0
    payload = load_json(str(export_path))
    assert "manifest" in payload
    assert "validation" in payload
    assert "human_summary" in payload
