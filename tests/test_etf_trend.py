from pathlib import Path

from cli import main
from quant_platform.io import load_json
from quant_platform.sample_data import write_sample_daily_equities_dataset


def test_etf_trend_cli_runs_on_sample_dataset(tmp_path: Path) -> None:
    data_root = tmp_path / 'sample'
    export_path = tmp_path / 'etf_cycle.json'
    write_sample_daily_equities_dataset(str(data_root))
    rc = main(['run-etf-trend-cycle', '--data-root', str(data_root), '--export-path', str(export_path), '--candidate-id', 'tsmom_ret_63_cash'])
    assert rc == 0
    payload = load_json(str(export_path))
    assert payload['cycle_label'] == 'etf_trend_following_cycle'
    assert 'best_candidate' in payload


def test_etf_refined_cli_runs_on_sample_dataset(tmp_path: Path) -> None:
    data_root = tmp_path / 'sample'
    export_path = tmp_path / 'etf_refined_cycle.json'
    write_sample_daily_equities_dataset(str(data_root))
    rc = main(['run-etf-trend-cycle', '--family', 'refined', '--data-root', str(data_root), '--export-path', str(export_path), '--candidate-id', 'tsmom_ma_200_cash'])
    assert rc == 0
    payload = load_json(str(export_path))
    assert payload['cycle_label'] == 'etf_trend_following_cycle'
    assert payload['best_candidate']['candidate_id'] == 'tsmom_ma_200_cash'
