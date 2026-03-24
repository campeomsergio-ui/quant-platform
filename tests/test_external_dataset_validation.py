from pathlib import Path

import pandas as pd
import pytest

from cli import main
from quant_platform.data_access import validate_external_table_source


def _write_valid_bundle(root: Path) -> None:
    dates = pd.date_range('2023-01-03', periods=3, freq='B')
    pd.DataFrame([
        {'date': d.isoformat(), 'symbol': s, 'open': 10.0, 'high': 11.0, 'low': 9.0, 'close': 10.5, 'volume': 1000}
        for d in dates for s in ['AAA', 'BBB']
    ]).to_csv(root / 'bars.csv', index=False)
    pd.DataFrame([
        {'symbol': 'AAA', 'sector': 'Tech', 'industry': 'Software', 'security_type': 'common_stock', 'is_primary_listing': True, 'effective_from': dates[0].isoformat(), 'effective_to': ''},
        {'symbol': 'BBB', 'sector': 'Health', 'industry': 'Biotech', 'security_type': 'common_stock', 'is_primary_listing': True, 'effective_from': dates[0].isoformat(), 'effective_to': ''},
    ]).to_csv(root / 'metadata.csv', index=False)
    pd.DataFrame([
        {'raw_symbol': 'AAA', 'canonical_symbol': 'AAA', 'effective_from': dates[0].isoformat(), 'effective_to': ''},
        {'raw_symbol': 'BBB', 'canonical_symbol': 'BBB', 'effective_from': dates[0].isoformat(), 'effective_to': ''},
    ]).to_csv(root / 'symbol_mapping.csv', index=False)
    pd.DataFrame([{'date': d.isoformat(), 'return': 0.001} for d in dates]).to_csv(root / 'benchmark.csv', index=False)
    pd.DataFrame(columns=['symbol', 'delisting_date', 'delisting_return']).to_csv(root / 'delistings.csv', index=False)
    pd.DataFrame(columns=['symbol', 'effective_date', 'action_type', 'value']).to_csv(root / 'corporate_actions.csv', index=False)


def test_validate_external_dataset_ok(tmp_path: Path) -> None:
    root = tmp_path / 'bundle'
    root.mkdir()
    _write_valid_bundle(root)
    payload = validate_external_table_source(str(root), preferred_format='csv')
    assert payload['ok'] is True
    assert payload['row_counts']['bars'] == 6


def test_validate_external_dataset_catches_missing_columns(tmp_path: Path) -> None:
    root = tmp_path / 'bundle'
    root.mkdir()
    _write_valid_bundle(root)
    pd.read_csv(root / 'bars.csv').drop(columns=['close']).to_csv(root / 'bars.csv', index=False)
    payload = validate_external_table_source(str(root), preferred_format='csv')
    assert payload['ok'] is False
    assert any(issue['code'] == 'missing_bar_columns' for issue in payload['issues'])


def test_import_external_dataset_fails_fast_on_invalid_source(tmp_path: Path) -> None:
    root = tmp_path / 'bundle'
    root.mkdir()
    _write_valid_bundle(root)
    pd.read_csv(root / 'metadata.csv').drop(columns=['industry']).to_csv(root / 'metadata.csv', index=False)
    with pytest.raises(ValueError):
        main([
            'import-external-dataset',
            '--source-root', str(root),
            '--dest-root', str(tmp_path / 'dest'),
            '--source-name', 'broken_vendor',
            '--preferred-format', 'csv',
        ])
