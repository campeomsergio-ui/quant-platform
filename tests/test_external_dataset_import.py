from pathlib import Path

import pandas as pd

from cli import main
from quant_platform.io import load_json


def test_import_external_dataset_cli(tmp_path: Path) -> None:
    src = tmp_path / 'vendor_bundle'
    src.mkdir(parents=True, exist_ok=True)

    dates = pd.date_range('2023-01-03', periods=5, freq='B')
    bars = pd.DataFrame([
        {'date': d.isoformat(), 'symbol': s, 'open': 10.0, 'high': 11.0, 'low': 9.0, 'close': 10.5, 'volume': 1000, 'adv': 10000.0, 'daily_volatility': 0.02}
        for d in dates for s in ['AAA', 'BBB']
    ])
    metadata = pd.DataFrame([
        {'symbol': 'AAA', 'sector': 'Tech', 'industry': 'Software', 'security_type': 'common_stock', 'is_primary_listing': True, 'effective_from': dates[0].isoformat(), 'effective_to': '', 'beta': 1.0, 'market_cap': 1e9, 'country': 'US', 'currency': 'USD', 'region': 'North America'},
        {'symbol': 'BBB', 'sector': 'Health', 'industry': 'Biotech', 'security_type': 'common_stock', 'is_primary_listing': True, 'effective_from': dates[0].isoformat(), 'effective_to': '', 'beta': 0.9, 'market_cap': 2e9, 'country': 'US', 'currency': 'USD', 'region': 'North America'},
    ])
    mapping = pd.DataFrame([
        {'raw_symbol': 'AAA', 'canonical_symbol': 'AAA', 'effective_from': dates[0].isoformat(), 'effective_to': ''},
        {'raw_symbol': 'BBB', 'canonical_symbol': 'BBB', 'effective_from': dates[0].isoformat(), 'effective_to': ''},
    ])
    benchmark = pd.DataFrame([{'date': d.isoformat(), 'return': 0.001} for d in dates])
    delistings = pd.DataFrame(columns=['symbol', 'delisting_date', 'delisting_return'])
    corporate_actions = pd.DataFrame(columns=['symbol', 'effective_date', 'action_type', 'value'])

    bars.to_csv(src / 'bars.csv', index=False)
    metadata.to_csv(src / 'metadata.csv', index=False)
    mapping.to_csv(src / 'symbol_mapping.csv', index=False)
    benchmark.to_csv(src / 'benchmark.csv', index=False)
    delistings.to_csv(src / 'delistings.csv', index=False)
    corporate_actions.to_csv(src / 'corporate_actions.csv', index=False)

    dest = tmp_path / 'imported'
    rc = main([
        'import-external-dataset',
        '--source-root', str(src),
        '--dest-root', str(dest),
        '--source-name', 'manual_vendor_bundle',
        '--benchmark-name', 'SPXTR',
        '--preferred-format', 'csv',
        '--notes', 'manual survivorship-aware vendor export',
    ])
    assert rc == 0
    manifest = load_json(str(dest / 'manifest.json'))
    quality = load_json(str(dest / 'data_quality.json'))
    assert manifest['symbol_count'] == 2
    assert quality['source']['provider'] == 'manual_vendor_bundle'
    assert quality['source']['benchmark_name'] == 'SPXTR'
    assert quality['dataset_notes']['delistings_present'] == 0
