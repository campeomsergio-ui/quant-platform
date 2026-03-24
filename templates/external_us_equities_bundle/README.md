# External US Equities Bundle Template

Use this template when supplying a better external/manual dataset for the current stock anchor baseline and stock residual momentum sleeve.

## Required files

Place these files in one source directory:

- `bars.csv` (or `.json` / `.parquet`)
- `metadata.csv` (or `.json` / `.parquet`)
- `symbol_mapping.csv` (or `.json` / `.parquet`)
- `benchmark.csv` (or `.json` / `.parquet`)
- `delistings.csv` (or `.json` / `.parquet`)
- `corporate_actions.csv` (or `.json` / `.parquet`)

CSV is the simplest path.

## Required columns

### bars.csv
Required:
- `date`
- `symbol`
- `open`
- `high`
- `low`
- `close`
- `volume`

Useful optional columns:
- `adv`
- `daily_volatility`
- `shares_outstanding`
- `market_cap`
- `currency`
- `primary_exchange`

### metadata.csv
Required:
- `symbol`
- `sector`
- `industry`
- `security_type`
- `is_primary_listing`
- `effective_from`
- `effective_to`

Useful optional columns:
- `beta`
- `market_cap`
- `shares_outstanding`
- `country`
- `currency`
- `region`
- `calendar`

### symbol_mapping.csv
Required:
- `raw_symbol`
- `canonical_symbol`
- `effective_from`
- `effective_to`

### benchmark.csv
Required:
- `date`
- `return`

Recommended:
- daily total-return series for the benchmark if available

### delistings.csv
Required:
- `symbol`
- `delisting_date`
- `delisting_return`

If none exist, provide an empty file with headers only.

### corporate_actions.csv
Expected columns:
- `symbol`
- `effective_date`
- `action_type`
- `value`

If none exist, provide an empty file with headers only.

## Minimal coverage expectations

For honest stock-sleeve evaluation, the bundle should ideally provide:
- at least ~252 trading days minimum
- much better if 3y+ of daily history
- benchmark coverage for every bar date
- sector and industry metadata coverage close to complete
- symbol mapping intervals without overlaps

## Validation workflow

1. Preflight the raw external bundle:

```bash
PYTHONPATH=src .venv/bin/python -m quant_platform.cli validate-external-dataset \
  --source-root /path/to/vendor_bundle \
  --preferred-format csv
```

2. Import into repo-native format:

```bash
PYTHONPATH=src .venv/bin/python -m quant_platform.cli import-external-dataset \
  --source-root /path/to/vendor_bundle \
  --dest-root data/external_us_equities_vendor_bundle \
  --source-name manual_vendor_bundle \
  --benchmark-name SPXTR \
  --preferred-format csv \
  --notes "manual survivorship-aware vendor export"
```

3. Inspect imported bundle:

```bash
PYTHONPATH=src .venv/bin/python -m quant_platform.cli inspect-data-local \
  --data-root data/external_us_equities_vendor_bundle \
  --preferred-format csv
```

## What gets generated during import

The repo-native destination bundle will contain:
- normalized `*.csv` tables
- `manifest.json`
- `data_quality.json`

That imported bundle can then be used by the stock baseline and stock residual research flows.
