from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_platform.io import save_json


def write_sample_daily_equities_dataset(root: str) -> None:
    base = Path(root)
    base.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range("2022-01-03", periods=15, freq="B")
    symbols = ["AAA", "BBB", "CCC", "OLD", "DDD"]
    rows: list[dict[str, object]] = []
    for di, date in enumerate(dates):
        for si, symbol in enumerate(symbols):
            open_px = 100.0 + si * 3.0 + di
            close_px = open_px * (1.0 + ((si - 2) * 0.002))
            rows.append(
                {
                    "date": date.isoformat(),
                    "symbol": symbol,
                    "open": open_px,
                    "high": open_px * 1.01,
                    "low": open_px * 0.99,
                    "close": close_px,
                    "volume": 1_000_000 + si * 100_000,
                    "adv": 50_000_000.0,
                    "daily_volatility": 0.02 + si * 0.001,
                }
            )
    metadata_rows = [
        {"symbol": "AAA", "sector": "Tech", "industry": "Software", "beta": 1.0, "market_cap": 2e9, "security_type": "COMMON", "is_primary_listing": True, "effective_from": dates[0].isoformat(), "effective_to": None},
        {"symbol": "BBB", "sector": "Tech", "industry": "Hardware", "beta": 1.1, "market_cap": 3e9, "security_type": "COMMON", "is_primary_listing": True, "effective_from": dates[0].isoformat(), "effective_to": None},
        {"symbol": "CCC", "sector": "Health", "industry": "Biotech", "beta": 0.9, "market_cap": 2.5e9, "security_type": "COMMON", "is_primary_listing": True, "effective_from": dates[0].isoformat(), "effective_to": None},
        {"symbol": "NEW", "sector": "Energy", "industry": "Energy", "beta": 1.2, "market_cap": 1.8e9, "security_type": "COMMON", "is_primary_listing": True, "effective_from": dates[8].isoformat(), "effective_to": None},
        {"symbol": "DDD", "sector": "Health", "industry": "Services", "beta": 1.05, "market_cap": 2.2e9, "security_type": "COMMON", "is_primary_listing": True, "effective_from": dates[0].isoformat(), "effective_to": None},
    ]
    mapping_rows = [
        {"raw_symbol": "OLD", "canonical_symbol": "OLD", "effective_from": dates[0].isoformat(), "effective_to": dates[7].isoformat()},
        {"raw_symbol": "OLD", "canonical_symbol": "NEW", "effective_from": dates[8].isoformat(), "effective_to": None},
        {"raw_symbol": "AAA", "canonical_symbol": "AAA", "effective_from": dates[0].isoformat(), "effective_to": None},
        {"raw_symbol": "BBB", "canonical_symbol": "BBB", "effective_from": dates[0].isoformat(), "effective_to": None},
        {"raw_symbol": "CCC", "canonical_symbol": "CCC", "effective_from": dates[0].isoformat(), "effective_to": None},
        {"raw_symbol": "DDD", "canonical_symbol": "DDD", "effective_from": dates[0].isoformat(), "effective_to": None},
    ]
    benchmark_rows = [{"date": date.isoformat(), "return": 0.001} for date in dates]
    delisting_rows = [{"symbol": "CCC", "delisting_date": dates[-1].isoformat(), "delisting_return": -0.75}]
    corporate_action_rows = [{"symbol": "BBB", "effective_date": dates[5].isoformat(), "action_type": "split", "value": 2.0}]
    save_json(str(base / "bars.json"), {"rows": rows})
    save_json(str(base / "metadata.json"), {"rows": metadata_rows})
    save_json(str(base / "symbol_mapping.json"), {"rows": mapping_rows})
    save_json(str(base / "benchmark.json"), {"rows": benchmark_rows})
    save_json(str(base / "delistings.json"), {"rows": delisting_rows})
    save_json(str(base / "corporate_actions.json"), {"rows": corporate_action_rows})
