from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd

from quant_platform.io import load_json


@dataclass(frozen=True)
class DataBundle:
    bars: pd.DataFrame
    corporate_actions: pd.DataFrame
    metadata: pd.DataFrame
    benchmark: pd.Series
    delistings: pd.DataFrame
    symbol_mapping: pd.DataFrame


class DailyEquitiesDataAdapter(Protocol):
    def load_bundle(self) -> DataBundle: ...


REQUIRED_BAR_COLUMNS = {"open", "high", "low", "close", "volume", "adv", "daily_volatility"}
REQUIRED_META_COLUMNS = {"sector", "industry", "beta", "market_cap", "security_type", "is_primary_listing", "symbol", "effective_from", "effective_to"}
REQUIRED_MAPPING_COLUMNS = {"raw_symbol", "canonical_symbol", "effective_from", "effective_to"}
REQUIRED_DELISTING_COLUMNS = {"symbol", "delisting_date", "delisting_return"}


@dataclass(frozen=True)
class LocalJsonDataAdapter:
    root: str

    def load_bundle(self) -> DataBundle:
        base = Path(self.root)
        bars = _load_frame(base / "bars.json", multi_index=["date", "symbol"])
        corporate_actions = _load_frame(base / "corporate_actions.json")
        metadata = _load_frame(base / "metadata.json")
        benchmark_df = _load_frame(base / "benchmark.json")
        delistings = _load_frame(base / "delistings.json")
        symbol_mapping = _load_frame(base / "symbol_mapping.json")
        benchmark = benchmark_df.set_index("date")["return"].sort_index()
        return DataBundle(
            bars=bars,
            corporate_actions=corporate_actions,
            metadata=metadata,
            benchmark=benchmark,
            delistings=delistings,
            symbol_mapping=symbol_mapping,
        )


def _load_frame(path: Path, multi_index: list[str] | None = None) -> pd.DataFrame:
    payload = load_json(str(path))
    frame = pd.DataFrame(payload.get("rows", []))
    if frame.empty:
        return frame
    datetime_cols = [
        col
        for col in frame.columns
        if any(token in col for token in ["date", "effective_from", "effective_to", "delisting_date"])
    ]
    for col in datetime_cols:
        frame[col] = pd.to_datetime(frame[col], errors="coerce")
    if multi_index:
        frame = frame.set_index(multi_index).sort_index()
    return frame


def apply_symbol_mapping(bars: pd.DataFrame, symbol_mapping: pd.DataFrame, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    if bars.empty or symbol_mapping.empty:
        return bars
    mapping = symbol_mapping.copy()
    if as_of is not None:
        mapping = mapping.loc[(mapping["effective_from"] <= as_of) & (mapping["effective_to"].isna() | (mapping["effective_to"] >= as_of))]
    symbol_map = dict(zip(mapping["raw_symbol"], mapping["canonical_symbol"]))
    frame = bars.reset_index()
    frame["symbol"] = frame["symbol"].map(symbol_map).fillna(frame["symbol"])
    return frame.set_index(["date", "symbol"]).sort_index()


def attach_delisting_returns(returns: pd.DataFrame, delistings: pd.DataFrame) -> pd.DataFrame:
    if returns.empty or delistings.empty:
        return returns
    updated = returns.copy()
    for row in delistings.itertuples(index=False):
        delist_date = pd.Timestamp(row.delisting_date)
        symbol = str(row.symbol)
        if delist_date in updated.index and symbol in updated.columns:
            updated.loc[delist_date, symbol] = float(row.delisting_return)
    return updated


def validate_point_in_time_bundle(bundle: DataBundle) -> None:
    bars = bundle.bars
    meta = bundle.metadata
    mapping = bundle.symbol_mapping
    delistings = bundle.delistings
    if not isinstance(bars.index, pd.MultiIndex) or list(bars.index.names) != ["date", "symbol"]:
        raise ValueError("bars must be indexed by [date, symbol]")
    missing_bar_cols = REQUIRED_BAR_COLUMNS.difference(bars.columns)
    if missing_bar_cols:
        raise ValueError(f"bars missing columns: {sorted(missing_bar_cols)}")
    missing_meta_cols = REQUIRED_META_COLUMNS.difference(meta.columns)
    if missing_meta_cols:
        raise ValueError(f"metadata missing columns: {sorted(missing_meta_cols)}")
    missing_mapping_cols = REQUIRED_MAPPING_COLUMNS.difference(mapping.columns)
    if missing_mapping_cols:
        raise ValueError(f"symbol_mapping missing columns: {sorted(missing_mapping_cols)}")
    missing_delisting_cols = REQUIRED_DELISTING_COLUMNS.difference(delistings.columns)
    if missing_delisting_cols:
        raise ValueError(f"delistings missing columns: {sorted(missing_delisting_cols)}")
    if bars.index.duplicated().any():
        raise ValueError("bars contain duplicate [date, symbol] rows")
    if not bars.index.get_level_values("date").is_monotonic_increasing:
        raise ValueError("bars dates must be monotonic increasing")
    if (meta["effective_to"].notna() & (meta["effective_to"] < meta["effective_from"])).any():
        raise ValueError("metadata has invalid effective date ranges")
    if (mapping["effective_to"].notna() & (mapping["effective_to"] < mapping["effective_from"])).any():
        raise ValueError("symbol_mapping has invalid effective date ranges")
    if not set(bundle.benchmark.index).issubset(set(bars.index.get_level_values("date").unique())):
        raise ValueError("benchmark dates must be subset of bar dates")
