from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from quant_platform.io import load_json, save_json


@dataclass(frozen=True)
class ValidationIssue:
    level: str
    code: str
    message: str
    context: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DataValidationReport:
    ok: bool
    issues: list[ValidationIssue]
    summary: dict[str, Any]


@dataclass(frozen=True)
class DataBundle:
    bars: pd.DataFrame
    corporate_actions: pd.DataFrame
    metadata: pd.DataFrame
    benchmark: pd.Series
    delistings: pd.DataFrame
    symbol_mapping: pd.DataFrame
    dataset_manifest: dict[str, Any] = field(default_factory=dict)
    data_quality_metadata: dict[str, Any] = field(default_factory=dict)


class DailyEquitiesDataAdapter(Protocol):
    def load_bundle(self) -> DataBundle: ...


REQUIRED_BAR_COLUMNS = {"open", "high", "low", "close", "volume"}
OPTIONAL_BAR_COLUMNS = {"adv", "daily_volatility", "shares_outstanding", "market_cap", "currency", "primary_exchange"}
REQUIRED_META_COLUMNS = {"sector", "industry", "security_type", "is_primary_listing", "symbol", "effective_from", "effective_to"}
OPTIONAL_META_COLUMNS = {"beta", "market_cap", "shares_outstanding", "country", "currency", "region", "calendar"}
REQUIRED_MAPPING_COLUMNS = {"raw_symbol", "canonical_symbol", "effective_from", "effective_to"}
REQUIRED_DELISTING_COLUMNS = {"symbol", "delisting_date", "delisting_return"}


@dataclass(frozen=True)
class LocalJsonDataAdapter:
    root: str

    def load_bundle(self) -> DataBundle:
        return LocalTableDataAdapter(self.root, preferred_format="json").load_bundle()


@dataclass(frozen=True)
class LocalTableDataAdapter:
    root: str
    preferred_format: str = "auto"

    def load_bundle(self) -> DataBundle:
        base = Path(self.root)
        bars = _load_named_frame(base, "bars", multi_index=["date", "symbol"], preferred_format=self.preferred_format)
        corporate_actions = _load_named_frame(base, "corporate_actions", preferred_format=self.preferred_format)
        metadata = _load_named_frame(base, "metadata", preferred_format=self.preferred_format)
        benchmark_df = _load_named_frame(base, "benchmark", preferred_format=self.preferred_format)
        delistings = _load_named_frame(base, "delistings", preferred_format=self.preferred_format)
        symbol_mapping = _load_named_frame(base, "symbol_mapping", preferred_format=self.preferred_format)
        manifest = _load_optional_json(base / "manifest.json")
        quality = _load_optional_json(base / "data_quality.json")
        benchmark = pd.Series(dtype=float) if benchmark_df.empty else benchmark_df.set_index("date")["return"].sort_index()
        dataset_manifest = build_dataset_manifest(bars, metadata, benchmark, delistings, symbol_mapping, existing_manifest=manifest)
        return DataBundle(
            bars=bars,
            corporate_actions=corporate_actions,
            metadata=metadata,
            benchmark=benchmark,
            delistings=delistings,
            symbol_mapping=symbol_mapping,
            dataset_manifest=dataset_manifest,
            data_quality_metadata=quality,
        )


def _load_optional_json(path: Path) -> dict[str, Any]:
    return load_json(str(path)) if path.exists() else {}


def validate_external_table_source(source_root: str, *, preferred_format: str = "auto") -> dict[str, Any]:
    source = Path(source_root)
    bars = _load_named_frame(source, "bars", preferred_format=preferred_format)
    metadata = _load_named_frame(source, "metadata", preferred_format=preferred_format)
    symbol_mapping = _load_named_frame(source, "symbol_mapping", preferred_format=preferred_format)
    benchmark = _load_named_frame(source, "benchmark", preferred_format=preferred_format)
    delistings = _load_named_frame(source, "delistings", preferred_format=preferred_format)
    corporate_actions = _load_named_frame(source, "corporate_actions", preferred_format=preferred_format)

    missing_files = [
        stem for stem, frame in {
            "bars": bars,
            "metadata": metadata,
            "symbol_mapping": symbol_mapping,
            "benchmark": benchmark,
            "delistings": delistings,
            "corporate_actions": corporate_actions,
        }.items() if frame.empty and not any((source / f"{stem}.{ext}").exists() for ext in ("csv", "json", "parquet"))
    ]
    issues: list[dict[str, Any]] = []
    if missing_files:
        issues.append({"level": "error", "code": "missing_required_files", "message": "required bundle files are missing", "context": {"files": missing_files}})
    if "date" not in bars.columns:
        issues.append({"level": "error", "code": "missing_bars_date", "message": "bars must include date column"})
    if "symbol" not in bars.columns:
        issues.append({"level": "error", "code": "missing_bars_symbol", "message": "bars must include symbol column"})
    missing_bar_cols = sorted(REQUIRED_BAR_COLUMNS.difference(bars.columns)) if not bars.empty else sorted(REQUIRED_BAR_COLUMNS)
    if missing_bar_cols:
        issues.append({"level": "error", "code": "missing_bar_columns", "message": "bars missing required columns", "context": {"columns": missing_bar_cols}})
    missing_meta_cols = sorted(REQUIRED_META_COLUMNS.difference(metadata.columns)) if not metadata.empty else sorted(REQUIRED_META_COLUMNS)
    if missing_meta_cols:
        issues.append({"level": "error", "code": "missing_metadata_columns", "message": "metadata missing required columns", "context": {"columns": missing_meta_cols}})
    missing_mapping_cols = sorted(REQUIRED_MAPPING_COLUMNS.difference(symbol_mapping.columns)) if not symbol_mapping.empty else sorted(REQUIRED_MAPPING_COLUMNS)
    if missing_mapping_cols:
        issues.append({"level": "error", "code": "missing_mapping_columns", "message": "symbol_mapping missing required columns", "context": {"columns": missing_mapping_cols}})
    benchmark_required = {"date", "return"}
    missing_benchmark_cols = sorted(benchmark_required.difference(benchmark.columns)) if not benchmark.empty else sorted(benchmark_required)
    if missing_benchmark_cols:
        issues.append({"level": "error", "code": "missing_benchmark_columns", "message": "benchmark missing required columns", "context": {"columns": missing_benchmark_cols}})
    missing_delisting_cols = sorted(REQUIRED_DELISTING_COLUMNS.difference(delistings.columns)) if not delistings.empty else []
    if missing_delisting_cols:
        issues.append({"level": "error", "code": "missing_delisting_columns", "message": "delistings missing required columns", "context": {"columns": missing_delisting_cols}})
    corp_required = {"symbol", "effective_date", "action_type", "value"}
    missing_corp_cols = sorted(corp_required.difference(corporate_actions.columns)) if not corporate_actions.empty else []
    if missing_corp_cols:
        issues.append({"level": "error", "code": "missing_corporate_action_columns", "message": "corporate_actions missing required columns", "context": {"columns": missing_corp_cols}})

    return {
        "ok": not any(issue["level"] == "error" for issue in issues),
        "issues": issues,
        "row_counts": {
            "bars": int(len(bars)),
            "metadata": int(len(metadata)),
            "symbol_mapping": int(len(symbol_mapping)),
            "benchmark": int(len(benchmark)),
            "delistings": int(len(delistings)),
            "corporate_actions": int(len(corporate_actions)),
        },
    }


def import_external_table_bundle(source_root: str, dest_root: str, *, source_name: str, notes: str = "", benchmark_name: str = "", preferred_format: str = "auto") -> dict[str, Any]:
    source = Path(source_root)
    dest = Path(dest_root)
    dest.mkdir(parents=True, exist_ok=True)

    preflight = validate_external_table_source(source_root, preferred_format=preferred_format)
    if not preflight["ok"]:
        first_error = next(issue for issue in preflight["issues"] if issue["level"] == "error")
        raise ValueError(f"{first_error['code']}: {first_error['message']}")

    bars = _load_named_frame(source, "bars", preferred_format=preferred_format)
    metadata = _load_named_frame(source, "metadata", preferred_format=preferred_format)
    symbol_mapping = _load_named_frame(source, "symbol_mapping", preferred_format=preferred_format)
    benchmark = _load_named_frame(source, "benchmark", preferred_format=preferred_format)
    delistings = _load_named_frame(source, "delistings", preferred_format=preferred_format)
    corporate_actions = _load_named_frame(source, "corporate_actions", preferred_format=preferred_format)

    bars.to_csv(dest / "bars.csv", index=False)
    metadata.to_csv(dest / "metadata.csv", index=False)
    symbol_mapping.to_csv(dest / "symbol_mapping.csv", index=False)
    benchmark.to_csv(dest / "benchmark.csv", index=False)
    delistings.to_csv(dest / "delistings.csv", index=False)
    corporate_actions.to_csv(dest / "corporate_actions.csv", index=False)

    bundle = LocalTableDataAdapter(str(dest), preferred_format="csv").load_bundle()
    report = validate_point_in_time_bundle(bundle)
    quality_payload = {
        "source": {
            "provider": source_name,
            "source_root": str(source.resolve()),
            "benchmark_name": benchmark_name,
            "import_format": preferred_format,
        },
        "dataset_notes": {
            "import_workflow": "Imported from external/manual normalized tables into repo-native bundle format.",
            "notes": notes,
            "delistings_present": int(len(delistings)),
            "corporate_actions_present": int(len(corporate_actions)),
        },
        "row_counts": {
            "bars": int(len(bars)),
            "metadata": int(len(metadata)),
            "symbol_mapping": int(len(symbol_mapping)),
            "benchmark": int(len(benchmark)),
            "delistings": int(len(delistings)),
            "corporate_actions": int(len(corporate_actions)),
        },
        "validation_summary": report.summary,
    }
    save_json(str(dest / "manifest.json"), bundle.dataset_manifest)
    save_json(str(dest / "data_quality.json"), quality_payload)
    return {
        "dest_root": str(dest),
        "manifest": bundle.dataset_manifest,
        "validation": {
            "ok": report.ok,
            "issue_count": len(report.issues),
            "issues": [
                {"level": issue.level, "code": issue.code, "message": issue.message, "context": issue.context}
                for issue in report.issues
            ],
        },
    }


def _load_named_frame(base: Path, stem: str, multi_index: list[str] | None = None, preferred_format: str = "auto") -> pd.DataFrame:
    candidates: list[Path]
    if preferred_format == "json":
        candidates = [base / f"{stem}.json"]
    elif preferred_format == "csv":
        candidates = [base / f"{stem}.csv"]
    elif preferred_format == "parquet":
        candidates = [base / f"{stem}.parquet"]
    else:
        candidates = [base / f"{stem}.parquet", base / f"{stem}.csv", base / f"{stem}.json"]
    for path in candidates:
        if path.exists():
            return _load_frame(path, multi_index=multi_index)
    return pd.DataFrame()


def _load_frame(path: Path, multi_index: list[str] | None = None) -> pd.DataFrame:
    if path.suffix == ".json":
        payload = load_json(str(path))
        frame = pd.DataFrame(payload.get("rows", []))
    elif path.suffix == ".csv":
        frame = pd.read_csv(path)
    elif path.suffix == ".parquet":
        frame = pd.read_parquet(path)
    else:
        raise ValueError(f"unsupported data file format: {path.suffix}")
    if frame.empty:
        return frame
    datetime_cols = [col for col in frame.columns if any(token in col for token in ["date", "effective_from", "effective_to", "delisting_date"])]
    for col in datetime_cols:
        frame[col] = pd.to_datetime(frame[col], errors="coerce")
    if multi_index:
        frame = frame.set_index(multi_index).sort_index()
    return frame


def _field_availability_summary(bars: pd.DataFrame, metadata: pd.DataFrame) -> dict[str, Any]:
    bar_fields = {col: float(bars[col].notna().mean()) if col in bars.columns and len(bars) else 0.0 for col in sorted(REQUIRED_BAR_COLUMNS | OPTIONAL_BAR_COLUMNS) if col in bars.columns}
    meta_fields = {col: float(metadata[col].notna().mean()) if col in metadata.columns and len(metadata) else 0.0 for col in sorted(REQUIRED_META_COLUMNS | OPTIONAL_META_COLUMNS) if col in metadata.columns}
    return {"bars": bar_fields, "metadata": meta_fields}


def build_dataset_manifest(bars: pd.DataFrame, metadata: pd.DataFrame, benchmark: pd.Series, delistings: pd.DataFrame, symbol_mapping: pd.DataFrame, existing_manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    existing_manifest = existing_manifest or {}
    if isinstance(bars.index, pd.MultiIndex) and len(bars):
        dates = bars.index.get_level_values("date")
        symbols = bars.index.get_level_values("symbol")
        symbol_counts = bars.reset_index().groupby("date")["symbol"].nunique()
        symbol_coverage = {
            "min_symbols_per_day": int(symbol_counts.min()),
            "median_symbols_per_day": float(symbol_counts.median()),
            "max_symbols_per_day": int(symbol_counts.max()),
            "coverage_stability": float(symbol_counts.min() / max(symbol_counts.max(), 1)),
        }
        date_range = {"start": str(dates.min().date()), "end": str(dates.max().date()), "trading_days": int(len(dates.unique()))}
        history_quality = "adequate_history" if len(dates.unique()) >= 252 else "short_history"
        coverage_quality = "broad" if symbol_coverage["median_symbols_per_day"] >= 100 else "narrow_or_sample"
        uneven_symbol_coverage = float(symbol_coverage["coverage_stability"]) < 0.5
        symbol_count = int(symbols.nunique())
    else:
        symbol_coverage = {"min_symbols_per_day": 0, "median_symbols_per_day": 0.0, "max_symbols_per_day": 0, "coverage_stability": 0.0}
        date_range = {"start": None, "end": None, "trading_days": 0}
        history_quality = "no_history"
        coverage_quality = "no_coverage"
        uneven_symbol_coverage = False
        symbol_count = 0
    benchmark_coverage = {
        "count": int(len(benchmark)),
        "coverage_ratio_vs_trading_days": float(len(benchmark) / max(date_range["trading_days"], 1)),
    }
    metadata_coverage = {
        "row_count": int(len(metadata)),
        "sector_coverage": float(metadata["sector"].notna().mean()) if "sector" in metadata.columns and len(metadata) else 0.0,
        "industry_coverage": float(metadata["industry"].notna().mean()) if "industry" in metadata.columns and len(metadata) else 0.0,
    }
    manifest = {
        **existing_manifest,
        "date_range": date_range,
        "symbol_count": symbol_count,
        "symbol_coverage": symbol_coverage,
        "benchmark_coverage": benchmark_coverage,
        "metadata_coverage": metadata_coverage,
        "delisting_coverage": {"count": int(len(delistings))},
        "symbol_mapping_coverage": {"count": int(len(symbol_mapping)), "raw_symbols": int(symbol_mapping["raw_symbol"].nunique()) if not symbol_mapping.empty and "raw_symbol" in symbol_mapping.columns else 0},
        "field_availability": _field_availability_summary(bars, metadata),
        "history_quality": history_quality,
        "coverage_quality": coverage_quality,
        "uneven_symbol_coverage": uneven_symbol_coverage,
    }
    return manifest


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


def _check_symbol_mapping_continuity(mapping: pd.DataFrame) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if mapping.empty:
        return issues
    for raw_symbol, group in mapping.sort_values(["raw_symbol", "effective_from"]).groupby("raw_symbol"):
        prev_end: pd.Timestamp | None = None
        for row in group.itertuples(index=False):
            if prev_end is not None and pd.notna(prev_end) and pd.notna(row.effective_from) and row.effective_from < prev_end:
                issues.append(ValidationIssue(level="error", code="mapping_overlap", message="symbol mapping intervals overlap", context={"raw_symbol": str(raw_symbol)}))
            prev_end = row.effective_to
    return issues


def validate_point_in_time_bundle(bundle: DataBundle) -> DataValidationReport:
    issues: list[ValidationIssue] = []
    bars = bundle.bars
    meta = bundle.metadata
    mapping = bundle.symbol_mapping
    delistings = bundle.delistings
    benchmark = bundle.benchmark
    manifest = bundle.dataset_manifest or build_dataset_manifest(bars, meta, benchmark, delistings, mapping)

    if not isinstance(bars.index, pd.MultiIndex) or list(bars.index.names) != ["date", "symbol"]:
        issues.append(ValidationIssue("error", "bad_bar_index", "bars must be indexed by [date, symbol]"))
    missing_bar_cols = REQUIRED_BAR_COLUMNS.difference(bars.columns)
    if missing_bar_cols:
        issues.append(ValidationIssue("error", "missing_bar_columns", "bars missing required columns", {"columns": sorted(missing_bar_cols)}))
    missing_meta_cols = REQUIRED_META_COLUMNS.difference(meta.columns)
    if missing_meta_cols:
        issues.append(ValidationIssue("error", "missing_metadata_columns", "metadata missing required columns", {"columns": sorted(missing_meta_cols)}))
    missing_mapping_cols = REQUIRED_MAPPING_COLUMNS.difference(mapping.columns)
    if missing_mapping_cols:
        issues.append(ValidationIssue("error", "missing_mapping_columns", "symbol_mapping missing required columns", {"columns": sorted(missing_mapping_cols)}))
    missing_delisting_cols = REQUIRED_DELISTING_COLUMNS.difference(delistings.columns)
    if missing_delisting_cols:
        issues.append(ValidationIssue("error", "missing_delisting_columns", "delistings missing required columns", {"columns": sorted(missing_delisting_cols)}))
    if bars.index.duplicated().any():
        issues.append(ValidationIssue("error", "duplicate_bar_rows", "bars contain duplicate [date, symbol] rows"))
    if isinstance(bars.index, pd.MultiIndex) and not bars.index.get_level_values("date").is_monotonic_increasing:
        issues.append(ValidationIssue("error", "non_monotonic_bar_dates", "bars dates must be monotonic increasing"))
    if not meta.empty and (meta["effective_to"].notna() & (meta["effective_to"] < meta["effective_from"])).any():
        issues.append(ValidationIssue("error", "invalid_metadata_ranges", "metadata has invalid effective date ranges"))
    if not mapping.empty and (mapping["effective_to"].notna() & (mapping["effective_to"] < mapping["effective_from"])).any():
        issues.append(ValidationIssue("error", "invalid_mapping_ranges", "symbol_mapping has invalid effective date ranges"))
    issues.extend(_check_symbol_mapping_continuity(mapping))
    if isinstance(bars.index, pd.MultiIndex) and not benchmark.empty:
        bar_dates = set(bars.index.get_level_values("date").unique())
        missing_benchmark = sorted(bar_dates.difference(set(benchmark.index)))
        if missing_benchmark:
            issues.append(ValidationIssue("error", "missing_benchmark_coverage", "benchmark dates do not fully cover bar dates", {"missing_count": len(missing_benchmark)}))
    if not delistings.empty and isinstance(bars.index, pd.MultiIndex):
        known_symbols = set(bars.index.get_level_values("symbol")) | set(mapping.get("canonical_symbol", pd.Series(dtype=object)))
        missing_delisting_symbols = sorted(set(delistings["symbol"].astype(str)).difference(set(map(str, known_symbols))))
        if missing_delisting_symbols:
            issues.append(ValidationIssue("warning", "unknown_delisting_symbols", "delistings contain symbols not found in bars/mapping", {"symbols": missing_delisting_symbols[:10]}))
    if manifest.get("date_range", {}).get("trading_days", 0) < 252:
        issues.append(ValidationIssue("warning", "insufficient_history", "dataset history is short for robust folds", {"trading_days": manifest.get("date_range", {}).get("trading_days", 0)}))
    if manifest.get("benchmark_coverage", {}).get("coverage_ratio_vs_trading_days", 0.0) < 1.0:
        issues.append(ValidationIssue("error", "benchmark_gap_ratio", "benchmark coverage ratio below full trading-day coverage", {"coverage_ratio": manifest.get("benchmark_coverage", {}).get("coverage_ratio_vs_trading_days", 0.0)}))
    if manifest.get("metadata_coverage", {}).get("sector_coverage", 0.0) < 0.8 or manifest.get("metadata_coverage", {}).get("industry_coverage", 0.0) < 0.8:
        issues.append(ValidationIssue("warning", "sparse_metadata_coverage", "sector/industry metadata coverage is sparse", {"metadata_coverage": manifest.get("metadata_coverage", {})}))
    if manifest.get("uneven_symbol_coverage", False):
        issues.append(ValidationIssue("warning", "uneven_symbol_coverage", "symbol coverage varies materially through time", {"symbol_coverage": manifest.get("symbol_coverage", {})}))
    if manifest.get("history_quality") == "short_history":
        issues.append(ValidationIssue("warning", "short_history_quality", "history may be too short for stable walk-forward evaluation", {"history_quality": manifest.get("history_quality")}))

    summary = {
        "bar_row_count": int(len(bars)),
        "symbol_count": int(len(bars.index.get_level_values("symbol").unique())) if isinstance(bars.index, pd.MultiIndex) and len(bars) else 0,
        "benchmark_count": int(len(benchmark)),
        "issue_count": int(len(issues)),
        "manifest": manifest,
        "quality_metadata": bundle.data_quality_metadata,
        "history_quality": manifest.get("history_quality", "unknown"),
        "coverage_quality": manifest.get("coverage_quality", "unknown"),
    }
    return DataValidationReport(ok=not any(issue.level == "error" for issue in issues), issues=issues, summary=summary)


def ensure_valid_point_in_time_bundle(bundle: DataBundle) -> DataValidationReport:
    report = validate_point_in_time_bundle(bundle)
    if not report.ok:
        first_error = next(issue for issue in report.issues if issue.level == "error")
        raise ValueError(f"{first_error.code}: {first_error.message}")
    return report


def inspect_local_dataset(root: str, preferred_format: str = "auto") -> dict[str, Any]:
    bundle = LocalTableDataAdapter(root=root, preferred_format=preferred_format).load_bundle()
    report = validate_point_in_time_bundle(bundle)
    issue_counts = {
        "errors": sum(1 for issue in report.issues if issue.level == "error"),
        "warnings": sum(1 for issue in report.issues if issue.level == "warning"),
    }
    human_summary = {
        "ok": report.ok,
        "date_range": report.summary.get("manifest", {}).get("date_range", {}),
        "history_quality": report.summary.get("history_quality"),
        "coverage_quality": report.summary.get("coverage_quality"),
        "issue_counts": issue_counts,
    }
    return {
        "manifest": bundle.dataset_manifest,
        "validation": {
            "ok": report.ok,
            "summary": report.summary,
            "issues": [
                {
                    "level": issue.level,
                    "code": issue.code,
                    "message": issue.message,
                    "context": issue.context,
                }
                for issue in report.issues
            ],
        },
        "human_summary": human_summary,
    }
