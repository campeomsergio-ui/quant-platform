from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quant_platform.data_contracts import BacktestResult, CandidateConfig, WalkForwardFold


@dataclass(frozen=True)
class WalkForwardConfig:
    train_years: int = 5
    validation_years: int = 1
    step_years: int = 1
    final_test_start: str = "2023-01-01"
    final_test_end: str = "2025-12-31"


@dataclass(frozen=True)
class WalkForwardRunner:
    config: WalkForwardConfig


def generate_folds(dates: pd.DatetimeIndex, config: WalkForwardConfig) -> list[WalkForwardFold]:
    years = sorted({ts.year for ts in dates if ts.year <= 2022})
    folds: list[WalkForwardFold] = []
    for start_idx in range(0, len(years) - config.train_years - config.validation_years + 1, config.step_years):
        train_years = years[start_idx:start_idx + config.train_years]
        val_years = years[start_idx + config.train_years:start_idx + config.train_years + config.validation_years]
        folds.append(
            WalkForwardFold(
                train_start=pd.Timestamp(f"{train_years[0]}-01-01"),
                train_end=pd.Timestamp(f"{train_years[-1]}-12-31"),
                validation_start=pd.Timestamp(f"{val_years[0]}-01-01"),
                validation_end=pd.Timestamp(f"{val_years[-1]}-12-31"),
                fold_id=f"train_{train_years[0]}_{train_years[-1]}__val_{val_years[0]}_{val_years[-1]}",
            )
        )
    return folds


def run_walk_forward(candidates: list[CandidateConfig], data: dict[str, pd.DataFrame], config: WalkForwardConfig) -> list[BacktestResult]:
    folds = generate_folds(pd.DatetimeIndex(data["dates"]), config)
    results: list[BacktestResult] = []
    for candidate in candidates:
        for fold in folds:
            series = pd.Series(0.0, index=pd.date_range(fold.validation_start, fold.validation_end, freq="B"))
            results.append(BacktestResult(candidate=candidate, gross_returns=series, net_returns=series, turnover=series, costs=series, diagnostics={"fold_id": fold.fold_id}))
    return results
