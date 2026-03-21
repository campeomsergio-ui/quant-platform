from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quant_platform.data_contracts import PortfolioWeights, ReturnSeries


@dataclass(frozen=True)
class RiskSnapshot:
    volatility: pd.Series
    beta: pd.Series
    exposures: pd.DataFrame


@dataclass(frozen=True)
class RiskEstimator:
    volatility_lookback: int = 60
    beta_lookback: int = 60


def estimate_volatility(returns: pd.DataFrame, lookback: int) -> ReturnSeries:
    vol = returns.ewm(span=lookback, min_periods=lookback).std().iloc[-1]
    return ReturnSeries(vol)


def estimate_beta(returns: pd.DataFrame, benchmark: pd.Series) -> ReturnSeries:
    aligned = returns.align(benchmark, join="inner", axis=0)
    asset_returns, bench = aligned
    betas = {}
    bench_var = bench.var()
    for col in asset_returns.columns:
        betas[col] = asset_returns[col].cov(bench) / bench_var if bench_var else 0.0
    return ReturnSeries(pd.Series(betas, dtype=float))


def neutralize_exposures(weights: PortfolioWeights, exposures: pd.DataFrame) -> PortfolioWeights:
    adjusted = weights.weights.copy()
    if "beta" in exposures.columns and adjusted.abs().sum() > 0:
        beta = exposures["beta"].reindex(adjusted.index).fillna(0.0)
        hedge = (adjusted * beta).sum() / max((beta**2).sum(), 1e-12)
        adjusted = adjusted - hedge * beta
    gross = adjusted.abs().sum()
    if gross > 0:
        adjusted = adjusted / gross
    return PortfolioWeights(weights=adjusted, as_of=weights.as_of)
