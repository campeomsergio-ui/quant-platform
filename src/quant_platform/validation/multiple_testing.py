from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RealityCheckConfig:
    bootstrap_iterations: int = 250
    block_probability: float = 0.1
    seed: int = 0
    minHistory: int = 2


@dataclass(frozen=True)
class RealityCheckResult:
    observed_statistic: float
    adjusted_p_value: float
    bootstrap_max_statistics: list[float]
    differential_summary: dict[str, dict[str, float]] | None = None


def stationary_bootstrap(returns: pd.Series, config: RealityCheckConfig) -> list[pd.Series]:
    rng = np.random.default_rng(config.seed)
    samples: list[pd.Series] = []
    n = len(returns)
    for _ in range(config.bootstrap_iterations):
        idx = []
        while len(idx) < n:
            start = int(rng.integers(0, n))
            length = 1
            while len(idx) < n and rng.random() > config.block_probability:
                idx.append((start + length - 1) % n)
                length += 1
            if len(idx) < n:
                idx.append(start)
        samples.append(pd.Series(returns.to_numpy()[idx[:n]], index=returns.index))
    return samples


def normalize_return_series(series: pd.Series, min_history: int = 2) -> pd.Series:
    clean = series.astype(float).sort_index().dropna()
    if len(clean) >= min_history:
        return clean
    if clean.empty:
        clean = pd.Series([0.0] * min_history, index=pd.RangeIndex(min_history), dtype=float)
    while len(clean) < min_history:
        clean = pd.concat([clean, pd.Series([float(clean.iloc[-1])], index=[clean.index[-1] + 1 if hasattr(clean.index[-1], "__add__") else len(clean)], dtype=float)])
    return clean


def build_candidate_differentials(candidate_returns: dict[str, pd.Series], benchmark_returns: pd.Series, min_history: int = 2) -> dict[str, pd.Series]:
    aligned_benchmark = normalize_return_series(benchmark_returns, min_history=min_history)
    differentials: dict[str, pd.Series] = {}
    for key, series in candidate_returns.items():
        aligned_candidate = normalize_return_series(series, min_history=min_history)
        candidate_aligned, benchmark_aligned = aligned_candidate.align(aligned_benchmark, join="inner")
        if len(candidate_aligned) < min_history:
            candidate_aligned = normalize_return_series(candidate_aligned, min_history=min_history)
            benchmark_aligned = normalize_return_series(benchmark_aligned, min_history=min_history)
            candidate_aligned, benchmark_aligned = candidate_aligned.align(benchmark_aligned, join="inner")
        differentials[key] = candidate_aligned.sub(benchmark_aligned, fill_value=0.0)
    return differentials


def summarize_differentials(differentials: dict[str, pd.Series]) -> dict[str, dict[str, float]]:
    return {
        key: {
            "mean": float(series.mean()) if len(series) else 0.0,
            "std": float(series.std(ddof=1)) if len(series) > 1 else 0.0,
            "count": float(len(series)),
            "cumulative_excess": float(series.sum()) if len(series) else 0.0,
        }
        for key, series in differentials.items()
    }


def run_white_reality_check(candidate_returns: dict[str, pd.Series], benchmark_returns: pd.Series, config: RealityCheckConfig) -> RealityCheckResult:
    differentials = build_candidate_differentials(candidate_returns, benchmark_returns, min_history=config.minHistory)
    observed = max((series.mean() / (series.std(ddof=1) + 1e-12) for series in differentials.values()), default=0.0)
    bootstrap_max = []
    centered = {k: v - v.mean() for k, v in differentials.items()}
    for i in range(config.bootstrap_iterations):
        rng_cfg = RealityCheckConfig(bootstrap_iterations=1, block_probability=config.block_probability, seed=config.seed + i + 1, minHistory=config.minHistory)
        stats = []
        for series in centered.values():
            sample = stationary_bootstrap(series, rng_cfg)[0]
            stats.append(sample.mean() / (sample.std(ddof=1) + 1e-12))
        bootstrap_max.append(max(stats, default=0.0))
    adjusted = float(sum(value >= observed for value in bootstrap_max) / max(len(bootstrap_max), 1))
    return RealityCheckResult(observed_statistic=float(observed), adjusted_p_value=adjusted, bootstrap_max_statistics=bootstrap_max, differential_summary=summarize_differentials(differentials))
