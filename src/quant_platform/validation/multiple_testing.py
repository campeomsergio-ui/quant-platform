from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RealityCheckConfig:
    bootstrap_iterations: int = 250
    block_probability: float = 0.1
    seed: int = 0


@dataclass(frozen=True)
class RealityCheckResult:
    observed_statistic: float
    adjusted_p_value: float
    bootstrap_max_statistics: list[float]


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


def run_white_reality_check(candidate_returns: dict[str, pd.Series], benchmark_returns: pd.Series, config: RealityCheckConfig) -> RealityCheckResult:
    differentials = {k: v.sub(benchmark_returns, fill_value=0.0) for k, v in candidate_returns.items()}
    observed = max((series.mean() / (series.std(ddof=1) + 1e-12) for series in differentials.values()), default=0.0)
    bootstrap_max = []
    centered = {k: v - v.mean() for k, v in differentials.items()}
    for i in range(config.bootstrap_iterations):
        rng_cfg = RealityCheckConfig(bootstrap_iterations=1, block_probability=config.block_probability, seed=config.seed + i + 1)
        stats = []
        for series in centered.values():
            sample = stationary_bootstrap(series, rng_cfg)[0]
            stats.append(sample.mean() / (sample.std(ddof=1) + 1e-12))
        bootstrap_max.append(max(stats, default=0.0))
    adjusted = float(sum(value >= observed for value in bootstrap_max) / max(len(bootstrap_max), 1))
    return RealityCheckResult(observed_statistic=float(observed), adjusted_p_value=adjusted, bootstrap_max_statistics=bootstrap_max)
