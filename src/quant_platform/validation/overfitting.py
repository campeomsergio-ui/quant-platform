from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OverfittingConfig:
    threshold: float = 0.2
    seed: int = 0


@dataclass(frozen=True)
class OverfittingResult:
    probability_of_backtest_overfitting: float
    probability_of_loss: float
    rank_stability: float


def compute_rank_stability(results_by_fold: pd.DataFrame) -> float:
    if results_by_fold.shape[0] < 2 or results_by_fold.shape[1] < 2:
        return 0.0
    in_sample = results_by_fold.iloc[0].rank()
    out_of_sample = results_by_fold.iloc[-1].rank()
    return float(in_sample.corr(out_of_sample, method="spearman"))


def estimate_pbo(results_by_fold: pd.DataFrame, config: OverfittingConfig) -> OverfittingResult:
    in_sample_rank = results_by_fold.mean(axis=0).rank(ascending=False, method="average")
    best = in_sample_rank.idxmin()
    oos = results_by_fold.iloc[-1]
    oos_rank = oos.rank(ascending=False, method="average")
    pbo = float((oos_rank[best] > (len(oos_rank) / 2)) if len(oos_rank) else 1.0)
    p_loss = float((oos[best] < 0) if len(oos) else 1.0)
    return OverfittingResult(probability_of_backtest_overfitting=pbo, probability_of_loss=p_loss, rank_stability=compute_rank_stability(results_by_fold))
