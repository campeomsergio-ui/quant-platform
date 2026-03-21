from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Mapping, Sequence

import pandas as pd


@dataclass(frozen=True)
class BarData:
    frame: pd.DataFrame


@dataclass(frozen=True)
class CorporateAction:
    symbol: str
    effective_date: date
    action_type: str
    value: float


@dataclass(frozen=True)
class SecurityMeta:
    frame: pd.DataFrame


@dataclass(frozen=True)
class PortfolioWeights:
    weights: pd.Series
    as_of: pd.Timestamp | None = None


@dataclass(frozen=True)
class ReturnSeries:
    values: pd.Series


@dataclass(frozen=True)
class WalkForwardFold:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    fold_id: str


@dataclass(frozen=True)
class CandidateConfig:
    candidate_id: str
    params: Mapping[str, float | int | str]


@dataclass(frozen=True)
class BacktestResult:
    candidate: CandidateConfig
    gross_returns: pd.Series
    net_returns: pd.Series
    turnover: pd.Series
    costs: pd.Series
    diagnostics: Mapping[str, float | int | str | Sequence[str]] = field(default_factory=dict)
