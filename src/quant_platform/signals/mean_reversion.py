from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from quant_platform.data_contracts import PortfolioWeights, ReturnSeries
from quant_platform.signals.base import SignalContext

ResidualModel = Literal[
    "sector_only",
    "industry_only",
    "industry_beta",
    "industry_beta_log_mcap",
]


@dataclass(frozen=True)
class MeanReversionParams:
    residual_lookback: int = 5
    volatility_lookback: int = 60
    long_short_bucket: float = 0.10
    winsorization: float = 2.5
    execution_delay_days: int = 1
    residual_model: ResidualModel = "industry_beta_log_mcap"


@dataclass(frozen=True)
class MeanReversionSignal:
    params: MeanReversionParams

    def compute(self, context: SignalContext) -> PortfolioWeights:
        residuals = compute_residual_returns(context.bars, context.meta, self.params)
        signal_date = context.as_of
        day = residuals.values.xs(signal_date)
        return rank_signal(day, self.params.long_short_bucket)


def _winsorize_cross_section(values: pd.Series, winsorization: float) -> pd.Series:
    clean = values.dropna()
    if clean.empty:
        return values
    mean = clean.mean()
    std = clean.std(ddof=1)
    if pd.isna(std) or std == 0:
        return values.fillna(mean)
    lower = mean - winsorization * std
    upper = mean + winsorization * std
    return values.clip(lower=lower, upper=upper)


def _build_design_matrix(meta: pd.DataFrame, symbols: list[str], model: ResidualModel) -> pd.DataFrame:
    local = meta.reindex(symbols).copy()
    parts: list[pd.DataFrame] = []
    if model == "sector_only":
        parts.append(pd.get_dummies(local["sector"], prefix="sector", dummy_na=False, dtype=float))
    else:
        parts.append(pd.get_dummies(local["industry"], prefix="industry", dummy_na=False, dtype=float))
    if model in {"industry_beta", "industry_beta_log_mcap"}:
        parts.append(local[["beta"]].astype(float).rename(columns={"beta": "beta"}))
    if model == "industry_beta_log_mcap":
        parts.append(np.log(local[["market_cap"]].clip(lower=1.0)).rename(columns={"market_cap": "log_market_cap"}))
    x = pd.concat(parts, axis=1)
    x = x.loc[:, x.notna().any(axis=0)]
    x.insert(0, "intercept", 1.0)
    return x.fillna(0.0)


def _residualize_day(response: pd.Series, meta: pd.DataFrame, model: ResidualModel) -> pd.Series:
    clean = response.dropna()
    if clean.empty:
        return response * np.nan
    x = _build_design_matrix(meta, clean.index.astype(str).tolist(), model)
    y = clean.astype(float)
    common = y.index.intersection(x.index)
    x = x.reindex(common)
    y = y.reindex(common)
    if len(common) <= x.shape[1]:
        demeaned = y - y.mean()
        return demeaned.reindex(response.index)
    beta, *_ = np.linalg.lstsq(x.to_numpy(dtype=float), y.to_numpy(dtype=float), rcond=None)
    fitted = pd.Series(x.to_numpy(dtype=float) @ beta, index=common, dtype=float)
    residual = (y - fitted).reindex(response.index)
    return residual


def compute_residual_returns(bars: pd.DataFrame, meta: pd.DataFrame, params: MeanReversionParams) -> ReturnSeries:
    closes = bars["close"].unstack("symbol").sort_index()
    raw = closes.pct_change(params.residual_lookback)
    residual_rows: list[pd.Series] = []
    for date, row in raw.iterrows():
        residual = _residualize_day(row, meta, params.residual_model)
        residual = _winsorize_cross_section(residual, params.winsorization)
        residual.name = date
        residual_rows.append(residual)
    residual_frame = pd.DataFrame(residual_rows).sort_index()
    return ReturnSeries(residual_frame.stack())


def rank_signal(values: pd.Series, bucket: float = 0.10) -> PortfolioWeights:
    clean = values.dropna().sort_values()
    n_bucket = max(1, int(np.floor(len(clean) * bucket)))
    longs = clean.iloc[:n_bucket].index
    shorts = clean.iloc[-n_bucket:].index
    weights = pd.Series(0.0, index=clean.index, dtype=float)
    if len(longs) > 0:
        weights.loc[longs] = 1.0 / len(longs)
    if len(shorts) > 0:
        weights.loc[shorts] = -1.0 / len(shorts)
    return PortfolioWeights(weights=weights)
