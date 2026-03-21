from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quant_platform.data_contracts import PortfolioWeights, ReturnSeries
from quant_platform.signals.base import SignalContext
from quant_platform.signals.mean_reversion import ResidualModel, _residualize_day, _winsorize_cross_section, rank_signal


@dataclass(frozen=True)
class ResidualMomentumParams:
    lookback: int = 60
    skip_window: int = 10
    long_short_bucket: float = 0.10
    winsorization: float = 2.5
    execution_delay_days: int = 1
    residual_model: ResidualModel = "industry_beta_log_mcap"


@dataclass(frozen=True)
class ResidualMomentumSignal:
    params: ResidualMomentumParams

    def compute(self, context: SignalContext) -> PortfolioWeights:
        residuals = compute_residual_momentum_scores(context.bars, context.meta, self.params)
        day = residuals.values.xs(context.as_of)
        ranked = rank_signal(day, self.params.long_short_bucket)
        return PortfolioWeights(weights=-ranked.weights, as_of=ranked.as_of)


def compute_residual_momentum_scores(bars: pd.DataFrame, meta: pd.DataFrame, params: ResidualMomentumParams) -> ReturnSeries:
    closes = bars["close"].unstack("symbol").sort_index()
    raw = closes.pct_change()
    residual_rows: list[pd.Series] = []
    for date, row in raw.iterrows():
        residual = _residualize_day(row, meta, params.residual_model)
        residual = _winsorize_cross_section(residual, params.winsorization)
        residual.name = date
        residual_rows.append(residual)
    residual_frame = pd.DataFrame(residual_rows).sort_index()
    signal = residual_frame.shift(params.skip_window).rolling(params.lookback, min_periods=params.lookback).sum()
    return ReturnSeries(signal.stack())
