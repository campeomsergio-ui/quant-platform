from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant_platform.data_contracts import BacktestResult


@dataclass(frozen=True)
class PerformanceMetrics:
    net_sharpe: float
    annualized_return: float
    max_drawdown: float
    t_stat: float


@dataclass(frozen=True)
class DiagnosticReport:
    items: dict[str, float | str]


def compute_primary_metrics(returns: pd.Series, costs: pd.Series) -> PerformanceMetrics:
    net = returns.sub(costs, fill_value=0.0)
    mean = float(net.mean())
    std = float(net.std(ddof=1))
    sharpe = mean / (std + 1e-12) * np.sqrt(252.0)
    annualized = mean * 252.0
    equity = (1.0 + net.fillna(0.0)).cumprod()
    drawdown = (equity / equity.cummax() - 1.0).min() if not equity.empty else 0.0
    t_stat = mean / ((std + 1e-12) / np.sqrt(max(len(net), 1)))
    return PerformanceMetrics(net_sharpe=float(sharpe), annualized_return=float(annualized), max_drawdown=float(abs(drawdown)), t_stat=float(t_stat))


def compute_diagnostics(result: BacktestResult) -> DiagnosticReport:
    items = {
        "average_turnover": float(result.turnover.mean()) if len(result.turnover) else 0.0,
        "effective_bets": float((result.net_returns.mean() / (result.net_returns.std(ddof=1) + 1e-12)) ** 2) if len(result.net_returns) else 0.0,
        "factor_exposure_status": "placeholder",
    }
    return DiagnosticReport(items=items)
