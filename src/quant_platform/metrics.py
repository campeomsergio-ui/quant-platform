from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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
    items: dict[str, Any]


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


def _aggregate_leg_pnl(leg_pnl: dict[str, dict[str, float]]) -> dict[str, float]:
    long_total = sum(v.get("long_contribution", 0.0) for v in leg_pnl.values())
    short_total = sum(v.get("short_contribution", 0.0) for v in leg_pnl.values())
    return {
        "long_total": float(long_total),
        "short_total": float(short_total),
        "net_total": float(long_total + short_total),
        "one_sided_pnl_ratio": float(abs(long_total) / (abs(short_total) + 1e-12)) if abs(short_total) > 0 else float("inf"),
    }


def _aggregate_turnover(turnover_decomposition: dict[str, dict[str, float]]) -> dict[str, float]:
    keys = ["signal_change", "tranche_rolloff", "constraint_repair", "internal_netting_effect"]
    return {key: float(sum(day.get(key, 0.0) for day in turnover_decomposition.values())) for key in keys}


def _latest_exposure(exposure_summary: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not exposure_summary:
        return {"beta": 0.0, "sector_exposures": {}, "size_proxy": 0.0, "concentration": 0.0, "effective_breadth": 0.0}
    latest = exposure_summary[sorted(exposure_summary.keys())[-1]]
    return latest


def _aggregate_capacity(capacity_stress: dict[str, dict[str, float]]) -> dict[str, float]:
    if not capacity_stress:
        return {"baseline_cost": 0.0, "cost_1_5x": 0.0, "cost_2_0x": 0.0, "avg_adv_participation": 0.0, "max_adv_participation": 0.0}
    values = list(capacity_stress.values())
    return {
        "baseline_cost": float(sum(v.get("baseline_cost", 0.0) for v in values)),
        "cost_1_5x": float(sum(v.get("cost_1_5x", 0.0) for v in values)),
        "cost_2_0x": float(sum(v.get("cost_2_0x", 0.0) for v in values)),
        "avg_adv_participation": float(np.mean([v.get("avg_adv_participation", 0.0) for v in values])),
        "max_adv_participation": float(max(v.get("max_adv_participation", 0.0) for v in values)),
    }


def compute_diagnostics(result: BacktestResult) -> DiagnosticReport:
    leg = _aggregate_leg_pnl(result.diagnostics.get("leg_pnl", {}))
    turnover_decomp = _aggregate_turnover(result.diagnostics.get("turnover_decomposition", {}))
    exposure = _latest_exposure(result.diagnostics.get("exposure_summary", {}))
    capacity = _aggregate_capacity(result.diagnostics.get("capacity_stress", {}))
    sensitivity = result.diagnostics.get("sensitivity", {})
    items = {
        "average_turnover": float(result.turnover.mean()) if len(result.turnover) else 0.0,
        "effective_bets": float((result.net_returns.mean() / (result.net_returns.std(ddof=1) + 1e-12)) ** 2) if len(result.net_returns) else 0.0,
        "long_short_attribution": leg,
        "turnover_decomposition": turnover_decomp,
        "exposure_summary": exposure,
        "capacity_stress": capacity,
        "delay_cost_sensitivity": sensitivity,
    }
    return DiagnosticReport(items=items)
