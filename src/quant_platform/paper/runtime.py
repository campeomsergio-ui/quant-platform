from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from quant_platform.backtest.engine import BacktestConfig
from quant_platform.monitoring.kill_switch import KillSwitchState
from quant_platform.paper.adapter import PaperBrokerAdapter
from quant_platform.paper.execution import PaperCycleResult, gate_paper_trading, run_paper_cycle
from quant_platform.paper.reports import DailyReport, build_daily_report
from quant_platform.signals.base import SignalContext
from quant_platform.signals.mean_reversion import MeanReversionSignal


@dataclass(frozen=True)
class PaperRuntimeConfig:
    safety_enabled: bool = True
    acceptance_met: bool = True
    seed: int = 0


@dataclass(frozen=True)
class PaperRuntimeResult:
    cycles: list[PaperCycleResult]
    reports: list[DailyReport]


def run_paper_event_loop(bars: pd.DataFrame, meta: pd.DataFrame, signal_model: MeanReversionSignal, broker: PaperBrokerAdapter, config: PaperRuntimeConfig, kill_switch: KillSwitchState) -> PaperRuntimeResult:
    decision = gate_paper_trading(config.acceptance_met, config.safety_enabled)
    if decision.status != "allowed":
        return PaperRuntimeResult(cycles=[], reports=[])
    dates = sorted(bars.index.get_level_values("date").unique())
    current_book = pd.Series(dtype=float)
    cycles: list[PaperCycleResult] = []
    reports: list[DailyReport] = []
    for idx, signal_date in enumerate(dates[:-1]):
        context = SignalContext(bars=bars.loc[bars.index.get_level_values("date") <= signal_date], meta=meta, as_of=signal_date, seed=config.seed)
        target = signal_model.compute(context).weights
        exec_date = dates[idx + 1]
        market_slice = bars.xs(exec_date)
        cycle = run_paper_cycle(target, current_book, market_slice, broker, kill_switch)
        current_book = cycle.actual_book
        intraday_returns = (market_slice["close"] / market_slice["open"] - 1.0).reindex(current_book.index).fillna(0.0)
        gross = float(current_book.mul(intraday_returns, fill_value=0.0).sum())
        net = gross
        turnover = float(target.sub(pd.Series(dtype=float) if idx == 0 else cycles[-1].actual_book, fill_value=0.0).abs().sum())
        cycles.append(cycle)
        reports.append(build_daily_report(exec_date, cycle.status, gross, net, turnover))
        if kill_switch.enabled:
            break
    return PaperRuntimeResult(cycles=cycles, reports=reports)
