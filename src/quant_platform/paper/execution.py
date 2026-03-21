from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quant_platform.monitoring.kill_switch import KillSwitchState
from quant_platform.paper.adapter import PaperBrokerAdapter, PaperOrder
from quant_platform.paper.reconciliation import PositionReconciliation, reconcile_positions


@dataclass(frozen=True)
class PaperExecutionDecision:
    status: str
    reason: str


@dataclass(frozen=True)
class PaperCycleResult:
    status: str
    orders: list[PaperOrder]
    reconciliation: PositionReconciliation
    actual_book: pd.Series


def gate_paper_trading(acceptance_met: bool, safety_enabled: bool) -> PaperExecutionDecision:
    if not safety_enabled:
        return PaperExecutionDecision(status="blocked", reason="paper trading safety flag disabled")
    if not acceptance_met:
        return PaperExecutionDecision(status="blocked", reason="acceptance criteria not met")
    return PaperExecutionDecision(status="allowed", reason="paper-only")


def run_paper_cycle(target_book: pd.Series, current_book: pd.Series, market_slice: pd.DataFrame, adapter: PaperBrokerAdapter, kill_switch: KillSwitchState) -> PaperCycleResult:
    if kill_switch.enabled:
        reconciliation = reconcile_positions(target_book, current_book)
        return PaperCycleResult(status="killed", orders=[], reconciliation=reconciliation, actual_book=current_book)
    delta = target_book.sub(current_book, fill_value=0.0)
    orders = [PaperOrder(symbol=str(symbol), target_weight_delta=float(value)) for symbol, value in delta.items() if float(value) != 0.0]
    fills = adapter.simulate_orders(orders, market_slice)
    actual = current_book.copy()
    for fill in fills:
        if fill.status == "filled":
            actual.loc[fill.symbol] = actual.get(fill.symbol, 0.0) + fill.filled_weight_delta
    reconciliation = reconcile_positions(target_book, actual)
    return PaperCycleResult(status="ok", orders=orders, reconciliation=reconciliation, actual_book=actual.sort_index())
