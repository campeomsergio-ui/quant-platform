from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from quant_platform.monitoring.kill_switch import KillSwitchState
from quant_platform.paper.adapter import PaperBrokerAdapter, PaperFill, PaperOrder
from quant_platform.paper.reconciliation import PositionReconciliation, reconcile_positions


@dataclass(frozen=True)
class PaperExecutionDecision:
    status: str
    reason: str


@dataclass(frozen=True)
class PaperCycleResult:
    status: str
    orders: list[PaperOrder]
    fills: list[PaperFill]
    reconciliation: PositionReconciliation
    actual_book: pd.Series
    diagnostics: dict[str, object] = field(default_factory=dict)


def gate_paper_trading(acceptance_met: bool, safety_enabled: bool) -> PaperExecutionDecision:
    if not safety_enabled:
        return PaperExecutionDecision(status="blocked", reason="paper trading safety flag disabled")
    if not acceptance_met:
        return PaperExecutionDecision(status="blocked", reason="acceptance criteria not met")
    return PaperExecutionDecision(status="allowed", reason="paper-only")


def run_paper_cycle(target_book: pd.Series, current_book: pd.Series, market_slice: pd.DataFrame, adapter: PaperBrokerAdapter, kill_switch: KillSwitchState, dry_run: bool = False) -> PaperCycleResult:
    if kill_switch.enabled:
        reconciliation = reconcile_positions(target_book, current_book)
        return PaperCycleResult(status="killed", orders=[], fills=[], reconciliation=reconciliation, actual_book=current_book, diagnostics={"kill_reason": kill_switch.reason, "kill_triggers": kill_switch.triggers})
    delta = target_book.sub(current_book, fill_value=0.0)
    orders = [PaperOrder(symbol=str(symbol), target_weight_delta=float(value)) for symbol, value in delta.items() if float(value) != 0.0]
    if dry_run:
        reconciliation = reconcile_positions(target_book, current_book)
        return PaperCycleResult(status="dry_run", orders=orders, fills=[], reconciliation=reconciliation, actual_book=current_book, diagnostics={"dry_run": True})
    fills = adapter.simulate_orders(orders, market_slice)
    actual = current_book.copy()
    for fill in fills:
        if fill.status == "filled":
            actual.loc[fill.symbol] = actual.get(fill.symbol, 0.0) + fill.filled_weight_delta
    reconciliation = reconcile_positions(target_book, actual)
    return PaperCycleResult(status="ok", orders=orders, fills=fills, reconciliation=reconciliation, actual_book=actual.sort_index(), diagnostics={"dry_run": False})
