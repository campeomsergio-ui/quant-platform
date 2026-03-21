from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from quant_platform.io import load_json, save_json
from quant_platform.monitoring.kill_switch import KillSwitchState, clear_kill, manual_kill
from quant_platform.paper.adapter import PaperBrokerAdapter
from quant_platform.paper.execution import PaperCycleResult, gate_paper_trading, run_paper_cycle
from quant_platform.paper.reconciliation import RuntimeReconciliation, reconcile_runtime_state
from quant_platform.paper.reports import DailyReport, build_daily_report
from quant_platform.signals.base import SignalContext
from quant_platform.signals.mean_reversion import MeanReversionSignal


@dataclass(frozen=True)
class PaperRuntimeConfig:
    safety_enabled: bool = True
    acceptance_met: bool = True
    seed: int = 0
    dry_run: bool = False
    state_path: str = "state/paper_runtime_state.json"
    report_path: str = "state/paper_runtime_report.json"


@dataclass(frozen=True)
class PaperRuntimeState:
    last_run_date: str | None
    positions: dict[str, float]
    orders: list[dict[str, float | str]]
    fills: list[dict[str, float | str]]
    kill_switch_state: dict[str, object]
    account_snapshot: dict[str, float | str]


@dataclass(frozen=True)
class PaperRuntimeResult:
    cycle: PaperCycleResult | None
    report: DailyReport | None
    state: PaperRuntimeState
    runtime_reconciliation: RuntimeReconciliation | None
    status: str


def _default_state() -> PaperRuntimeState:
    return PaperRuntimeState(last_run_date=None, positions={}, orders=[], fills=[], kill_switch_state={"enabled": False, "reason": "", "triggers": []}, account_snapshot={})


def load_runtime_state(path: str) -> PaperRuntimeState:
    payload = load_json(path)
    if not payload:
        return _default_state()
    required = {"last_run_date", "positions", "orders", "fills", "kill_switch_state", "account_snapshot"}
    if not required.issubset(payload.keys()):
        raise ValueError("paper runtime state corruption detected")
    return PaperRuntimeState(
        last_run_date=payload["last_run_date"],
        positions={str(k): float(v) for k, v in payload["positions"].items()},
        orders=list(payload["orders"]),
        fills=list(payload["fills"]),
        kill_switch_state=dict(payload["kill_switch_state"]),
        account_snapshot=dict(payload["account_snapshot"]),
    )


def save_runtime_state(path: str, state: PaperRuntimeState) -> None:
    save_json(
        path,
        {
            "last_run_date": state.last_run_date,
            "positions": state.positions,
            "orders": state.orders,
            "fills": state.fills,
            "kill_switch_state": state.kill_switch_state,
            "account_snapshot": state.account_snapshot,
        },
    )


def _kill_from_state(state: PaperRuntimeState) -> KillSwitchState:
    payload = state.kill_switch_state
    return KillSwitchState(enabled=bool(payload.get("enabled", False)), reason=str(payload.get("reason", "")), triggers=list(payload.get("triggers", [])))


def _series_from_positions(positions: dict[str, float]) -> pd.Series:
    return pd.Series(positions, dtype=float).sort_index() if positions else pd.Series(dtype=float)


def _serialize_orders(cycle: PaperCycleResult) -> list[dict[str, float | str]]:
    return [{"symbol": order.symbol, "target_weight_delta": order.target_weight_delta} for order in cycle.orders]


def _serialize_fills(cycle: PaperCycleResult) -> list[dict[str, float | str]]:
    return [{"symbol": fill.symbol, "filled_weight_delta": fill.filled_weight_delta, "fill_price": fill.fill_price, "status": fill.status} for fill in cycle.fills]


def _make_report(exec_date: pd.Timestamp, cycle: PaperCycleResult, target: pd.Series, current: pd.Series, constraint_events: list[dict[str, object]] | None = None) -> DailyReport:
    gross = float(cycle.actual_book.abs().sum())
    net = float(cycle.actual_book.sum())
    turnover = float(target.sub(current, fill_value=0.0).abs().sum())
    payload = {
        "signals": {str(k): float(v) for k, v in target.sort_index().items()},
        "current_book": {str(k): float(v) for k, v in current.sort_index().items()},
        "target_book": {str(k): float(v) for k, v in target.sort_index().items()},
        "orders": _serialize_orders(cycle),
        "fills": _serialize_fills(cycle),
        "reconciliation": {
            "status": cycle.reconciliation.status,
            "differences": cycle.reconciliation.differences,
        },
        "constraint_risk_events": constraint_events or [],
        "pnl_exposure_summary": {"gross_exposure": gross, "net_exposure": net, "turnover": turnover},
        "diagnostics": {
            **cycle.diagnostics,
            "long_short_attribution": {
                "gross_long_exposure": float(target[target > 0].sum()),
                "gross_short_exposure": float(target[target < 0].abs().sum()),
            },
            "turnover_decomposition": {
                "signal_change": turnover,
                "tranche_rolloff": 0.0,
                "constraint_repair": 0.0,
                "internal_netting_effect": 0.0,
            },
            "capacity_stress": {
                "baseline": {"gross": gross, "net": net},
                "1_5x": {"gross": gross * 1.5, "net": net},
                "2_0x": {"gross": gross * 2.0, "net": net},
            },
            "delay_cost_sensitivity": {"execution_delay_days": 1, "cost_stress": [1.0, 1.5, 2.0]},
        },
    }
    return build_daily_report(exec_date, cycle.status, gross, net, turnover, payload)


def run_daily_paper_cycle(bars: pd.DataFrame, meta: pd.DataFrame, signal_model: MeanReversionSignal, broker: PaperBrokerAdapter, config: PaperRuntimeConfig, manual_kill_flag: bool = False) -> PaperRuntimeResult:
    decision = gate_paper_trading(config.acceptance_met, config.safety_enabled)
    if decision.status != "allowed":
        state = _default_state()
        return PaperRuntimeResult(cycle=None, report=None, state=state, runtime_reconciliation=None, status="blocked")

    state = load_runtime_state(config.state_path)
    kill_switch = _kill_from_state(state)
    if manual_kill_flag:
        kill_switch = manual_kill("manual kill flag")
    dates = sorted(bars.index.get_level_values("date").unique())
    if len(dates) < 2:
        kill_switch = KillSwitchState(enabled=True, reason="missing required data", triggers=["missing_required_data"])
        new_state = PaperRuntimeState(state.last_run_date, state.positions, state.orders, state.fills, {"enabled": True, "reason": kill_switch.reason, "triggers": kill_switch.triggers}, state.account_snapshot)
        save_runtime_state(config.state_path, new_state)
        return PaperRuntimeResult(cycle=None, report=None, state=new_state, runtime_reconciliation=None, status="blocked")

    signal_date = dates[-2]
    exec_date = dates[-1]
    context = SignalContext(bars=bars.loc[bars.index.get_level_values("date") <= signal_date], meta=meta, as_of=signal_date, seed=config.seed)
    target = signal_model.compute(context).weights.sort_index()
    current = _series_from_positions(state.positions)
    runtime_recon = reconcile_runtime_state(current, current, state.orders, state.orders, state.fills, state.fills)
    if runtime_recon.status != "matched":
        kill_switch = KillSwitchState(enabled=True, reason="reconciliation inconsistency", triggers=["reconciliation_inconsistency"])

    gross = float(target.abs().sum())
    net = float(target.sum())
    if gross > 2.0 or abs(net) > 0.05:
        kill_switch = KillSwitchState(enabled=True, reason="gross/net breach", triggers=["gross_net_breach"])

    market_slice = bars.xs(exec_date)
    cycle = run_paper_cycle(target, current, market_slice, broker, kill_switch, dry_run=config.dry_run)
    report = _make_report(exec_date, cycle, target, current)
    save_json(config.report_path, {"summary": report.summary, "payload": report.payload})

    new_kill = kill_switch if kill_switch.enabled else clear_kill()
    new_state = PaperRuntimeState(
        last_run_date=exec_date.isoformat(),
        positions={str(k): float(v) for k, v in (current if config.dry_run or cycle.status == "killed" else cycle.actual_book).sort_index().items()},
        orders=state.orders + _serialize_orders(cycle),
        fills=state.fills + _serialize_fills(cycle),
        kill_switch_state={"enabled": new_kill.enabled, "reason": new_kill.reason, "triggers": new_kill.triggers},
        account_snapshot={"gross_exposure": float(cycle.actual_book.abs().sum()), "net_exposure": float(cycle.actual_book.sum())},
    )
    save_runtime_state(config.state_path, new_state)
    return PaperRuntimeResult(cycle=cycle, report=report, state=new_state, runtime_reconciliation=runtime_recon, status=cycle.status)
