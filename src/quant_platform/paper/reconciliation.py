from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class PositionReconciliation:
    status: str
    differences: list[str]
    expected_book: dict[str, float] = field(default_factory=dict)
    actual_book: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeReconciliation:
    status: str
    position_check: PositionReconciliation
    order_mismatches: list[str]
    fill_mismatches: list[str]


def reconcile_positions(target: pd.Series, actual: pd.Series, tolerance: float = 1e-9) -> PositionReconciliation:
    delta = target.sub(actual, fill_value=0.0).sort_index()
    diffs = [f"{symbol}:{float(value):.8f}" for symbol, value in delta.items() if abs(float(value)) > tolerance]
    return PositionReconciliation(
        status="matched" if not diffs else "mismatch",
        differences=diffs,
        expected_book={str(k): float(v) for k, v in target.sort_index().items()},
        actual_book={str(k): float(v) for k, v in actual.sort_index().items()},
    )


def reconcile_runtime_state(expected_positions: pd.Series, stored_positions: pd.Series, expected_orders: list[dict[str, float | str]], stored_orders: list[dict[str, float | str]], expected_fills: list[dict[str, float | str]], stored_fills: list[dict[str, float | str]]) -> RuntimeReconciliation:
    position_check = reconcile_positions(expected_positions, stored_positions)
    order_mismatches = [] if expected_orders == stored_orders else ["order_history_mismatch"]
    fill_mismatches = [] if expected_fills == stored_fills else ["fill_history_mismatch"]
    status = "matched" if position_check.status == "matched" and not order_mismatches and not fill_mismatches else "mismatch"
    return RuntimeReconciliation(status=status, position_check=position_check, order_mismatches=order_mismatches, fill_mismatches=fill_mismatches)
