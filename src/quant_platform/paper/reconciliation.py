from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class PositionReconciliation:
    status: str
    differences: list[str]


def reconcile_positions(target: pd.Series, actual: pd.Series, tolerance: float = 1e-9) -> PositionReconciliation:
    delta = target.sub(actual, fill_value=0.0)
    diffs = [f"{symbol}:{float(value):.8f}" for symbol, value in delta.items() if abs(float(value)) > tolerance]
    return PositionReconciliation(status="matched" if not diffs else "mismatch", differences=diffs)
