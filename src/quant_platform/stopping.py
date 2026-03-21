from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StopDecision:
    should_stop: bool
    reasons: list[str]


def evaluate_stop_rules(plan: Any, results: dict[str, Any]) -> StopDecision:
    reasons: list[str] = []
    if results.get("final_test_touched", False):
        reasons.append("locked test touched once")
    if results.get("adjusted_p_value", 0.0) > 0.05:
        reasons.append("white-style adjusted p-value above threshold")
    if results.get("pbo", 0.0) > 0.2:
        reasons.append("probability of backtest overfitting above threshold")
    if results.get("cost_stress_sharpe_drop", 0.0) > 0.25:
        reasons.append("out-of-sample sharpe degraded more than 25% under cost stress")
    if results.get("requires_search_space_expansion", False):
        reasons.append("new research cycle required")
    return StopDecision(should_stop=bool(reasons), reasons=reasons)
