from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class DailyReport:
    summary: str
    payload: dict[str, object] = field(default_factory=dict)


def build_daily_report(date: pd.Timestamp, status: str, gross: float, net: float, turnover: float, payload: dict[str, Any] | None = None) -> DailyReport:
    body = payload or {}
    concise = {
        "status": status,
        "gross": gross,
        "net": net,
        "turnover": turnover,
        "has_diagnostics": bool(body),
    }
    body.setdefault("human_summary", concise)
    return DailyReport(summary=f"date={date.date()} status={status} gross={gross:.6f} net={net:.6f} turnover={turnover:.6f}", payload=body)
