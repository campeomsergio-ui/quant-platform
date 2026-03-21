from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class DailyReport:
    summary: str


def build_daily_report(date: pd.Timestamp, status: str, gross: float, net: float, turnover: float) -> DailyReport:
    return DailyReport(summary=f"date={date.date()} status={status} gross={gross:.6f} net={net:.6f} turnover={turnover:.6f}")
