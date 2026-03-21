from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class RebalanceScheduler:
    rule: str = "daily_close_signal_open_next_day_execution"


def generate_rebalance_dates(calendar: pd.DatetimeIndex, rule: str) -> list[pd.Timestamp]:
    if rule == "daily":
        return list(calendar)
    if rule == "weekly":
        return list(calendar[calendar.weekday == 4])
    return list(calendar)


def generate_orders(current_weights: pd.Series, target_weights: pd.Series) -> list[dict[str, float | str]]:
    delta = target_weights.sub(current_weights, fill_value=0.0)
    delta = delta[delta != 0.0]
    return [{"symbol": str(symbol), "target_delta": float(value)} for symbol, value in delta.items()]
