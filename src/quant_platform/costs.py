from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class CostEstimate:
    per_name_cost: pd.Series
    total_cost: float


@dataclass(frozen=True)
class TransactionCostModel:
    commission_bps: float = 0.5
    slippage_floor_bps: float = 5.0
    impact_coeff: float = 0.1
    borrow_annual_bps: float = 25.0


def estimate_costs(trades: pd.DataFrame, market_data: pd.DataFrame) -> CostEstimate:
    overlap = set(trades.columns).intersection(set(market_data.columns))
    aligned = trades.copy() if overlap == set(market_data.columns) else trades.join(market_data, how="left", rsuffix="_md")
    if "adv" not in aligned.columns and "adv_md" in aligned.columns:
        aligned["adv"] = aligned["adv_md"]
    if "daily_volatility" not in aligned.columns and "daily_volatility_md" in aligned.columns:
        aligned["daily_volatility"] = aligned["daily_volatility_md"]
    participation = (aligned["trade_notional"].abs() / aligned["adv"].replace(0, pd.NA)).fillna(0.0)
    slippage_bps = pd.concat([
        pd.Series(5.0, index=aligned.index),
        10_000 * 0.1 * aligned["daily_volatility"].fillna(0.0) * participation,
    ], axis=1).max(axis=1)
    commission = aligned["trade_notional"].abs() * (0.5 / 10_000)
    slippage = aligned["trade_notional"].abs() * (slippage_bps / 10_000)
    impact = aligned["trade_notional"].abs() * participation * (0.1 / 10_000)
    borrow = aligned["short_notional"].clip(lower=0.0).fillna(0.0) * ((25.0 / 10_000) / 252.0)
    cost = commission + slippage + impact + borrow
    return CostEstimate(per_name_cost=cost, total_cost=float(cost.sum()))


def stress_costs(costs: CostEstimate, multiplier: float) -> CostEstimate:
    stressed = costs.per_name_cost * multiplier
    return CostEstimate(per_name_cost=stressed, total_cost=float(stressed.sum()))
