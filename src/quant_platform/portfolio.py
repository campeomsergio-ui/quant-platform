from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quant_platform.data_contracts import PortfolioWeights


@dataclass(frozen=True)
class PortfolioConstraints:
    gross_limit: float = 2.0
    net_limit: float = 0.05
    max_name_weight: float = 0.015
    max_sector_weight: float = 0.03
    min_longs: int = 50
    min_shorts: int = 50


@dataclass(frozen=True)
class PortfolioConstructor:
    constraints: PortfolioConstraints


def enforce_constraints(weights: PortfolioWeights, constraints: PortfolioConstraints, market_data: pd.DataFrame) -> PortfolioWeights:
    clipped = weights.weights.clip(lower=-constraints.max_name_weight, upper=constraints.max_name_weight)
    net = clipped.sum()
    if abs(net) > constraints.net_limit:
        clipped = clipped - (net / len(clipped))
    gross = clipped.abs().sum()
    if gross > constraints.gross_limit:
        clipped = clipped * (constraints.gross_limit / gross)
    return PortfolioWeights(weights=clipped, as_of=weights.as_of)


def construct_portfolio(signal: PortfolioWeights, risk: pd.DataFrame, constraints: PortfolioConstraints) -> PortfolioWeights:
    inv_vol = 1.0 / risk["volatility"].replace(0, pd.NA)
    scaled = signal.weights.mul(inv_vol.reindex(signal.weights.index).fillna(0.0), fill_value=0.0)
    gross = scaled.abs().sum()
    if gross > 0:
        scaled = scaled / gross
    return enforce_constraints(PortfolioWeights(scaled, signal.as_of), constraints, risk)
