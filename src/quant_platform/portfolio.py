from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from quant_platform.data_contracts import PortfolioWeights
from quant_platform.risk import neutralize_beta_exposure


@dataclass(frozen=True)
class PortfolioConstraints:
    gross_limit: float = 2.0
    net_limit: float = 0.05
    max_name_weight: float = 0.015
    max_sector_weight: float = 0.03
    min_longs: int = 50
    min_shorts: int = 50
    beta_target: float = 0.0
    beta_tolerance: float = 0.05
    participation_cap: float = 0.10


@dataclass(frozen=True)
class PortfolioConstructor:
    constraints: PortfolioConstraints


@dataclass(frozen=True)
class ConstraintEvent:
    constraint: str
    action: str
    before: float | str
    after: float | str
    reason: str


@dataclass(frozen=True)
class ConstraintResult:
    weights: PortfolioWeights
    events: list[ConstraintEvent]
    failed: bool
    failure_reasons: list[str]
    summary: dict[str, float | int | str]


def _sector_abs_weights(weights: pd.Series, market_data: pd.DataFrame) -> pd.Series:
    sectors = market_data.get("sector", pd.Series(index=weights.index, dtype=object)).reindex(weights.index)
    return weights.abs().groupby(sectors.fillna("UNKNOWN")).sum()


def _apply_name_cap(weights: pd.Series, constraints: PortfolioConstraints, events: list[ConstraintEvent]) -> pd.Series:
    capped = weights.clip(lower=-constraints.max_name_weight, upper=constraints.max_name_weight)
    if not capped.equals(weights):
        events.append(ConstraintEvent("max_name_weight", "clip", float(weights.abs().max()), float(capped.abs().max()), "applied single-name cap"))
    return capped


def _apply_liquidity_caps(weights: pd.Series, constraints: PortfolioConstraints, market_data: pd.DataFrame, events: list[ConstraintEvent]) -> tuple[pd.Series, list[str]]:
    if "adv" not in market_data.columns or "open" not in market_data.columns:
        return weights, []
    prices = market_data["open"].reindex(weights.index).replace(0.0, pd.NA)
    adv = market_data["adv"].reindex(weights.index)
    liquidity_cap = (constraints.participation_cap * adv / prices).replace([pd.NA, float("inf")], 0.0).fillna(0.0)
    capped = weights.copy()
    failures: list[str] = []
    for symbol in capped.index:
        limit = float(liquidity_cap.get(symbol, 0.0))
        if limit <= 0.0 and abs(float(capped[symbol])) > 0.0:
            capped[symbol] = 0.0
            failures.append(f"missing liquidity for {symbol}")
            continue
        clipped = max(min(float(capped[symbol]), limit), -limit)
        if clipped != float(capped[symbol]):
            events.append(ConstraintEvent("liquidity_cap", "clip", float(capped[symbol]), clipped, f"ADV participation cap for {symbol}"))
            capped[symbol] = clipped
    return capped, failures


def _apply_sector_caps(weights: pd.Series, constraints: PortfolioConstraints, market_data: pd.DataFrame, events: list[ConstraintEvent]) -> pd.Series:
    capped = weights.copy()
    sectors = market_data.get("sector", pd.Series(index=weights.index, dtype=object)).reindex(weights.index).fillna("UNKNOWN")
    for sector, names in sectors.groupby(sectors):
        members = names.index
        sector_abs = capped.reindex(members).abs().sum()
        if sector_abs > constraints.max_sector_weight and sector_abs > 0:
            scale = constraints.max_sector_weight / sector_abs
            before = float(sector_abs)
            capped.loc[members] = capped.loc[members] * scale
            after = float(capped.reindex(members).abs().sum())
            events.append(ConstraintEvent("sector_cap", "scale", before, after, f"sector cap enforced for {sector}"))
    return capped


def _apply_gross_net(weights: pd.Series, constraints: PortfolioConstraints, events: list[ConstraintEvent]) -> pd.Series:
    adjusted = weights.copy()
    gross = adjusted.abs().sum()
    if gross > constraints.gross_limit and gross > 0:
        before = float(gross)
        adjusted = adjusted * (constraints.gross_limit / gross)
        events.append(ConstraintEvent("gross_limit", "scale", before, float(adjusted.abs().sum()), "gross exposure scaled to limit"))
    net = adjusted.sum()
    if abs(net) > constraints.net_limit and len(adjusted) > 0:
        before = float(net)
        adjusted = adjusted - (net / len(adjusted))
        events.append(ConstraintEvent("net_limit", "shift", before, float(adjusted.sum()), "net exposure shifted to limit"))
    return adjusted


def enforce_constraints(weights: PortfolioWeights, constraints: PortfolioConstraints, market_data: pd.DataFrame) -> ConstraintResult:
    adjusted = weights.weights.copy().fillna(0.0)
    events: list[ConstraintEvent] = []
    failures: list[str] = []

    adjusted = _apply_name_cap(adjusted, constraints, events)
    adjusted, liquidity_failures = _apply_liquidity_caps(adjusted, constraints, market_data, events)
    failures.extend(liquidity_failures)
    adjusted = _apply_sector_caps(adjusted, constraints, market_data, events)

    if "beta" in market_data.columns and adjusted.abs().sum() > 0:
        before_beta = float(adjusted.mul(market_data["beta"].reindex(adjusted.index).fillna(0.0)).sum())
        adjusted = neutralize_beta_exposure(adjusted, market_data["beta"].reindex(adjusted.index).fillna(0.0), constraints.beta_target)
        after_beta = float(adjusted.mul(market_data["beta"].reindex(adjusted.index).fillna(0.0)).sum())
        if abs(before_beta - after_beta) > 1e-12:
            events.append(ConstraintEvent("beta_neutralization", "shift", before_beta, after_beta, "beta neutralization at total-book level"))

    adjusted = _apply_gross_net(adjusted, constraints, events)

    long_count = int((adjusted > 0).sum())
    short_count = int((adjusted < 0).sum())
    if long_count < constraints.min_longs:
        failures.append(f"insufficient longs: {long_count} < {constraints.min_longs}")
    if short_count < constraints.min_shorts:
        failures.append(f"insufficient shorts: {short_count} < {constraints.min_shorts}")

    sector_abs = _sector_abs_weights(adjusted, market_data)
    sector_breach = sector_abs[sector_abs > constraints.max_sector_weight + 1e-12]
    if not sector_breach.empty:
        failures.append("sector cap unsatisfied after enforcement")

    gross = float(adjusted.abs().sum())
    net = float(adjusted.sum())
    if gross > constraints.gross_limit + 1e-12:
        failures.append("gross limit unsatisfied after enforcement")
    if abs(net) > constraints.net_limit + 1e-12:
        failures.append("net limit unsatisfied after enforcement")

    beta_exposure = 0.0
    if "beta" in market_data.columns:
        beta_exposure = float(adjusted.mul(market_data["beta"].reindex(adjusted.index).fillna(0.0)).sum())
        if abs(beta_exposure - constraints.beta_target) > constraints.beta_tolerance:
            failures.append("beta target unsatisfied after enforcement")

    return ConstraintResult(
        weights=PortfolioWeights(weights=adjusted, as_of=weights.as_of),
        events=events,
        failed=bool(failures),
        failure_reasons=failures,
        summary={
            "gross": gross,
            "net": net,
            "long_count": long_count,
            "short_count": short_count,
            "beta_exposure": beta_exposure,
            "max_sector_abs": float(sector_abs.max()) if not sector_abs.empty else 0.0,
        },
    )


def construct_portfolio(signal: PortfolioWeights, risk: pd.DataFrame, constraints: PortfolioConstraints) -> PortfolioWeights:
    inv_vol = 1.0 / risk["volatility"].replace(0, pd.NA)
    scaled = signal.weights.mul(inv_vol.reindex(signal.weights.index).fillna(0.0), fill_value=0.0)
    gross = scaled.abs().sum()
    if gross > 0:
        scaled = scaled / gross
    return enforce_constraints(PortfolioWeights(scaled, signal.as_of), constraints, risk).weights
