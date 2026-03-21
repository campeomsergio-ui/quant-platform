from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class UniverseDefinition(BaseModel):
    model_config = ConfigDict(frozen=True)
    rules: list[str]


class DataRequirements(BaseModel):
    model_config = ConfigDict(frozen=True)
    items: list[str]


class SignalDefinition(BaseModel):
    model_config = ConfigDict(frozen=True)
    rules: list[str]


class RiskModel(BaseModel):
    model_config = ConfigDict(frozen=True)
    rules: list[str]


class ConstraintSet(BaseModel):
    model_config = ConfigDict(frozen=True)
    rules: list[str]


class ValidationPlan(BaseModel):
    model_config = ConfigDict(frozen=True)
    rules: list[str]


class StrategySpec(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str
    universe_definition: list[str]
    data_requirements: list[str]
    signal_definition: list[str]
    holding_period: str
    rebalancing_rule: list[str]
    transaction_cost_model: list[str]
    risk_model: list[str]
    constraints: list[str]
    hyperparameters: list[str]
    validation_plan: list[str]
    failure_modes: list[str]
    minimum_acceptance_criteria: list[str]
    conservative_defaults_extension: list[str] = Field(default_factory=list)


def load_strategy_spec(data: dict[str, Any]) -> StrategySpec:
    return StrategySpec.model_validate(data)


def validate_strategy_spec(spec: StrategySpec) -> None:
    required = [
        spec.universe_definition,
        spec.data_requirements,
        spec.signal_definition,
        spec.rebalancing_rule,
        spec.transaction_cost_model,
        spec.risk_model,
        spec.constraints,
        spec.hyperparameters,
        spec.validation_plan,
        spec.failure_modes,
        spec.minimum_acceptance_criteria,
    ]
    if any(len(section) == 0 for section in required):
        raise ValueError("strategy spec sections must be non-empty")
