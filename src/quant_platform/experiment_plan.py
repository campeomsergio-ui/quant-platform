from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SplitConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    rules: list[str]


class SweepConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    rules: list[str]


class MultipleTestingControl(BaseModel):
    model_config = ConfigDict(frozen=True)
    rules: list[str]


class OverfittingEstimation(BaseModel):
    model_config = ConfigDict(frozen=True)
    rules: list[str]


class MetricSet(BaseModel):
    model_config = ConfigDict(frozen=True)
    primary_and_secondary: list[str]


class StopRuleSet(BaseModel):
    model_config = ConfigDict(frozen=True)
    rules: list[str]


class ExperimentPlan(BaseModel):
    model_config = ConfigDict(frozen=True)
    splits: list[str]
    sweeps: list[str]
    multiple_testing_control: list[str]
    overfitting_estimation: list[str]
    metrics: list[str]
    diagnostics: list[str]
    stop_rules: list[str]
    conservative_defaults_extension: list[str] = Field(default_factory=list)


def load_experiment_plan(data: dict[str, Any]) -> ExperimentPlan:
    return ExperimentPlan.model_validate(data)


def validate_experiment_plan(plan: ExperimentPlan) -> None:
    if not plan.splits or not plan.sweeps or not plan.stop_rules:
        raise ValueError("experiment plan missing required sections")
