from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Any


@dataclass(frozen=True)
class ExperimentRecord:
    experiment_id: str
    spec_hash: str
    plan_hash: str
    blueprint_hash: str
    seed: int
    metadata: dict[str, Any]


@dataclass(frozen=True)
class CandidateRecord:
    candidate_id: str
    params: dict[str, Any]
    family_name: str
    stage: str


def stable_payload_hash(payload: dict[str, Any]) -> str:
    import json

    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def build_experiment_id(spec: dict[str, Any], plan: dict[str, Any], blueprint: dict[str, Any], seed: int) -> str:
    import json

    digest = sha256()
    digest.update(json.dumps(spec, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    digest.update(json.dumps(plan, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    digest.update(json.dumps(blueprint, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    digest.update(str(seed).encode("utf-8"))
    return digest.hexdigest()[:16]


def create_experiment_record(spec: dict[str, Any], plan: dict[str, Any], blueprint: dict[str, Any], seed: int, metadata: dict[str, Any]) -> ExperimentRecord:
    return ExperimentRecord(experiment_id=build_experiment_id(spec, plan, blueprint, seed), spec_hash=stable_payload_hash(spec), plan_hash=stable_payload_hash(plan), blueprint_hash=stable_payload_hash(blueprint), seed=seed, metadata=metadata)


def append_candidate_record(registry: dict[str, Any], experiment_id: str, candidate: CandidateRecord) -> dict[str, Any]:
    updated = dict(registry)
    experiments = dict(updated.get("experiments", {}))
    experiment = dict(experiments.get(experiment_id, {}))
    tried = list(experiment.get("candidates", []))
    tried.append(asdict(candidate))
    experiment["candidates"] = tried
    experiments[experiment_id] = experiment
    updated["experiments"] = experiments
    return updated


def mark_final_test_touched(registry: dict[str, Any], experiment_id: str, touched_at: str, stage: str, reason: str) -> dict[str, Any]:
    updated = dict(registry)
    experiments = dict(updated.get("experiments", {}))
    experiment = dict(experiments.get(experiment_id, {}))
    if stage != "final_test":
        raise ValueError("final test touch requires explicit final_test stage")
    if experiment.get("final_test_touched", False):
        raise ValueError("locked final test already touched for experiment")
    experiment["final_test_touched"] = True
    experiment["final_test_touched_at"] = touched_at
    experiment["final_test_stage"] = stage
    experiment["final_test_reason"] = reason
    experiments[experiment_id] = experiment
    updated["experiments"] = experiments
    return updated


def is_final_test_locked(registry: dict[str, Any], experiment_id: str) -> bool:
    return bool(registry.get("experiments", {}).get(experiment_id, {}).get("final_test_touched", False))
