from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class OpenClawTask:
    task_id: str
    title: str
    why: str
    implementation_notes: list[str]
    acceptance_checks: list[str]
    priority: str


_DEFAULT_TASKS: tuple[OpenClawTask, ...] = (
    OpenClawTask(
        task_id="qp-001",
        title="Wire persistent locked-test registry state",
        why="Final locked test touch enforcement is currently scaffolded and not durably persisted across all workflows.",
        implementation_notes=[
            "Add atomic registry write path and rollback-safe updates for experiment state transitions.",
            "Record immutable final-test touch metadata (timestamp, actor, reason, and source command).",
            "Enforce idempotency so repeated writes cannot silently mutate historical events.",
        ],
        acceptance_checks=[
            "touch-final-test can only succeed once per experiment_id",
            "registry history remains append-only",
            "CLI emits explicit error messages on duplicate touch",
        ],
        priority="high",
    ),
    OpenClawTask(
        task_id="qp-002",
        title="Upgrade PBO from rank-only placeholder to CSCV workflow",
        why="Current overfitting estimate is a conservative placeholder and should be upgraded for realistic strategy selection governance.",
        implementation_notes=[
            "Implement combinatorially symmetric cross-validation fold construction.",
            "Compute in-sample vs out-of-sample rank deterioration across partitions.",
            "Expose decomposition outputs in diagnostics for auditability.",
        ],
        acceptance_checks=[
            "PBO remains bounded in [0, 1]",
            "Known synthetic edge cases produce expected monotonic PBO behavior",
            "Diagnostics include partition count and rank stability details",
        ],
        priority="high",
    ),
    OpenClawTask(
        task_id="qp-003",
        title="Add realistic residualization model path",
        why="Residual momentum currently uses a conservative sector-demeaned proxy; adding explicit regression residualization improves economic fidelity.",
        implementation_notes=[
            "Implement optional cross-sectional regression residualization using sector, beta, and log market cap terms.",
            "Keep baseline behavior unchanged unless the advanced model is explicitly selected.",
            "Persist model choice into run diagnostics and registry candidate params.",
        ],
        acceptance_checks=[
            "Existing baseline tests continue to pass",
            "Advanced residualization produces finite outputs under sparse metadata",
            "CLI candidate output includes selected residual model",
        ],
        priority="medium",
    ),
)


def build_openclaw_plan(repo_root: str | Path = ".") -> dict[str, Any]:
    root = Path(repo_root).resolve()
    tasks = [
        {
            "task_id": task.task_id,
            "title": task.title,
            "priority": task.priority,
            "why": task.why,
            "implementation_notes": task.implementation_notes,
            "acceptance_checks": task.acceptance_checks,
        }
        for task in _DEFAULT_TASKS
    ]
    return {
        "project": "quant-platform",
        "handoff_target": "OpenClaw",
        "repo_root": str(root),
        "generated_tasks": tasks,
    }
