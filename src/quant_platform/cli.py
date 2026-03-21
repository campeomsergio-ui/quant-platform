from __future__ import annotations

import argparse
from datetime import datetime, UTC
from typing import Sequence

from quant_platform.experiment_plan import load_experiment_plan, validate_experiment_plan
from quant_platform.experiment_registry import CandidateRecord, append_candidate_record, create_experiment_record, is_final_test_locked, mark_final_test_touched
from quant_platform.io import load_json, save_json
from quant_platform.strategy_spec import load_strategy_spec, validate_strategy_spec


DEFAULT_REGISTRY_PATH = "registry/experiments.json"


def run_spec_check(args: argparse.Namespace) -> int:
    spec = load_strategy_spec(load_json(args.strategy))
    plan = load_experiment_plan(load_json(args.experiment))
    validate_strategy_spec(spec)
    validate_experiment_plan(plan)
    return 0


def run_experiment(args: argparse.Namespace) -> int:
    spec = load_json(args.strategy)
    plan = load_json(args.experiment)
    blueprint = load_json(args.blueprint)
    registry = load_json(args.registry)
    record = create_experiment_record(spec, plan, blueprint, seed=args.seed, metadata={"created_at": datetime.now(UTC).isoformat()})
    experiments = dict(registry.get("experiments", {}))
    experiment_entry = dict(experiments.get(record.experiment_id, {}))
    experiment_entry.update({
        "record": {
            "experiment_id": record.experiment_id,
            "spec_hash": record.spec_hash,
            "plan_hash": record.plan_hash,
            "blueprint_hash": record.blueprint_hash,
            "seed": record.seed,
            "metadata": record.metadata,
        }
    })
    experiments[record.experiment_id] = experiment_entry
    registry["experiments"] = experiments
    for candidate_id in args.candidates:
        registry = append_candidate_record(registry, record.experiment_id, CandidateRecord(candidate_id=candidate_id, params={}, family_name="pre_registered", stage=args.stage))
    if args.touch_final_test:
        if is_final_test_locked(registry, record.experiment_id):
            raise ValueError("locked final test already touched for experiment")
        registry = mark_final_test_touched(registry, record.experiment_id, datetime.now(UTC).isoformat())
    save_json(args.registry, registry)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quant-platform")
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("spec-check")
    check.add_argument("--strategy", required=True)
    check.add_argument("--experiment", required=True)
    check.set_defaults(func=run_spec_check)
    run = sub.add_parser("run-experiment")
    run.add_argument("--strategy", required=True)
    run.add_argument("--experiment", required=True)
    run.add_argument("--blueprint", required=True)
    run.add_argument("--registry", default=DEFAULT_REGISTRY_PATH)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--stage", default="development")
    run.add_argument("--candidate", dest="candidates", action="append", default=[])
    run.add_argument("--touch-final-test", action="store_true")
    run.set_defaults(func=run_experiment)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)
