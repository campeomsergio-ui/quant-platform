from __future__ import annotations

import argparse
from datetime import UTC, datetime
from typing import Sequence

from quant_platform.data_access import LocalJsonDataAdapter, inspect_local_dataset
from quant_platform.experiment_plan import load_experiment_plan, validate_experiment_plan
from quant_platform.experiment_registry import CandidateRecord, append_candidate_record, create_experiment_record, is_final_test_locked, mark_final_test_touched
from quant_platform.io import load_json, save_json
from quant_platform.paper.adapter import PaperBrokerAdapter
from quant_platform.paper.runtime import PaperRuntimeConfig, run_daily_paper_cycle
from quant_platform.research import BaselineResearchConfig, ResidualMomentumCycleConfig, run_baseline_research, run_residual_momentum_cycle
from quant_platform.strategy_spec import load_strategy_spec, validate_strategy_spec
from quant_platform.signals.mean_reversion import MeanReversionParams, MeanReversionSignal


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
    experiment_entry.update({"record": {"experiment_id": record.experiment_id, "spec_hash": record.spec_hash, "plan_hash": record.plan_hash, "blueprint_hash": record.blueprint_hash, "seed": record.seed, "metadata": record.metadata}})
    experiments[record.experiment_id] = experiment_entry
    registry["experiments"] = experiments
    for candidate_id in args.candidates:
        registry = append_candidate_record(registry, record.experiment_id, CandidateRecord(candidate_id=candidate_id, params={}, family_name="pre_registered", stage=args.stage))
    if args.touch_final_test:
        if is_final_test_locked(registry, record.experiment_id):
            raise ValueError("locked final test already touched for experiment")
        registry = mark_final_test_touched(registry, record.experiment_id, datetime.now(UTC).isoformat(), "final_test", "manual_cli_touch")
    save_json(args.registry, registry)
    return 0


def run_baseline_local(args: argparse.Namespace) -> int:
    bundle = LocalJsonDataAdapter(args.data_root).load_bundle()
    result = run_baseline_research(bundle, BaselineResearchConfig(residual_model=args.residual_model, signal_lookback=args.signal_lookback, holding_period=args.holding_period, execution_delay_days=args.execution_delay_days))
    print({"metrics": result.metrics, "diagnostics": result.diagnostics})
    return 0


def run_inspect_dataset(args: argparse.Namespace) -> int:
    payload = inspect_local_dataset(args.data_root, preferred_format=args.preferred_format)
    if args.export_path:
        save_json(args.export_path, payload)
    print(payload)
    return 0


def run_paper_daily(args: argparse.Namespace) -> int:
    bundle = LocalJsonDataAdapter(args.data_root).load_bundle()
    signal_model = MeanReversionSignal(MeanReversionParams(residual_model=args.residual_model, residual_lookback=args.signal_lookback, execution_delay_days=1))
    result = run_daily_paper_cycle(bundle.bars, bundle.metadata.set_index("symbol") if "symbol" in bundle.metadata.columns else bundle.metadata, signal_model, PaperBrokerAdapter(), PaperRuntimeConfig(dry_run=args.dry_run, state_path=args.state_path, report_path=args.report_path, seed=args.seed), manual_kill_flag=args.manual_kill)
    print({"status": result.status, "report": None if result.report is None else result.report.summary, "state": {"last_run_date": result.state.last_run_date, "kill_switch_state": result.state.kill_switch_state}})
    return 0


def run_residual_momentum_cycle_cli(args: argparse.Namespace) -> int:
    print({"stage": "bundle_load_start", "data_root": args.data_root})
    bundle = LocalJsonDataAdapter(args.data_root).load_bundle()
    print({"stage": "bundle_load_done", "bars": len(bundle.bars), "benchmark": len(bundle.benchmark), "metadata": len(bundle.metadata)})
    registry = load_json(args.registry)
    result = run_residual_momentum_cycle(bundle, BaselineResearchConfig(), ResidualMomentumCycleConfig(lookbacks=tuple(args.lookbacks), skip_windows=tuple(args.skip_windows), residual_models=tuple(args.residual_models), seed=args.seed, stage=args.stage, touch_final_test=args.touch_final_test, final_test_reason=args.final_test_reason), registry=registry)
    export_payload = {"cycle_label": "new_research_cycle", "experiment_id": result.experiment_id, "candidate_family": result.candidate_family, "multiple_testing": result.multiple_testing, "overfitting": result.overfitting, "comparison": result.comparison, "market_tag": "US_equities_control_env", "final_test_state": result.comparison["aggregate"]["final_test_state"]}
    if args.export_path:
        print({"stage": "export_write_start", "path": args.export_path})
        save_json(args.export_path, export_payload)
        print({"stage": "export_write_done", "path": args.export_path})
    save_json(args.registry, result.registry)
    print({"cycle_label": "new_research_cycle", "experiment_id": result.experiment_id, "best_candidate": result.comparison["best_residual_momentum_candidate"], "final_test_state": result.comparison["aggregate"]["final_test_state"], "export_path": args.export_path})
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
    baseline = sub.add_parser("run-baseline-local")
    baseline.add_argument("--data-root", required=True)
    baseline.add_argument("--residual-model", default="industry_beta_log_mcap")
    baseline.add_argument("--signal-lookback", type=int, default=5)
    baseline.add_argument("--holding-period", type=int, default=5)
    baseline.add_argument("--execution-delay-days", type=int, default=1)
    baseline.set_defaults(func=run_baseline_local)
    inspect = sub.add_parser("inspect-data-local")
    inspect.add_argument("--data-root", required=True)
    inspect.add_argument("--preferred-format", default="auto")
    inspect.add_argument("--export-path")
    inspect.set_defaults(func=run_inspect_dataset)
    paper = sub.add_parser("run-paper-daily")
    paper.add_argument("--data-root", required=True)
    paper.add_argument("--state-path", default="state/paper_runtime_state.json")
    paper.add_argument("--report-path", default="state/paper_runtime_report.json")
    paper.add_argument("--residual-model", default="industry_beta_log_mcap")
    paper.add_argument("--signal-lookback", type=int, default=5)
    paper.add_argument("--seed", type=int, default=0)
    paper.add_argument("--dry-run", action="store_true")
    paper.add_argument("--manual-kill", action="store_true")
    paper.set_defaults(func=run_paper_daily)
    rmom = sub.add_parser("run-residual-momentum-cycle")
    rmom.add_argument("--data-root", required=True)
    rmom.add_argument("--registry", default=DEFAULT_REGISTRY_PATH)
    rmom.add_argument("--export-path")
    rmom.add_argument("--lookback", dest="lookbacks", action="append", type=int, default=[])
    rmom.add_argument("--skip-window", dest="skip_windows", action="append", type=int, default=[])
    rmom.add_argument("--residual-model", dest="residual_models", action="append", default=[])
    rmom.add_argument("--seed", type=int, default=0)
    rmom.add_argument("--stage", default="development")
    rmom.add_argument("--touch-final-test", action="store_true")
    rmom.add_argument("--final-test-reason", default="")
    rmom.set_defaults(func=run_residual_momentum_cycle_cli)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "lookbacks", []) == [] and getattr(args, "command", "") == "run-residual-momentum-cycle":
        args.lookbacks = [20, 60, 120]
    if getattr(args, "skip_windows", []) == [] and getattr(args, "command", "") == "run-residual-momentum-cycle":
        args.skip_windows = [5, 10, 20]
    if getattr(args, "residual_models", []) == [] and getattr(args, "command", "") == "run-residual-momentum-cycle":
        args.residual_models = ["industry_only", "industry_beta", "industry_beta_log_mcap"]
    return args.func(args)
