from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quant_platform.data_access import LocalJsonDataAdapter, LocalTableDataAdapter, inspect_local_dataset
from quant_platform.etf_trend import DEFAULT_ETF_TREND_CANDIDATES, EtfTrendCycleConfig, run_etf_trend_cycle
from quant_platform.io import load_json
from quant_platform.research import BaselineResearchConfig, ResidualMomentumCycleConfig, run_baseline_research, run_residual_momentum_cycle


DEFAULT_STOCK_RESIDUAL_CANDIDATE = {
    "lookback": 20,
    "skip_window": 5,
    "residual_model": "industry_beta_log_mcap",
}
DEFAULT_ETF_BASELINE_CANDIDATE_ID = "tsmom_ret_252_cash"
SUPPORTED_SLEEVES = (
    "stock_baseline_anchor",
    "stock_residual_momentum_single_candidate",
    "etf_trend_baseline",
)


@dataclass(frozen=True)
class OrchestratorConfig:
    data_root: str
    preferred_format: str = "auto"
    sleeves: tuple[str, ...] = SUPPORTED_SLEEVES
    stock_residual_lookback: int = 20
    stock_residual_skip_window: int = 5
    stock_residual_model: str = "industry_beta_log_mcap"
    etf_candidate_id: str = DEFAULT_ETF_BASELINE_CANDIDATE_ID


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_bundle(data_root: str, preferred_format: str):
    adapter = LocalJsonDataAdapter(data_root) if preferred_format == "json" else LocalTableDataAdapter(data_root, preferred_format=preferred_format)
    return adapter.load_bundle()


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    return load_json(str(path)) if path.exists() else {}


def _collect_verified_repo_evidence(repo_root: str) -> dict[str, Any]:
    outputs = Path(repo_root) / "outputs"
    stock_baseline = []
    stock_residual = []
    etf_baseline = []

    for path in sorted(outputs.glob("*/baseline_result.json")):
        payload = _read_json_if_exists(path)
        metrics = payload.get("metrics", {})
        stock_baseline.append({
            "path": str(path.relative_to(repo_root)),
            "net_sharpe": _safe_float(metrics.get("net_sharpe")),
            "annualized_return": _safe_float(metrics.get("annualized_return")),
            "max_drawdown": _safe_float(metrics.get("max_drawdown")),
        })
    for path in sorted(outputs.glob("*/residual_momentum_cycle.json")):
        payload = _read_json_if_exists(path)
        best = payload.get("comparison", {}).get("best_residual_momentum_candidate", {})
        metrics = best.get("metrics", {})
        stock_residual.append({
            "path": str(path.relative_to(repo_root)),
            "candidate_id": best.get("candidate_id"),
            "net_sharpe": _safe_float(metrics.get("net_sharpe")),
            "annualized_return": _safe_float(metrics.get("annualized_return")),
            "max_drawdown": _safe_float(metrics.get("max_drawdown")),
        })
    for path in sorted(outputs.glob("etf_*/*cycle.json")):
        payload = _read_json_if_exists(path)
        best = payload.get("best_candidate", {})
        metrics = best.get("metrics", {})
        etf_baseline.append({
            "path": str(path.relative_to(repo_root)),
            "candidate_id": best.get("candidate_id"),
            "net_sharpe": _safe_float(metrics.get("net_sharpe")),
            "annualized_return": _safe_float(metrics.get("annualized_return")),
            "max_drawdown": _safe_float(metrics.get("max_drawdown")),
        })

    return {
        "stock_baseline_anchor": stock_baseline,
        "stock_residual_momentum_single_candidate": stock_residual,
        "etf_trend_baseline": etf_baseline,
    }


def _mean_metric(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return sum(values) / len(values) if values else None


def build_verified_sleeve_registry(repo_root: str) -> dict[str, Any]:
    evidence = _collect_verified_repo_evidence(repo_root)
    baseline_mean_sharpe = _mean_metric(evidence["stock_baseline_anchor"], "net_sharpe")
    baseline_mean_return = _mean_metric(evidence["stock_baseline_anchor"], "annualized_return")
    residual_mean_sharpe = _mean_metric(evidence["stock_residual_momentum_single_candidate"], "net_sharpe")
    residual_mean_return = _mean_metric(evidence["stock_residual_momentum_single_candidate"], "annualized_return")
    etf_mean_sharpe = _mean_metric(evidence["etf_trend_baseline"], "net_sharpe")
    etf_mean_return = _mean_metric(evidence["etf_trend_baseline"], "annualized_return")

    return {
        "stock_baseline_anchor": {
            "status": "killed",
            "reason": "Verified stock-baseline runs are consistently deeply negative; retain only as a comparison anchor, not as an active research sleeve.",
            "evidence": {
                "verified_run_count": len(evidence["stock_baseline_anchor"]),
                "mean_net_sharpe": baseline_mean_sharpe,
                "mean_annualized_return": baseline_mean_return,
                "sample_runs": evidence["stock_baseline_anchor"][:3],
            },
        },
        "stock_residual_momentum_single_candidate": {
            "status": "strongest_current_candidate",
            "reason": "The verified residual-momentum single candidate is still poor in absolute terms, but it is consistently less bad than the stock baseline across verified repo runs.",
            "evidence": {
                "verified_run_count": len(evidence["stock_residual_momentum_single_candidate"]),
                "mean_net_sharpe": residual_mean_sharpe,
                "mean_annualized_return": residual_mean_return,
                "dominates_stock_baseline_on_mean_sharpe": bool(residual_mean_sharpe is not None and baseline_mean_sharpe is not None and residual_mean_sharpe > baseline_mean_sharpe),
                "dominates_stock_baseline_on_mean_return": bool(residual_mean_return is not None and baseline_mean_return is not None and residual_mean_return > baseline_mean_return),
                "sample_runs": evidence["stock_residual_momentum_single_candidate"][:3],
            },
        },
        "etf_trend_baseline": {
            "status": "mechanically_valid_but_poor",
            "reason": "ETF trend baseline research artifacts exist and run end-to-end, but currently verified baseline-family outcomes are still materially negative and not promotion-ready.",
            "evidence": {
                "verified_run_count": len(evidence["etf_trend_baseline"]),
                "mean_net_sharpe": etf_mean_sharpe,
                "mean_annualized_return": etf_mean_return,
                "sample_runs": evidence["etf_trend_baseline"][:3],
            },
        },
    }


def _control_rules() -> dict[str, Any]:
    return {
        "minimum_data_quality": {
            "require_validation_ok": True,
            "max_error_count": 0,
            "warning_tolerance": "warnings allowed but carried into operator risk summary",
        },
        "degenerate_output_checks": {
            "require_positive_backtest_days": True,
            "require_finite_metrics": ["net_sharpe", "annualized_return", "max_drawdown"],
        },
        "anchor_dominance_rule": {
            "description": "A non-anchor stock sleeve should be killed for the current dataset if it is clearly worse than the stock baseline anchor on both Sharpe and annualized return.",
            "comparison_fields": ["net_sharpe", "annualized_return"],
        },
        "kill_conditions": [
            "dataset validation contains any error",
            "backtest output is degenerate or empty",
            "non-anchor stock sleeve is clearly worse than the stock baseline anchor on both Sharpe and annualized return",
        ],
    }


def _is_finite_number(value: Any) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return numeric == numeric and numeric not in (float("inf"), float("-inf"))


def _run_stock_baseline(bundle) -> dict[str, Any]:
    result = run_baseline_research(bundle, BaselineResearchConfig())
    return {
        "sleeve_id": "stock_baseline_anchor",
        "metrics": result.metrics,
        "diagnostics": result.diagnostics,
        "raw_result": result,
    }


def _run_stock_residual(bundle, baseline_result, *, lookback: int, skip_window: int, residual_model: str) -> dict[str, Any]:
    cycle = run_residual_momentum_cycle(
        bundle,
        BaselineResearchConfig(),
        ResidualMomentumCycleConfig(
            lookbacks=(lookback,),
            skip_windows=(skip_window,),
            residual_models=(residual_model,),
            stage="research_ops_orchestrator",
        ),
        registry={},
        baseline_result=baseline_result,
    )
    best = cycle.comparison["best_residual_momentum_candidate"]
    return {
        "sleeve_id": "stock_residual_momentum_single_candidate",
        "candidate_id": best.get("candidate_id"),
        "metrics": best.get("metrics", {}),
        "diagnostics": best.get("diagnostics", {}),
        "raw_result": cycle,
    }


def _run_etf_baseline(bundle, candidate_id: str) -> dict[str, Any]:
    family = tuple(candidate for candidate in DEFAULT_ETF_TREND_CANDIDATES if candidate.candidate_id == candidate_id)
    if not family:
        raise ValueError(f"unsupported ETF baseline candidate_id: {candidate_id}")
    payload = run_etf_trend_cycle(bundle, EtfTrendCycleConfig(candidates=family))
    best = payload["best_candidate"]
    return {
        "sleeve_id": "etf_trend_baseline",
        "candidate_id": best.get("candidate_id"),
        "metrics": best.get("metrics", {}),
        "diagnostics": best.get("diagnostics", {}),
        "raw_result": payload,
    }


def _sleeve_decision(sleeve_id: str, metrics: dict[str, Any], diagnostics: dict[str, Any], inspect_payload: dict[str, Any], baseline_metrics: dict[str, Any] | None) -> dict[str, Any]:
    validation = inspect_payload.get("validation", {})
    issues = validation.get("issues", [])
    error_count = sum(1 for issue in issues if issue.get("level") == "error")
    warning_count = sum(1 for issue in issues if issue.get("level") == "warning")
    num_backtest_days = diagnostics.get("num_backtest_days", 0)

    kill_reasons: list[str] = []
    if not validation.get("ok", False) or error_count > 0:
        kill_reasons.append("dataset failed validation")
    if not num_backtest_days:
        kill_reasons.append("degenerate output: no backtest days")
    if not all(_is_finite_number(metrics.get(field)) for field in ("net_sharpe", "annualized_return", "max_drawdown")):
        kill_reasons.append("degenerate output: non-finite metrics")
    if sleeve_id != "stock_baseline_anchor" and sleeve_id.startswith("stock_") and baseline_metrics:
        if float(metrics.get("net_sharpe", 0.0)) < float(baseline_metrics.get("net_sharpe", 0.0)) and float(metrics.get("annualized_return", 0.0)) < float(baseline_metrics.get("annualized_return", 0.0)):
            kill_reasons.append("dominated by stock baseline anchor on both Sharpe and annualized return")

    if kill_reasons:
        action = "kill"
    elif float(metrics.get("net_sharpe", 0.0)) < 0 or float(metrics.get("annualized_return", 0.0)) < 0:
        action = "watch"
    else:
        action = "keep"

    return {
        "action": action,
        "reason": "; ".join(kill_reasons) if kill_reasons else ("metrics remain negative, so keep under watch only" if action == "watch" else "passes current mechanical control checks"),
        "evidence": {
            "metrics": metrics,
            "num_backtest_days": num_backtest_days,
            "dataset_validation_ok": validation.get("ok", False),
            "dataset_warning_count": warning_count,
            "dataset_error_count": error_count,
        },
    }


def run_research_orchestrator(config: OrchestratorConfig, *, repo_root: str) -> dict[str, Any]:
    bundle = _load_bundle(config.data_root, config.preferred_format)
    inspect_payload = inspect_local_dataset(config.data_root, preferred_format=config.preferred_format)
    verified_registry = build_verified_sleeve_registry(repo_root)
    run_results: dict[str, Any] = {}
    decisions: dict[str, Any] = {}
    baseline_metrics: dict[str, Any] | None = None
    baseline_raw = None

    if "stock_baseline_anchor" in config.sleeves:
        baseline_payload = _run_stock_baseline(bundle)
        baseline_raw = baseline_payload.pop("raw_result")
        baseline_metrics = baseline_payload["metrics"]
        run_results["stock_baseline_anchor"] = baseline_payload
    if "stock_residual_momentum_single_candidate" in config.sleeves:
        if baseline_raw is None:
            baseline_payload = _run_stock_baseline(bundle)
            baseline_raw = baseline_payload.pop("raw_result")
            baseline_metrics = baseline_payload["metrics"]
            run_results.setdefault("stock_baseline_anchor", baseline_payload)
        residual_payload = _run_stock_residual(
            bundle,
            baseline_raw,
            lookback=config.stock_residual_lookback,
            skip_window=config.stock_residual_skip_window,
            residual_model=config.stock_residual_model,
        )
        residual_payload.pop("raw_result")
        run_results["stock_residual_momentum_single_candidate"] = residual_payload
    if "etf_trend_baseline" in config.sleeves:
        etf_payload = _run_etf_baseline(bundle, config.etf_candidate_id)
        etf_payload.pop("raw_result")
        run_results["etf_trend_baseline"] = etf_payload

    for sleeve_id, payload in run_results.items():
        decisions[sleeve_id] = _sleeve_decision(sleeve_id, payload.get("metrics", {}), payload.get("diagnostics", {}), inspect_payload, baseline_metrics)

    top_risks = []
    if inspect_payload.get("validation", {}).get("issues"):
        top_risks.extend(issue.get("message", "unspecified data issue") for issue in inspect_payload["validation"]["issues"][:5])
    top_risks.extend([
        "Current sleeve registry state is based on verified repo artifacts, not live capital-readiness.",
        "Negative backtest metrics should not be overinterpreted as stable estimates of future behavior.",
        "External dataset quality remains the main trust bottleneck for stock-sleeve evaluation.",
    ])

    strongest_current = None
    strongest_sharpe = None
    for sleeve_id, payload in run_results.items():
        sharpe = _safe_float(payload.get("metrics", {}).get("net_sharpe"))
        if sharpe is None:
            continue
        if strongest_sharpe is None or sharpe > strongest_sharpe:
            strongest_sharpe = sharpe
            strongest_current = sleeve_id

    return {
        "artifact_type": "agent_research_ops_control",
        "generated_at": _utc_now(),
        "dataset": {
            "data_root": config.data_root,
            "preferred_format": config.preferred_format,
            "manifest": inspect_payload.get("manifest", {}),
            "validation": inspect_payload.get("validation", {}),
            "human_summary": inspect_payload.get("human_summary", {}),
        },
        "selected_sleeves": list(config.sleeves),
        "control_rules": _control_rules(),
        "sleeve_registry": verified_registry,
        "run_results": run_results,
        "operator_decision_report": {
            "dataset_used": {
                "data_root": config.data_root,
                "date_range": inspect_payload.get("human_summary", {}).get("date_range", {}),
                "history_quality": inspect_payload.get("human_summary", {}).get("history_quality"),
                "coverage_quality": inspect_payload.get("human_summary", {}).get("coverage_quality"),
            },
            "sleeve_results": run_results,
            "decisions": decisions,
            "strongest_current_run_by_sharpe": strongest_current,
            "top_risks": top_risks,
            "not_trustworthy_yet": [
                "live trading behavior",
                "claims of investability for any current sleeve",
                "ETF and stock sleeve conclusions on narrow or convenience datasets without stronger external validation",
            ],
        },
    }
