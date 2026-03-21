from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from quant_platform.backtest.engine import BacktestConfig, run_backtest
from quant_platform.data_access import DataBundle, apply_symbol_mapping, attach_delisting_returns, validate_point_in_time_bundle
from quant_platform.experiment_registry import CandidateRecord, append_candidate_record, create_experiment_record, is_final_test_locked, mark_final_test_touched
from quant_platform.metrics import compute_diagnostics, compute_primary_metrics
from quant_platform.signals.base import SignalContext
from quant_platform.signals.mean_reversion import MeanReversionParams, MeanReversionSignal
from quant_platform.signals.residual_momentum import ResidualMomentumParams, ResidualMomentumSignal
from quant_platform.validation.multiple_testing import RealityCheckConfig, build_candidate_differentials, run_white_reality_check
from quant_platform.validation.overfitting import OverfittingConfig, estimate_pbo
from quant_platform.validation.walk_forward import WalkForwardConfig, generate_folds


@dataclass(frozen=True)
class ResearchRunResult:
    metrics: dict[str, float]
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class BaselineResearchConfig:
    residual_model: str = "industry_beta_log_mcap"
    signal_lookback: int = 5
    holding_period: int = 5
    execution_delay_days: int = 1


@dataclass(frozen=True)
class ResidualMomentumCycleConfig:
    lookbacks: tuple[int, ...] = (20, 60, 120)
    skip_windows: tuple[int, ...] = (5, 10, 20)
    residual_models: tuple[str, ...] = ("industry_only", "industry_beta", "industry_beta_log_mcap")
    holding_period: int = 5
    execution_delay_days: int = 1
    registry_experiment_name: str = "residual_momentum_cycle"
    seed: int = 0
    walk_forward: WalkForwardConfig = WalkForwardConfig()
    stage: str = "development"
    touch_final_test: bool = False
    final_test_reason: str = ""


@dataclass(frozen=True)
class ResidualMomentumCycleResult:
    experiment_id: str
    candidate_results: dict[str, ResearchRunResult]
    candidate_family: list[dict[str, Any]]
    multiple_testing: dict[str, Any]
    overfitting: dict[str, float]
    comparison: dict[str, Any]
    fold_results: dict[str, dict[str, ResearchRunResult]]
    registry: dict[str, Any]


def _point_in_time_meta(metadata: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    active = metadata.loc[(metadata["effective_from"] <= as_of) & (metadata["effective_to"].isna() | (metadata["effective_to"] >= as_of))].copy()
    return active.set_index("symbol").sort_index()


def _subset_bundle(bundle: DataBundle, start: pd.Timestamp | None = None, end: pd.Timestamp | None = None) -> DataBundle:
    bars = bundle.bars
    if start is not None:
        bars = bars.loc[bars.index.get_level_values("date") >= start]
    if end is not None:
        bars = bars.loc[bars.index.get_level_values("date") <= end]
    benchmark = bundle.benchmark
    if start is not None:
        benchmark = benchmark.loc[benchmark.index >= start]
    if end is not None:
        benchmark = benchmark.loc[benchmark.index <= end]
    return DataBundle(bars=bars, corporate_actions=bundle.corporate_actions, metadata=bundle.metadata, benchmark=benchmark, delistings=bundle.delistings, symbol_mapping=bundle.symbol_mapping)


def _build_market_data(bundle: DataBundle) -> pd.DataFrame:
    bars = apply_symbol_mapping(bundle.bars, bundle.symbol_mapping)
    market_data = bars[["open", "close", "adv", "daily_volatility"]].copy()
    if not bundle.metadata.empty:
        latest_meta = bundle.metadata.sort_values(["symbol", "effective_from"]).drop_duplicates("symbol", keep="last").set_index("symbol")
        for col in ["sector", "beta", "market_cap"]:
            if col in latest_meta.columns:
                market_data[col] = market_data.index.get_level_values("symbol").map(latest_meta[col])
    return market_data


def build_baseline_signals(bundle: DataBundle, config: BaselineResearchConfig) -> dict[pd.Timestamp, pd.Series]:
    validate_point_in_time_bundle(bundle)
    bars = apply_symbol_mapping(bundle.bars, bundle.symbol_mapping)
    dates = sorted(bars.index.get_level_values("date").unique())
    model = MeanReversionSignal(MeanReversionParams(residual_lookback=config.signal_lookback, execution_delay_days=config.execution_delay_days, residual_model=config.residual_model))
    signals: dict[pd.Timestamp, pd.Series] = {}
    for as_of in dates:
        history = bars.loc[bars.index.get_level_values("date") <= as_of]
        meta = _point_in_time_meta(bundle.metadata, as_of)
        if history.empty or meta.empty:
            continue
        context = SignalContext(bars=history, meta=meta, as_of=as_of, seed=0)
        weights = model.compute(context).weights
        signals[as_of] = weights
    return signals


def build_residual_momentum_signals(bundle: DataBundle, lookback: int, skip_window: int, residual_model: str, execution_delay_days: int) -> dict[pd.Timestamp, pd.Series]:
    validate_point_in_time_bundle(bundle)
    bars = apply_symbol_mapping(bundle.bars, bundle.symbol_mapping)
    dates = sorted(bars.index.get_level_values("date").unique())
    model = ResidualMomentumSignal(ResidualMomentumParams(lookback=lookback, skip_window=skip_window, execution_delay_days=execution_delay_days, residual_model=residual_model))
    signals: dict[pd.Timestamp, pd.Series] = {}
    for as_of in dates:
        history = bars.loc[bars.index.get_level_values("date") <= as_of]
        meta = _point_in_time_meta(bundle.metadata, as_of)
        if history.empty or meta.empty:
            continue
        context = SignalContext(bars=history, meta=meta, as_of=as_of, seed=0)
        weights = model.compute(context).weights
        signals[as_of] = weights
    return signals


def _run_strategy(bundle: DataBundle, signals: dict[pd.Timestamp, pd.Series], holding_period: int, market_tag: str = "US_equities") -> ResearchRunResult:
    market_data = _build_market_data(bundle)
    result = run_backtest(None, {"signals": signals, "market_data": market_data, "config": BacktestConfig(holding_period_days=holding_period, execution_time="open_t_plus_1", min_longs=1, min_shorts=1, max_sector_weight=1.0)})
    metrics = compute_primary_metrics(result.gross_returns, result.costs)
    diagnostics = compute_diagnostics(result)
    return ResearchRunResult(metrics={"net_sharpe": metrics.net_sharpe, "annualized_return": metrics.annualized_return, "max_drawdown": metrics.max_drawdown, "t_stat": metrics.t_stat}, diagnostics={"num_backtest_days": float(len(result.net_returns)), "backtest_status": str(result.diagnostics["status"]), "market_tag": market_tag, **diagnostics.items})


def run_baseline_research(bundle: DataBundle, config: BaselineResearchConfig) -> ResearchRunResult:
    bars = apply_symbol_mapping(bundle.bars, bundle.symbol_mapping)
    closes = bars["close"].unstack("symbol").sort_index()
    opens = bars["open"].unstack("symbol").sort_index()
    returns = closes.div(opens).sub(1.0)
    _ = attach_delisting_returns(returns, bundle.delistings)
    signals = build_baseline_signals(bundle, config)
    result = _run_strategy(bundle, signals, config.holding_period)
    result.diagnostics["num_signal_days"] = float(len(signals))
    result.diagnostics["strategy_label"] = "frozen_baseline"
    return result


def _candidate_id(lookback: int, skip_window: int, residual_model: str) -> str:
    return f"rmom_lb{lookback}_sk{skip_window}_{residual_model}"


def _candidate_series(run: ResearchRunResult) -> pd.Series:
    n = max(2, int(run.diagnostics.get("num_backtest_days", 2)))
    return pd.Series([run.metrics["annualized_return"] / 252.0] * n, index=pd.RangeIndex(n))


def _evaluate_residual_momentum_candidate(bundle: DataBundle, lookback: int, skip_window: int, residual_model: str, holding_period: int, execution_delay_days: int) -> ResearchRunResult:
    signals = build_residual_momentum_signals(bundle, lookback, skip_window, residual_model, execution_delay_days)
    run = _run_strategy(bundle, signals, holding_period)
    run.diagnostics["cycle_label"] = "new_research_cycle"
    run.diagnostics["strategy_label"] = "residual_momentum"
    run.diagnostics["candidate_params"] = {"lookback": lookback, "skip_window": skip_window, "residual_model": residual_model}
    run.diagnostics["num_signal_days"] = float(len(signals))
    return run


def _evaluate_folded_candidates(bundle: DataBundle, cycle_config: ResidualMomentumCycleConfig) -> dict[str, dict[str, ResearchRunResult]]:
    dates = pd.DatetimeIndex(sorted(bundle.bars.index.get_level_values("date").unique()))
    folds = generate_folds(dates, cycle_config.walk_forward)
    if not folds and len(dates) >= 2:
        from quant_platform.data_contracts import WalkForwardFold

        folds = [WalkForwardFold(train_start=dates[0], train_end=dates[max(0, len(dates) // 2 - 1)], validation_start=dates[max(1, len(dates) // 2)], validation_end=dates[-1], fold_id="exploratory_local_sample_fold")]
    fold_results: dict[str, dict[str, ResearchRunResult]] = {}
    for fold in folds:
        validation_bundle = _subset_bundle(bundle, fold.validation_start, fold.validation_end)
        per_fold: dict[str, ResearchRunResult] = {}
        for lookback in cycle_config.lookbacks:
            for skip_window in cycle_config.skip_windows:
                for residual_model in cycle_config.residual_models:
                    cid = _candidate_id(lookback, skip_window, residual_model)
                    run = _evaluate_residual_momentum_candidate(validation_bundle, lookback, skip_window, residual_model, cycle_config.holding_period, cycle_config.execution_delay_days)
                    run.diagnostics["fold_id"] = fold.fold_id
                    per_fold[cid] = run
        fold_results[fold.fold_id] = per_fold
    return fold_results


def run_residual_momentum_cycle(bundle: DataBundle, baseline_config: BaselineResearchConfig, cycle_config: ResidualMomentumCycleConfig, registry: dict[str, Any] | None = None) -> ResidualMomentumCycleResult:
    registry_payload = {} if registry is None else dict(registry)
    spec = {"cycle": cycle_config.registry_experiment_name, "family": "residual_momentum", "lookbacks": list(cycle_config.lookbacks), "skip_windows": list(cycle_config.skip_windows), "residual_models": list(cycle_config.residual_models), "market_tag": "US_equities_control_env"}
    plan = {"discipline": "same_as_baseline", "locked_test": True, "multiple_testing": "white_reality_check", "overfitting": "pbo", "walk_forward": {"train_years": cycle_config.walk_forward.train_years, "validation_years": cycle_config.walk_forward.validation_years, "step_years": cycle_config.walk_forward.step_years, "final_test_start": cycle_config.walk_forward.final_test_start, "final_test_end": cycle_config.walk_forward.final_test_end}}
    blueprint = {"comparison": ["frozen_mean_reversion", "residual_momentum_candidates"], "market_tag": "US_equities_control_env"}
    record = create_experiment_record(spec, plan, blueprint, cycle_config.seed, {"cycle_type": "new_research_cycle", "stage": cycle_config.stage, "baseline_separate": True})
    experiments = dict(registry_payload.get("experiments", {}))
    experiment_entry = dict(experiments.get(record.experiment_id, {}))
    experiment_entry.setdefault("record", {"experiment_id": record.experiment_id, "spec_hash": record.spec_hash, "plan_hash": record.plan_hash, "blueprint_hash": record.blueprint_hash, "seed": record.seed, "metadata": record.metadata})
    experiments[record.experiment_id] = experiment_entry
    registry_payload["experiments"] = experiments

    if cycle_config.touch_final_test:
        if cycle_config.stage != "final_test":
            raise ValueError("touch_final_test requires stage=final_test")
        if not cycle_config.final_test_reason:
            raise ValueError("touch_final_test requires explicit final_test_reason")
        if is_final_test_locked(registry_payload, record.experiment_id):
            raise ValueError("locked final test already touched for experiment")
        registry_payload = mark_final_test_touched(registry_payload, record.experiment_id, pd.Timestamp.now(tz="UTC").isoformat(), cycle_config.stage, cycle_config.final_test_reason)
    elif cycle_config.stage == "final_test":
        raise ValueError("stage=final_test requires touch_final_test=True")

    candidate_results: dict[str, ResearchRunResult] = {}
    candidate_family: list[dict[str, Any]] = []
    candidate_returns: dict[str, pd.Series] = {}
    pbo_frame: dict[str, list[float]] = {}
    baseline_result = run_baseline_research(bundle, baseline_config)
    fold_results = _evaluate_folded_candidates(bundle, cycle_config)
    baseline_series = _candidate_series(baseline_result)

    for lookback in cycle_config.lookbacks:
        for skip_window in cycle_config.skip_windows:
            for residual_model in cycle_config.residual_models:
                cid = _candidate_id(lookback, skip_window, residual_model)
                params = {"lookback": lookback, "skip_window": skip_window, "residual_model": residual_model, "market_tag": "US_equities_control_env"}
                registry_payload = append_candidate_record(registry_payload, record.experiment_id, CandidateRecord(candidate_id=cid, params=params, family_name="residual_momentum_cycle", stage=cycle_config.stage))
                run = _evaluate_residual_momentum_candidate(bundle, lookback, skip_window, residual_model, cycle_config.holding_period, cycle_config.execution_delay_days)
                candidate_results[cid] = run
                candidate_family.append({"candidate_id": cid, **params})
                candidate_returns[cid] = _candidate_series(run)
                per_fold_sharpes = [fold_results[fold_id][cid].metrics["net_sharpe"] for fold_id in fold_results if cid in fold_results[fold_id]]
                cleaned = [float(x) if pd.notna(x) else 0.0 for x in (per_fold_sharpes if per_fold_sharpes else [run.metrics["net_sharpe"], run.metrics["annualized_return"]])]
                if len(cleaned) == 1:
                    cleaned.append(cleaned[0])
                pbo_frame[cid] = cleaned

    differentials = build_candidate_differentials(candidate_returns, baseline_series)
    rc = run_white_reality_check(candidate_returns, baseline_series, RealityCheckConfig(seed=cycle_config.seed, bootstrap_iterations=50)) if candidate_returns else None
    pbo_matrix = pd.DataFrame(pbo_frame).fillna(0.0) if pbo_frame else pd.DataFrame()
    pbo = estimate_pbo(pbo_matrix, OverfittingConfig(seed=cycle_config.seed)) if not pbo_matrix.empty else None
    best_candidate = max(candidate_results.items(), key=lambda item: item[1].metrics["net_sharpe"])[0] if candidate_results else ""
    fold_comparison = {fold_id: {"baseline": baseline_result.metrics, "residual_momentum": {cid: run.metrics for cid, run in fold_map.items()}, "stress": {cid: run.diagnostics.get("delay_cost_sensitivity", {}) for cid, run in fold_map.items()}, "benchmark_differentials": {cid: {"excess_annualized_return": float(run.metrics["annualized_return"] - baseline_result.metrics["annualized_return"]), "excess_sharpe": float(run.metrics["net_sharpe"] - baseline_result.metrics["net_sharpe"])} for cid, run in fold_map.items()}} for fold_id, fold_map in fold_results.items()}
    final_test_state = registry_payload.get("experiments", {}).get(record.experiment_id, {})
    comparison = {
        "cycle_label": "new_research_cycle",
        "baseline": {"label": "frozen_mean_reversion", "metrics": baseline_result.metrics, "diagnostics": baseline_result.diagnostics, "market_tag": "US_equities_control_env"},
        "best_residual_momentum_candidate": {"candidate_id": best_candidate, "metrics": candidate_results[best_candidate].metrics if best_candidate else {}, "diagnostics": candidate_results[best_candidate].diagnostics if best_candidate else {}, "market_tag": "US_equities_control_env"},
        "by_fold": fold_comparison,
        "aggregate": {
            "multiple_testing": {} if rc is None else {"observed_statistic": rc.observed_statistic, "adjusted_p_value": rc.adjusted_p_value, "differential_summary": rc.differential_summary},
            "overfitting": {} if pbo is None else {"probability_of_backtest_overfitting": pbo.probability_of_backtest_overfitting, "probability_of_loss": pbo.probability_of_loss, "rank_stability": pbo.rank_stability},
            "benchmark_differentials": {cid: {"mean_excess_return": float(series.mean()) if len(series) else 0.0, "std_excess_return": float(series.std(ddof=1)) if len(series) > 1 else 0.0, "count": float(len(series))} for cid, series in differentials.items()},
            "final_test_state": {"final_test_touched": bool(final_test_state.get("final_test_touched", False)), "final_test_stage": final_test_state.get("final_test_stage"), "final_test_reason": final_test_state.get("final_test_reason")},
        },
        "by_stress_case": {cid: run.diagnostics.get("delay_cost_sensitivity", {}) for cid, run in candidate_results.items()},
    }
    return ResidualMomentumCycleResult(experiment_id=record.experiment_id, candidate_results=candidate_results, candidate_family=candidate_family, multiple_testing={} if rc is None else {"observed_statistic": rc.observed_statistic, "adjusted_p_value": rc.adjusted_p_value, "differential_summary": rc.differential_summary}, overfitting={} if pbo is None else {"probability_of_backtest_overfitting": pbo.probability_of_backtest_overfitting, "probability_of_loss": pbo.probability_of_loss, "rank_stability": pbo.rank_stability}, comparison=comparison, fold_results=fold_results, registry=registry_payload)
