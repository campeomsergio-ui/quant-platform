from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import pandas as pd

from quant_platform.costs import estimate_costs
from quant_platform.data_access import DataBundle, ensure_valid_point_in_time_bundle
from quant_platform.metrics import compute_primary_metrics


@dataclass(frozen=True)
class EtfTrendCandidate:
    candidate_id: str
    lookback: int
    rule: str
    defensive_asset: str


@dataclass(frozen=True)
class EtfTrendCycleConfig:
    candidates: tuple[EtfTrendCandidate, ...]
    rebalance_frequency: str = "M"
    execution_delay_days: int = 1
    participation_cap: float = 0.10


DEFAULT_ETF_TREND_CANDIDATES = (
    EtfTrendCandidate("tsmom_ret_63_cash", 63, "trailing_return_sign", "CASH"),
    EtfTrendCandidate("tsmom_ret_126_cash", 126, "trailing_return_sign", "CASH"),
    EtfTrendCandidate("tsmom_ret_252_cash", 252, "trailing_return_sign", "CASH"),
    EtfTrendCandidate("tsmom_ret_126_ief", 126, "trailing_return_sign", "IEF"),
)

DEFAULT_ETF_TREND_REFINED_CANDIDATES = (
    EtfTrendCandidate("tsmom_ma_200_cash", 200, "moving_average_filter", "CASH"),
    EtfTrendCandidate("tsmom_ma_200_shy", 200, "moving_average_filter", "SHY"),
    EtfTrendCandidate("tsmom_dual_63_252_cash", 252, "dual_horizon_agreement", "CASH"),
    EtfTrendCandidate("tsmom_dual_63_252_shy", 252, "dual_horizon_agreement", "SHY"),
)


def _trace(stage: str, **fields: Any) -> None:
    print({"stage": stage, **fields}, flush=True)


def _bundle_closes(bundle: DataBundle) -> pd.DataFrame:
    closes = bundle.bars["close"].unstack("symbol").sort_index()
    return closes


def _bundle_opens(bundle: DataBundle) -> pd.DataFrame:
    return bundle.bars["open"].unstack("symbol").sort_index()


def _bundle_adv(bundle: DataBundle) -> pd.DataFrame:
    if "adv" in bundle.bars.columns:
        return bundle.bars["adv"].unstack("symbol").sort_index()
    return pd.DataFrame(0.0, index=_bundle_closes(bundle).index, columns=_bundle_closes(bundle).columns)


def _bundle_vol(bundle: DataBundle) -> pd.DataFrame:
    if "daily_volatility" in bundle.bars.columns:
        return bundle.bars["daily_volatility"].unstack("symbol").sort_index()
    return pd.DataFrame(0.0, index=_bundle_closes(bundle).index, columns=_bundle_closes(bundle).columns)


def _month_end_rebalance_dates(index: pd.DatetimeIndex) -> list[pd.Timestamp]:
    return list(pd.Series(index=index, data=1).resample("ME").last().index.intersection(index))


def _compute_candidate_weights(closes: pd.DataFrame, candidate: EtfTrendCandidate, rebalance_dates: list[pd.Timestamp], tradable_symbols: list[str]) -> dict[pd.Timestamp, pd.Series]:
    trailing = closes[tradable_symbols].pct_change(candidate.lookback)
    ma_filter = closes[tradable_symbols] / closes[tradable_symbols].rolling(candidate.lookback).mean() - 1.0
    short_trailing = closes[tradable_symbols].pct_change(63)
    weights: dict[pd.Timestamp, pd.Series] = {}
    for date in rebalance_dates:
        if date not in trailing.index:
            continue
        if candidate.rule == "trailing_return_sign":
            signal = trailing.loc[date]
            positive = signal[signal > 0].dropna().index.tolist()
        elif candidate.rule == "moving_average_filter":
            signal = ma_filter.loc[date]
            positive = signal[signal > 0].dropna().index.tolist()
        elif candidate.rule == "dual_horizon_agreement":
            signal = trailing.loc[date]
            short_signal = short_trailing.loc[date]
            positive = signal[(signal > 0) & (short_signal > 0)].dropna().index.tolist()
        else:
            raise ValueError(f"unsupported ETF trend rule: {candidate.rule}")
        w = pd.Series(0.0, index=closes.columns)
        if positive:
            alloc = 1.0 / len(positive)
            w.loc[positive] = alloc
        elif candidate.defensive_asset != "CASH" and candidate.defensive_asset in closes.columns:
            w.loc[candidate.defensive_asset] = 1.0
        weights[date] = w
    return weights


def _run_candidate(bundle: DataBundle, candidate: EtfTrendCandidate, config: EtfTrendCycleConfig) -> dict[str, Any]:
    t0 = perf_counter()
    closes = _bundle_closes(bundle)
    opens = _bundle_opens(bundle)
    adv = _bundle_adv(bundle)
    vol = _bundle_vol(bundle)
    tradable = [c for c in closes.columns if c != "SHY"]
    rebalance_dates = _month_end_rebalance_dates(closes.index)
    _trace("etf_candidate_signal_start", candidate_id=candidate.candidate_id, rebalance_count=len(rebalance_dates), lookback=candidate.lookback)
    signals = _compute_candidate_weights(closes, candidate, rebalance_dates, tradable)
    _trace("etf_candidate_signal_done", candidate_id=candidate.candidate_id, elapsed_seconds=round(perf_counter() - t0, 3), num_signal_days=len(signals))

    current = pd.Series(0.0, index=closes.columns)
    net_returns: list[float] = []
    gross_returns: list[float] = []
    turnovers: list[float] = []
    costs: list[float] = []
    dates: list[pd.Timestamp] = []
    position_counts: list[int] = []
    gross_exposures: list[float] = []
    time_in_market: list[float] = []
    long_totals: list[float] = []
    benchmark_aligned: list[float] = []

    signal_dates = sorted(signals)
    for i, signal_date in enumerate(signal_dates):
        exec_idx = closes.index.get_indexer([signal_date])[0] + config.execution_delay_days
        if exec_idx < 0 or exec_idx >= len(closes.index):
            continue
        exec_date = closes.index[exec_idx]
        target = signals[signal_date].copy()
        delta = target.sub(current, fill_value=0.0)
        trade_frame = pd.DataFrame(index=delta.index)
        trade_frame["trade_weight"] = delta
        trade_frame["open"] = opens.loc[exec_date].reindex(delta.index).fillna(0.0)
        trade_frame["adv"] = adv.loc[exec_date].reindex(delta.index).fillna(0.0)
        trade_frame["daily_volatility"] = vol.loc[exec_date].reindex(delta.index).fillna(0.0)
        trade_frame["trade_notional"] = delta.abs() * trade_frame["open"]
        trade_frame["short_notional"] = 0.0
        cost_est = estimate_costs(trade_frame, trade_frame)
        if exec_idx + 1 >= len(closes.index):
            break
        next_date = closes.index[exec_idx + 1]
        oc_returns = (closes.loc[next_date] / opens.loc[exec_date] - 1.0).reindex(target.index).fillna(0.0)
        gross = float(target.mul(oc_returns, fill_value=0.0).sum())
        cost = float(cost_est.total_cost)
        net = gross - cost
        current = target
        dates.append(exec_date)
        gross_returns.append(gross)
        costs.append(cost)
        net_returns.append(net)
        turnovers.append(float(delta.abs().sum()))
        position_counts.append(int((target.abs() > 0).sum()))
        gross_exposures.append(float(target.abs().sum()))
        time_in_market.append(float(target.abs().sum() > 0))
        long_totals.append(float(target[target > 0].sum()))
        if exec_date in bundle.benchmark.index:
            benchmark_aligned.append(float(bundle.benchmark.loc[exec_date]))
        else:
            benchmark_aligned.append(0.0)

    gross_series = pd.Series(gross_returns, index=dates, dtype=float)
    cost_series = pd.Series(costs, index=dates, dtype=float)
    net_series = pd.Series(net_returns, index=dates, dtype=float)
    turnover_series = pd.Series(turnovers, index=dates, dtype=float)
    metrics = compute_primary_metrics(gross_series, cost_series)
    benchmark_series = pd.Series(benchmark_aligned, index=dates, dtype=float)
    diff = net_series.sub(benchmark_series, fill_value=0.0)
    validation = ensure_valid_point_in_time_bundle(bundle)
    return {
        "candidate_id": candidate.candidate_id,
        "params": {
            "lookback": candidate.lookback,
            "rule": candidate.rule,
            "defensive_asset": candidate.defensive_asset,
            "rebalance_frequency": config.rebalance_frequency,
            "execution_delay_days": config.execution_delay_days,
        },
        "metrics": {
            "net_sharpe": metrics.net_sharpe,
            "annualized_return": metrics.annualized_return,
            "max_drawdown": metrics.max_drawdown,
            "t_stat": metrics.t_stat,
        },
        "diagnostics": {
            "num_signal_days": float(len(signals)),
            "num_backtest_days": float(len(net_series)),
            "average_turnover": float(turnover_series.mean()) if len(turnover_series) else 0.0,
            "time_in_market": float(pd.Series(time_in_market).mean()) if time_in_market else 0.0,
            "average_position_count": float(pd.Series(position_counts).mean()) if position_counts else 0.0,
            "average_gross_exposure": float(pd.Series(gross_exposures).mean()) if gross_exposures else 0.0,
            "long_short_attribution": {
                "long_total": float(sum(net_returns)),
                "short_total": 0.0,
                "net_total": float(sum(net_returns)),
                "one_sided_pnl_ratio": float("inf") if sum(net_returns) > 0 else 0.0,
            },
            "delay_cost_sensitivity": {
                "cost_stress": {
                    "baseline_total_cost": float(cost_series.sum()),
                    "cost_1_5x_total": float((cost_series * 1.5).sum()),
                    "cost_2_0x_total": float((cost_series * 2.0).sum()),
                },
                "execution_delay": {"baseline_days": config.execution_delay_days, "optimistic_same_close_mode": False},
            },
            "benchmark_relative": {
                "mean_excess_return": float(diff.mean()) if len(diff) else 0.0,
                "std_excess_return": float(diff.std(ddof=1)) if len(diff) > 1 else 0.0,
                "count": float(len(diff)),
                "cumulative_excess_return": float(diff.sum()),
            },
            "history_quality": validation.summary.get("history_quality"),
            "coverage_quality": validation.summary.get("coverage_quality"),
            "data_quality": {
                "manifest": validation.summary.get("manifest", {}),
                "validation_summary": validation.summary,
                "history_quality": validation.summary.get("history_quality"),
                "coverage_quality": validation.summary.get("coverage_quality"),
            },
        },
    }


def run_etf_trend_cycle(bundle: DataBundle, config: EtfTrendCycleConfig) -> dict[str, Any]:
    ensure_valid_point_in_time_bundle(bundle)
    _trace("etf_trend_cycle_start", candidate_count=len(config.candidates))
    candidates = []
    for candidate in config.candidates:
        candidates.append(_run_candidate(bundle, candidate, config))
    best = max(candidates, key=lambda c: c["metrics"]["net_sharpe"])
    _trace("etf_trend_cycle_done", best_candidate=best["candidate_id"], best_sharpe=best["metrics"]["net_sharpe"])
    return {
        "cycle_label": "etf_trend_following_cycle",
        "dataset_manifest": bundle.dataset_manifest,
        "data_quality_metadata": bundle.data_quality_metadata,
        "candidate_family": [
            {
                "candidate_id": c.candidate_id,
                "lookback": c.lookback,
                "rule": c.rule,
                "defensive_asset": c.defensive_asset,
            }
            for c in config.candidates
        ],
        "results": candidates,
        "best_candidate": best,
    }
