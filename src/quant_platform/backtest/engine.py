from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from quant_platform.costs import estimate_costs, stress_costs
from quant_platform.data_contracts import BacktestResult, CandidateConfig, PortfolioWeights
from quant_platform.portfolio import PortfolioConstraints, enforce_constraints


@dataclass(frozen=True)
class BacktestConfig:
    holding_period_days: int = 5
    signal_time: str = "close_t"
    order_time: str = "after_close_t"
    execution_time: str = "open_t_plus_1"
    optimistic_same_close_mode: bool = False
    enforce_constraints_on_total_book: bool = True
    net_across_tranches_before_costs: bool = True
    participation_cap: float = 0.10
    max_name_weight: float = 0.015
    gross_limit: float = 2.0
    net_limit: float = 0.05
    max_sector_weight: float = 0.03
    min_longs: int = 50
    min_shorts: int = 50
    beta_target: float = 0.0
    beta_tolerance: float = 0.05
    seed: int = 0


@dataclass(frozen=True)
class BacktestEngine:
    config: BacktestConfig


def simulate_tranches(signals: dict[pd.Timestamp, pd.Series], holding_period: int) -> pd.DataFrame:
    ordered_dates = sorted(signals)
    columns = [entry_date.strftime("%Y-%m-%d") for entry_date in ordered_dates[:holding_period]]
    frame = pd.DataFrame(index=ordered_dates, columns=columns, dtype=object)
    for active_date in ordered_dates:
        active_idx = ordered_dates.index(active_date)
        for entry_date in ordered_dates[max(0, active_idx - holding_period + 1) : active_idx + 1]:
            entry_idx = ordered_dates.index(entry_date)
            col = ordered_dates[entry_idx].strftime("%Y-%m-%d")
            if col in frame.columns:
                frame.at[active_date, col] = signals[entry_date]
    return frame


def _aggregate_book(tranche_state: pd.DataFrame) -> pd.DataFrame:
    aggregated: dict[pd.Timestamp, pd.Series] = {}
    for date, row in tranche_state.iterrows():
        total = pd.Series(dtype=float)
        for value in row:
            if isinstance(value, pd.Series):
                total = total.add(value, fill_value=0.0)
        aggregated[date] = total
    return pd.DataFrame.from_dict(aggregated, orient="index").sort_index().fillna(0.0)


def _blocked_names_for_date(blocked: pd.DataFrame, date: pd.Timestamp) -> pd.Index:
    if blocked.empty or date not in blocked.index:
        return pd.Index([])
    return blocked.columns[blocked.loc[date].fillna(False).astype(bool)]


def _build_trade_frame(delta: pd.Series, market_data: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
    md = market_data.xs(date)
    frame = pd.DataFrame(index=delta.index)
    frame["trade_weight"] = delta
    frame["open"] = md.reindex(delta.index)["open"].fillna(0.0)
    frame["adv"] = md.reindex(delta.index)["adv"].fillna(0.0)
    frame["daily_volatility"] = md.reindex(delta.index)["daily_volatility"].fillna(0.0)
    frame["beta"] = md.reindex(delta.index)["beta"].fillna(0.0) if "beta" in md.columns else 0.0
    frame["sector"] = md.reindex(delta.index)["sector"].fillna("UNKNOWN") if "sector" in md.columns else "UNKNOWN"
    frame["market_cap"] = md.reindex(delta.index)["market_cap"].fillna(0.0) if "market_cap" in md.columns else 0.0
    frame["trade_notional"] = delta.abs() * frame["open"]
    frame["short_notional"] = delta.clip(upper=0.0).abs() * frame["open"]
    return frame


def _constraint_config(config: BacktestConfig) -> PortfolioConstraints:
    return PortfolioConstraints(
        gross_limit=config.gross_limit,
        net_limit=config.net_limit,
        max_name_weight=config.max_name_weight,
        max_sector_weight=config.max_sector_weight,
        min_longs=config.min_longs,
        min_shorts=config.min_shorts,
        beta_target=config.beta_target,
        beta_tolerance=config.beta_tolerance,
        participation_cap=config.participation_cap,
    )


def _serialize_series(series: pd.Series) -> dict[str, float]:
    return {str(k): float(v) for k, v in series.sort_index().items()}


def _turnover_decomposition(row: pd.Series, previous_raw: pd.Series, constrained_target: pd.Series) -> dict[str, float]:
    active = [value for value in row if isinstance(value, pd.Series)]
    newest = active[-1] if active else pd.Series(dtype=float)
    survivors = pd.Series(dtype=float)
    for value in active[:-1]:
        survivors = survivors.add(value, fill_value=0.0)
    raw_aggregate = survivors.add(newest, fill_value=0.0)
    rolloff = previous_raw.sub(survivors, fill_value=0.0).abs().sum() if len(previous_raw) else 0.0
    signal_change = newest.abs().sum()
    internal_netting = sum(value.abs().sum() for value in active) - raw_aggregate.abs().sum()
    constraint_repair = constrained_target.sub(raw_aggregate, fill_value=0.0).abs().sum()
    return {
        "signal_change": float(signal_change),
        "tranche_rolloff": float(rolloff),
        "constraint_repair": float(constraint_repair),
        "internal_netting_effect": float(max(internal_netting, 0.0)),
    }


def run_backtest(spec: object, data: dict[str, object]) -> BacktestResult:
    config = data.get("config", BacktestConfig())
    assert isinstance(config, BacktestConfig)
    signals = data["signals"]
    assert isinstance(signals, dict)
    market_data = data["market_data"]
    assert isinstance(market_data, pd.DataFrame)
    blocked = data.get("blocked", pd.DataFrame(index=market_data.index.get_level_values("date").unique()))
    assert isinstance(blocked, pd.DataFrame)

    tranche_state = simulate_tranches(signals, holding_period=config.holding_period_days)
    intended_book = _aggregate_book(tranche_state)

    execution_dates = intended_book.index
    actual_book_rows: list[pd.Series] = []
    turnover_rows: list[float] = []
    gross_rows: list[float] = []
    net_rows: list[float] = []
    cost_rows: list[float] = []
    gross_pnl_rows: list[float] = []
    executed_dates: list[pd.Timestamp] = []
    current_book = pd.Series(dtype=float)
    previous_raw = pd.Series(dtype=float)
    constraint_events: list[dict[str, Any]] = []
    failure_days: list[str] = []
    book_history: dict[str, dict[str, float]] = {}
    target_history: dict[str, dict[str, float]] = {}
    turnover_decomposition: dict[str, dict[str, float]] = {}
    leg_pnl: dict[str, dict[str, float]] = {}
    exposure_summary: dict[str, dict[str, Any]] = {}
    capacity_stress: dict[str, dict[str, float]] = {}
    sensitivity: dict[str, Any] = {"cost_stress": {}, "execution_delay": {"baseline_days": 1, "optimistic_same_close_mode": config.optimistic_same_close_mode}}

    for i, signal_date in enumerate(execution_dates):
        execution_idx = i if config.optimistic_same_close_mode else i + 1
        if execution_idx >= len(execution_dates):
            break
        execution_date = execution_dates[execution_idx]
        target = intended_book.loc[signal_date].copy()
        blocked_names = _blocked_names_for_date(blocked, execution_date)
        if len(blocked_names) > 0:
            target.loc[target.index.intersection(blocked_names)] = current_book.reindex(target.index.intersection(blocked_names)).fillna(0.0)
            constraint_events.append({"date": execution_date.isoformat(), "constraint": "blocked_names", "action": "hold", "names": sorted(map(str, blocked_names.tolist()))})

        md = market_data.xs(execution_date).reindex(target.index)
        constrained = enforce_constraints(PortfolioWeights(target, execution_date), _constraint_config(config), md)
        for event in constrained.events:
            constraint_events.append({"date": execution_date.isoformat(), "constraint": event.constraint, "action": event.action, "before": event.before, "after": event.after, "reason": event.reason})
        if constrained.failed:
            failure_days.append(execution_date.isoformat())
            constraint_events.append({"date": execution_date.isoformat(), "constraint": "portfolio_constraints", "action": "failure", "reasons": constrained.failure_reasons})
            continue

        target = constrained.weights.weights
        target_history[execution_date.isoformat()] = _serialize_series(target)
        turnover_decomposition[execution_date.isoformat()] = _turnover_decomposition(tranche_state.loc[signal_date], previous_raw, target)
        previous_raw = intended_book.loc[signal_date].copy()

        delta = target.sub(current_book, fill_value=0.0)
        trade_frame = _build_trade_frame(delta, market_data, execution_date)
        costs = estimate_costs(trade_frame, trade_frame)
        day_slice = market_data.xs(execution_date)
        returns = (day_slice["close"] / day_slice["open"] - 1.0).reindex(target.index).fillna(0.0)
        pnl_by_name = target.mul(returns, fill_value=0.0)
        long_pnl = float(pnl_by_name[target > 0].sum())
        short_pnl = float(pnl_by_name[target < 0].sum())
        gross_pnl = float(pnl_by_name.sum())
        gross_long = float(target[target > 0].sum())
        gross_short = float(target[target < 0].abs().sum())
        leg_pnl[execution_date.isoformat()] = {
            "long_contribution": long_pnl,
            "short_contribution": short_pnl,
            "gross_long_exposure": gross_long,
            "gross_short_exposure": gross_short,
            "net_contribution": gross_pnl,
            "one_sided_pnl": float(abs(long_pnl) / (abs(short_pnl) + 1e-12)) if abs(short_pnl) > 0 else float("inf"),
        }
        sector_exposure = target.groupby(md["sector"].fillna("UNKNOWN")).sum() if "sector" in md.columns else pd.Series(dtype=float)
        effective_breadth = float((target.abs().sum() ** 2) / max((target.pow(2)).sum(), 1e-12)) if len(target) else 0.0
        exposure_summary[execution_date.isoformat()] = {
            "beta": float(target.mul(md["beta"].fillna(0.0)).sum()) if "beta" in md.columns else 0.0,
            "sector_exposures": {str(k): float(v) for k, v in sector_exposure.items()},
            "size_proxy": float(target.mul(pd.Series(pd.to_numeric(md.get("market_cap", 0.0), errors="coerce"), index=target.index).fillna(0.0)).sum()) if "market_cap" in md.columns else 0.0,
            "concentration": float(target.abs().max()) if len(target) else 0.0,
            "effective_breadth": effective_breadth,
        }
        participation = (trade_frame["trade_notional"] / trade_frame["adv"].replace(0, pd.NA)).fillna(0.0)
        base_total = float(costs.total_cost)
        stress_15 = float(stress_costs(costs, 1.5).total_cost)
        stress_20 = float(stress_costs(costs, 2.0).total_cost)
        capacity_stress[execution_date.isoformat()] = {
            "avg_adv_participation": float(participation.mean()) if len(participation) else 0.0,
            "max_adv_participation": float(participation.max()) if len(participation) else 0.0,
            "baseline_cost": base_total,
            "cost_1_5x": stress_15,
            "cost_2_0x": stress_20,
            "feasible_under_baseline": float((participation <= config.participation_cap).all()),
            "feasible_under_1_5x": float((participation * 1.5 <= config.participation_cap).all()),
            "feasible_under_2_0x": float((participation * 2.0 <= config.participation_cap).all()),
        }

        current_book = target
        book_history[execution_date.isoformat()] = _serialize_series(current_book)
        actual_book_rows.append(current_book.rename(execution_date))
        turnover_rows.append(float(delta.abs().sum()))
        gross_rows.append(float(current_book.abs().sum()))
        net_rows.append(float(current_book.sum()))
        cost_rows.append(base_total)
        gross_pnl_rows.append(gross_pnl)
        executed_dates.append(execution_date)

    book = pd.DataFrame(actual_book_rows).sort_index().fillna(0.0) if actual_book_rows else pd.DataFrame()
    gross_returns = pd.Series(gross_pnl_rows, index=executed_dates, dtype=float)
    costs_series = pd.Series(cost_rows, index=executed_dates, dtype=float)
    net_returns = gross_returns - costs_series
    turnover = pd.Series(turnover_rows, index=executed_dates, dtype=float)
    if len(costs_series):
        sensitivity["cost_stress"] = {
            "baseline_total_cost": float(costs_series.sum()),
            "cost_1_5x_total": float((costs_series * 1.5).sum()),
            "cost_2_0x_total": float((costs_series * 2.0).sum()),
        }
    candidate = CandidateConfig(candidate_id="baseline", params={"holding_period": config.holding_period_days})
    diagnostics: dict[str, Any] = {
        "status": "constraint_failures_present" if failure_days else "implemented_core_backtest",
        "book_rows": len(book),
        "tranche_rows": len(tranche_state),
        "avg_gross": float(pd.Series(gross_rows).mean()) if gross_rows else 0.0,
        "avg_net": float(pd.Series(net_rows).mean()) if net_rows else 0.0,
        "constraint_events": constraint_events,
        "failure_days": failure_days,
        "book_history": book_history,
        "target_history": target_history,
        "turnover_decomposition": turnover_decomposition,
        "leg_pnl": leg_pnl,
        "exposure_summary": exposure_summary,
        "capacity_stress": capacity_stress,
        "sensitivity": sensitivity,
    }
    return BacktestResult(candidate=candidate, gross_returns=gross_returns, net_returns=net_returns, turnover=turnover, costs=costs_series, diagnostics=diagnostics)
