from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quant_platform.costs import estimate_costs
from quant_platform.data_contracts import BacktestResult, CandidateConfig


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
    seed: int = 0


@dataclass(frozen=True)
class BacktestEngine:
    config: BacktestConfig


def _normalize_total_book(target: pd.Series, config: BacktestConfig) -> pd.Series:
    clipped = target.clip(lower=-config.max_name_weight, upper=config.max_name_weight)
    gross = clipped.abs().sum()
    if gross > config.gross_limit and gross > 0:
        clipped = clipped * (config.gross_limit / gross)
    net = clipped.sum()
    if abs(net) > config.net_limit and len(clipped) > 0:
        clipped = clipped - (net / len(clipped))
    return clipped.fillna(0.0)


def simulate_tranches(signals: dict[pd.Timestamp, pd.Series], holding_period: int) -> pd.DataFrame:
    ordered_dates = sorted(signals)
    columns = [entry_date.strftime("%Y-%m-%d") for entry_date in ordered_dates[:holding_period]]
    frame = pd.DataFrame(index=ordered_dates, columns=columns, dtype=object)
    for active_date in ordered_dates:
        active_idx = ordered_dates.index(active_date)
        for entry_date in ordered_dates[max(0, active_idx - holding_period + 1): active_idx + 1]:
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
    frame["trade_notional"] = delta.abs() * frame["open"]
    frame["short_notional"] = (delta.clip(upper=0.0).abs()) * frame["open"]
    return frame


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

    for i, signal_date in enumerate(execution_dates):
        execution_idx = i if config.optimistic_same_close_mode else i + 1
        if execution_idx >= len(execution_dates):
            break
        execution_date = execution_dates[execution_idx]
        target = intended_book.loc[signal_date].copy()
        blocked_names = _blocked_names_for_date(blocked, execution_date)
        if len(blocked_names) > 0:
            target.loc[target.index.intersection(blocked_names)] = current_book.reindex(target.index.intersection(blocked_names)).fillna(0.0)
        target = _normalize_total_book(target, config)
        delta = target.sub(current_book, fill_value=0.0)
        trade_frame = _build_trade_frame(delta, market_data, execution_date)
        participation = (trade_frame["trade_notional"] / trade_frame["adv"].replace(0, pd.NA)).fillna(0.0)
        capped = participation > config.participation_cap
        if capped.any():
            scale = (config.participation_cap / participation[capped]).clip(upper=1.0)
            delta.loc[capped.index[capped]] = delta.loc[capped.index[capped]] * scale
            target = current_book.add(delta, fill_value=0.0)
            target = _normalize_total_book(target, config)
            delta = target.sub(current_book, fill_value=0.0)
            trade_frame = _build_trade_frame(delta, market_data, execution_date)
        costs = estimate_costs(trade_frame, trade_frame)
        next_return_date = execution_date
        day_slice = market_data.xs(next_return_date)
        returns = (day_slice["close"] / day_slice["open"] - 1.0).reindex(target.index).fillna(0.0)
        gross_pnl = float(target.mul(returns, fill_value=0.0).sum())
        current_book = target
        actual_book_rows.append(current_book.rename(execution_date))
        turnover_rows.append(float(delta.abs().sum()))
        gross_rows.append(float(current_book.abs().sum()))
        net_rows.append(float(current_book.sum()))
        cost_rows.append(float(costs.total_cost))
        gross_pnl_rows.append(gross_pnl)
        executed_dates.append(execution_date)

    book = pd.DataFrame(actual_book_rows).sort_index().fillna(0.0) if actual_book_rows else pd.DataFrame()
    gross_returns = pd.Series(gross_pnl_rows, index=executed_dates, dtype=float)
    costs_series = pd.Series(cost_rows, index=executed_dates, dtype=float)
    net_returns = gross_returns - costs_series
    turnover = pd.Series(turnover_rows, index=executed_dates, dtype=float)
    candidate = CandidateConfig(candidate_id="baseline", params={"holding_period": config.holding_period_days})
    diagnostics = {
        "status": "implemented_core_backtest",
        "book_rows": len(book),
        "tranche_rows": len(tranche_state),
        "avg_gross": float(pd.Series(gross_rows).mean()) if gross_rows else 0.0,
        "avg_net": float(pd.Series(net_rows).mean()) if net_rows else 0.0,
    }
    return BacktestResult(candidate=candidate, gross_returns=gross_returns, net_returns=net_returns, turnover=turnover, costs=costs_series, diagnostics=diagnostics)
