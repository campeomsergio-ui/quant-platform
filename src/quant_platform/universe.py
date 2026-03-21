from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class UniverseBuilder:
    max_names: int = 1000
    price_min: float = 5.0
    market_cap_min: float = 500_000_000.0
    adv_min: float = 5_000_000.0
    lookback_days: int = 60
    min_history_days: int = 252
    monthly_reconstitution: bool = True
    entry_buffer_names: int = 50
    exit_buffer_names: int = 50


def apply_security_exclusions(meta: pd.DataFrame, as_of: pd.Timestamp) -> list[str]:
    eligible = meta.copy()
    if "security_type" in eligible.columns:
        disallowed = {"ADR", "ETF", "CEF", "SPAC", "WARRANT", "PREFERRED"}
        eligible = eligible.loc[~eligible["security_type"].isin(disallowed)]
    if "is_primary_listing" in eligible.columns:
        eligible = eligible.loc[eligible["is_primary_listing"]]
    return eligible.index.astype(str).tolist()


def apply_liquidity_filters(bars: pd.DataFrame, meta: pd.DataFrame, as_of: pd.Timestamp) -> list[str]:
    eligible = apply_security_exclusions(meta, as_of)
    frame = bars.loc[bars.index.get_level_values("date") <= as_of]
    frame = frame.loc[frame.index.get_level_values("symbol").isin(eligible)]
    if frame.empty:
        return []
    recent = frame.groupby("symbol").tail(60)
    adv = (recent["close"] * recent["volume"]).groupby(level="symbol").median()
    px = recent.groupby(level="symbol")["close"].last()
    counts = frame.groupby(level="symbol").size()
    filt = meta.reindex(adv.index).copy()
    filt["adv"] = adv
    filt["price"] = px
    filt["history_count"] = counts.reindex(adv.index).fillna(0)
    selected = filt.loc[
        (filt["market_cap"] >= 500_000_000.0)
        & (filt["adv"] >= 5_000_000.0)
        & (filt["price"] > 5.0)
        & (filt["history_count"] >= 252)
    ]
    return selected.sort_values("adv", ascending=False).head(1000).index.astype(str).tolist()


def build_universe(bars: pd.DataFrame, meta: pd.DataFrame, as_of: pd.Timestamp) -> list[str]:
    return apply_liquidity_filters(bars, meta, as_of)
