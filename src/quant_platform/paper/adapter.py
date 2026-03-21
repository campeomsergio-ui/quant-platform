from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd


@dataclass(frozen=True)
class PaperOrder:
    symbol: str
    target_weight_delta: float


@dataclass(frozen=True)
class PaperFill:
    symbol: str
    filled_weight_delta: float
    fill_price: float
    status: str


class PaperBrokerBoundary(Protocol):
    def simulate_orders(self, orders: list[PaperOrder], market_slice: pd.DataFrame) -> list[PaperFill]: ...


@dataclass(frozen=True)
class PaperBrokerAdapter:
    name: str = "paper"

    def simulate_orders(self, orders: list[PaperOrder], market_slice: pd.DataFrame) -> list[PaperFill]:
        fills: list[PaperFill] = []
        for order in orders:
            price = float(market_slice.reindex([order.symbol])["open"].fillna(0.0).iloc[0])
            status = "filled" if price > 0 else "rejected"
            filled_delta = order.target_weight_delta if status == "filled" else 0.0
            fills.append(PaperFill(symbol=order.symbol, filled_weight_delta=filled_delta, fill_price=price, status=status))
        return fills
