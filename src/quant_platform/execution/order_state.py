from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OrderState:
    order_id: str
    symbol: str
    quantity: float
    status: str
