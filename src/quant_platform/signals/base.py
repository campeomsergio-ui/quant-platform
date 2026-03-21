from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from quant_platform.data_contracts import PortfolioWeights


@dataclass(frozen=True)
class SignalContext:
    bars: pd.DataFrame
    meta: pd.DataFrame
    as_of: pd.Timestamp
    seed: int


class SignalModel(Protocol):
    def compute(self, context: SignalContext) -> PortfolioWeights: ...
