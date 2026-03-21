from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class BenchmarkSeries:
    name: str
    returns: pd.Series
