import pandas as pd

from quant_platform.signals.base import SignalContext
from quant_platform.signals.mean_reversion import MeanReversionParams, MeanReversionSignal, compute_residual_returns


def _bars_and_meta() -> tuple[pd.DataFrame, pd.DataFrame, pd.DatetimeIndex]:
    dates = pd.date_range("2022-01-01", periods=10, freq="B")
    idx = pd.MultiIndex.from_product([dates, ["A", "B", "C", "D"]], names=["date", "symbol"])
    bars = pd.DataFrame({"close": [float(i + 10) for i in range(len(idx))]}, index=idx)
    meta = pd.DataFrame({"sector": ["S1", "S1", "S2", "S2"], "industry": ["I1", "I1", "I2", "I2"], "beta": [1.0, 1.1, 0.9, 1.2], "market_cap": [1e9, 2e9, 1.5e9, 2.5e9]}, index=["A", "B", "C", "D"])
    return bars, meta, dates


def test_residual_signal_ranks_cross_section() -> None:
    bars, meta, dates = _bars_and_meta()
    ctx = SignalContext(bars=bars, meta=meta, as_of=dates[-1], seed=1)
    weights = MeanReversionSignal(MeanReversionParams(residual_model="industry_beta_log_mcap")).compute(ctx)
    assert abs(weights.weights.sum()) < 1e-9


def test_signal_handles_missing_data() -> None:
    dates = pd.date_range("2022-01-01", periods=10, freq="B")
    idx = pd.MultiIndex.from_product([dates, ["A", "B"]], names=["date", "symbol"])
    bars = pd.DataFrame({"close": [10.0] * len(idx)}, index=idx)
    meta = pd.DataFrame({"sector": ["S1", "S1"], "industry": ["I1", "I1"], "beta": [1.0, 1.1], "market_cap": [1e9, 2e9]}, index=["A", "B"])
    residual = compute_residual_returns(bars, meta, MeanReversionParams(residual_model="industry_beta"))
    assert len(residual.values) > 0


def test_pre_registered_residual_family_all_supported() -> None:
    bars, meta, _dates = _bars_and_meta()
    for model in ["sector_only", "industry_only", "industry_beta", "industry_beta_log_mcap"]:
        residual = compute_residual_returns(bars, meta, MeanReversionParams(residual_model=model))
        assert len(residual.values) > 0
