import pandas as pd

from quant_platform.validation.overfitting import OverfittingConfig, compute_rank_stability, estimate_pbo


def test_estimate_pbo_bounds_between_zero_and_one() -> None:
    frame = pd.DataFrame({"a": [0.1, -0.1], "b": [0.05, 0.02]})
    result = estimate_pbo(frame, OverfittingConfig())
    assert 0.0 <= result.probability_of_backtest_overfitting <= 1.0


def test_compute_rank_stability_returns_correlation_like_value() -> None:
    frame = pd.DataFrame({"a": [0.1, -0.1], "b": [0.05, 0.02]})
    value = compute_rank_stability(frame)
    assert -1.0 <= value <= 1.0
