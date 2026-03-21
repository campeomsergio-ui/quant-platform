import pandas as pd

from quant_platform.validation.multiple_testing import RealityCheckConfig, run_white_reality_check, stationary_bootstrap


def test_white_reality_check_returns_adjusted_pvalue() -> None:
    idx = pd.date_range("2022-01-01", periods=20, freq="B")
    bench = pd.Series(0.0, index=idx)
    candidates = {"a": pd.Series(0.01, index=idx), "b": pd.Series(0.0, index=idx)}
    result = run_white_reality_check(candidates, bench, RealityCheckConfig(bootstrap_iterations=10, seed=1))
    assert 0.0 <= result.adjusted_p_value <= 1.0


def test_stationary_bootstrap_preserves_length() -> None:
    idx = pd.date_range("2022-01-01", periods=20, freq="B")
    series = pd.Series(range(20), index=idx)
    sample = stationary_bootstrap(series, RealityCheckConfig(bootstrap_iterations=1, seed=1))[0]
    assert len(sample) == len(series)
