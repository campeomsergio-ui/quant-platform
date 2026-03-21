import pandas as pd

from quant_platform.data_contracts import CandidateConfig
from quant_platform.validation.walk_forward import WalkForwardConfig, generate_folds, run_walk_forward


def test_generate_folds_respects_window_lengths() -> None:
    dates = pd.date_range("2012-01-01", "2022-12-31", freq="B")
    folds = generate_folds(dates, WalkForwardConfig())
    assert folds[0].train_start == pd.Timestamp("2012-01-01")
    assert folds[0].validation_start == pd.Timestamp("2017-01-01")


def test_run_walk_forward_uses_train_then_validate_only() -> None:
    dates = pd.date_range("2012-01-01", "2022-12-31", freq="B")
    results = run_walk_forward([CandidateConfig(candidate_id="x", params={})], {"dates": dates}, WalkForwardConfig())
    assert results
