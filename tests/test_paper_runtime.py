from pathlib import Path

import pandas as pd

from cli import main
from quant_platform.monitoring.kill_switch import KillSwitchState
from quant_platform.paper.adapter import PaperBrokerAdapter
from quant_platform.paper.execution import run_paper_cycle
from quant_platform.paper.runtime import PaperRuntimeConfig, load_runtime_state, run_daily_paper_cycle
from quant_platform.sample_data import write_sample_daily_equities_dataset
from quant_platform.signals.mean_reversion import MeanReversionParams, MeanReversionSignal
from quant_platform.data_access import LocalJsonDataAdapter


def _bars() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.date_range("2022-01-03", periods=8, freq="B")
    idx = pd.MultiIndex.from_product([dates, ["A", "B", "C", "D"]], names=["date", "symbol"])
    rows = []
    price = 100.0
    for _ in idx:
        rows.append({"open": price, "close": price + 1.0})
        price += 0.5
    bars = pd.DataFrame(rows, index=idx)
    meta = pd.DataFrame(
        {"sector": ["S1", "S1", "S2", "S2"], "industry": ["I1", "I1", "I2", "I2"], "beta": [1.0, 1.1, 0.9, 1.2], "market_cap": [1e9, 2e9, 1.5e9, 2.5e9]},
        index=["A", "B", "C", "D"],
    )
    return bars, meta


def test_paper_trading_state_transitions() -> None:
    target = pd.Series({"A": 0.01})
    current = pd.Series(dtype=float)
    market = pd.DataFrame({"open": [100.0]}, index=["A"])
    adapter = PaperBrokerAdapter()
    result = run_paper_cycle(target, current, market, adapter, KillSwitchState(enabled=False, reason=""))
    assert result.status == "ok"
    assert result.reconciliation.status == "matched"


def test_paper_runtime_respects_kill_switch() -> None:
    bars, meta = _bars()
    signal = MeanReversionSignal(MeanReversionParams(residual_model="sector_only"))
    runtime = run_daily_paper_cycle(bars, meta, signal, PaperBrokerAdapter(), PaperRuntimeConfig(), manual_kill_flag=True)
    assert runtime.status == "killed"


def test_multi_day_runtime_state_transitions_and_readback(tmp_path: Path) -> None:
    data_root = tmp_path / "sample"
    state_path = tmp_path / "state.json"
    report_path = tmp_path / "report.json"
    write_sample_daily_equities_dataset(str(data_root))
    bundle = LocalJsonDataAdapter(str(data_root)).load_bundle()
    signal = MeanReversionSignal(MeanReversionParams(residual_model="industry_beta_log_mcap"))
    result1 = run_daily_paper_cycle(bundle.bars, bundle.metadata, signal, PaperBrokerAdapter(), PaperRuntimeConfig(state_path=str(state_path), report_path=str(report_path)))
    state1 = load_runtime_state(str(state_path))
    result2 = run_daily_paper_cycle(bundle.bars, bundle.metadata, signal, PaperBrokerAdapter(), PaperRuntimeConfig(state_path=str(state_path), report_path=str(report_path), dry_run=True))
    state2 = load_runtime_state(str(state_path))
    assert result1.state.last_run_date is not None
    assert state1.last_run_date == result1.state.last_run_date
    assert state2.last_run_date == result2.state.last_run_date


def test_persistence_corruption_raises(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    state_path.write_text('{"bad": true}')
    try:
        load_runtime_state(str(state_path))
    except ValueError:
        assert True
    else:
        assert False


def test_report_generation_and_reconciliation_payload(tmp_path: Path) -> None:
    data_root = tmp_path / "sample"
    state_path = tmp_path / "state.json"
    report_path = tmp_path / "report.json"
    write_sample_daily_equities_dataset(str(data_root))
    bundle = LocalJsonDataAdapter(str(data_root)).load_bundle()
    signal = MeanReversionSignal(MeanReversionParams(residual_model="industry_beta_log_mcap"))
    result = run_daily_paper_cycle(bundle.bars, bundle.metadata, signal, PaperBrokerAdapter(), PaperRuntimeConfig(state_path=str(state_path), report_path=str(report_path)))
    assert result.report is not None
    assert "reconciliation" in result.report.payload
    assert "long_short_attribution" in result.report.payload["diagnostics"]
    assert "turnover_decomposition" in result.report.payload["diagnostics"]
    assert report_path.exists()


def test_kill_switch_blocks_on_missing_data(tmp_path: Path) -> None:
    dates = pd.date_range("2022-01-03", periods=1, freq="B")
    idx = pd.MultiIndex.from_product([dates, ["A"]], names=["date", "symbol"])
    bars = pd.DataFrame({"open": [100.0], "close": [101.0]}, index=idx)
    meta = pd.DataFrame({"sector": ["S1"], "industry": ["I1"], "beta": [1.0], "market_cap": [1e9]}, index=["A"])
    signal = MeanReversionSignal(MeanReversionParams(residual_model="sector_only"))
    result = run_daily_paper_cycle(bars, meta, signal, PaperBrokerAdapter(), PaperRuntimeConfig(state_path=str(tmp_path / "state.json"), report_path=str(tmp_path / "report.json")))
    assert result.status == "blocked"
    assert result.state.kill_switch_state["enabled"] is True


def test_cli_paper_runtime_path(tmp_path: Path) -> None:
    data_root = tmp_path / "sample"
    state_path = tmp_path / "state.json"
    report_path = tmp_path / "report.json"
    write_sample_daily_equities_dataset(str(data_root))
    rc = main(["run-paper-daily", "--data-root", str(data_root), "--state-path", str(state_path), "--report-path", str(report_path), "--dry-run"])
    assert rc == 0
    assert state_path.exists()
    assert report_path.exists()
