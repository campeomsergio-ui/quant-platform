import pandas as pd

from quant_platform.monitoring.kill_switch import KillSwitchState
from quant_platform.paper.adapter import PaperBrokerAdapter
from quant_platform.paper.execution import run_paper_cycle
from quant_platform.paper.runtime import PaperRuntimeConfig, run_paper_event_loop
from quant_platform.signals.mean_reversion import MeanReversionParams, MeanReversionSignal


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
    runtime = run_paper_event_loop(bars, meta, signal, PaperBrokerAdapter(), PaperRuntimeConfig(), KillSwitchState(enabled=True, reason="manual"))
    assert len(runtime.cycles) <= 1
