import pandas as pd

from quant_platform.data_contracts import PortfolioWeights
from quant_platform.portfolio import PortfolioConstraints, enforce_constraints


def _market(symbols: list[str], sectors: dict[str, str] | None = None, betas: dict[str, float] | None = None, advs: dict[str, float] | None = None) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sector": [(sectors or {}).get(symbol, "UNKNOWN") for symbol in symbols],
            "beta": [(betas or {}).get(symbol, 0.0) for symbol in symbols],
            "adv": [(advs or {}).get(symbol, 10_000.0) for symbol in symbols],
            "open": [100.0 for _ in symbols],
        },
        index=symbols,
    )


def test_sector_cap_breach() -> None:
    weights = PortfolioWeights(pd.Series({"A": 0.02, "B": -0.01, "C": -0.01}))
    market = _market(["A", "B", "C"], sectors={"A": "Tech", "B": "Tech", "C": "Health"})
    result = enforce_constraints(weights, PortfolioConstraints(max_sector_weight=0.015, min_longs=1, min_shorts=2), market)
    assert any(event.constraint == "sector_cap" for event in result.events)


def test_insufficient_breadth() -> None:
    weights = PortfolioWeights(pd.Series({"A": 0.01, "B": -0.01}))
    market = _market(["A", "B"])
    result = enforce_constraints(weights, PortfolioConstraints(min_longs=2, min_shorts=2, max_sector_weight=1.0), market)
    assert result.failed
    assert any("insufficient longs" in reason or "insufficient shorts" in reason for reason in result.failure_reasons)


def test_beta_neutralization_behavior() -> None:
    weights = PortfolioWeights(pd.Series({"A": 0.01, "B": -0.005, "C": -0.005}))
    market = _market(["A", "B", "C"], betas={"A": 2.0, "B": 0.5, "C": 0.5})
    result = enforce_constraints(weights, PortfolioConstraints(min_longs=1, min_shorts=2, max_sector_weight=1.0), market)
    assert any(event.constraint == "beta_neutralization" for event in result.events)


def test_liquidity_cap_breach() -> None:
    weights = PortfolioWeights(pd.Series({"A": 0.03, "B": -0.03}))
    market = _market(["A", "B"], advs={"A": 1.0, "B": 1.0})
    result = enforce_constraints(weights, PortfolioConstraints(participation_cap=0.01, min_longs=1, min_shorts=1, max_sector_weight=1.0, max_name_weight=0.03), market)
    assert any(event.constraint == "liquidity_cap" for event in result.events)


def test_gross_net_limit_enforcement_after_netting() -> None:
    weights = PortfolioWeights(pd.Series({"A": 0.08, "B": 0.08, "C": -0.02}))
    market = _market(["A", "B", "C"])
    result = enforce_constraints(weights, PortfolioConstraints(gross_limit=0.03, net_limit=0.0, min_longs=1, min_shorts=1, max_sector_weight=1.0, max_name_weight=0.08), market)
    constraints = {event.constraint for event in result.events}
    assert "gross_limit" in constraints or "net_limit" in constraints


def test_unsatisfiable_portfolio_case() -> None:
    weights = PortfolioWeights(pd.Series({"A": 0.05}))
    market = _market(["A"], sectors={"A": "Tech"}, betas={"A": 2.0})
    result = enforce_constraints(weights, PortfolioConstraints(min_longs=1, min_shorts=1, max_sector_weight=0.001, net_limit=0.0), market)
    assert result.failed
