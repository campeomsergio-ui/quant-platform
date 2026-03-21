from strategy_spec import load_strategy_spec, validate_strategy_spec


def test_load_strategy_spec_minimal() -> None:
    spec = load_strategy_spec({
        "name": "x",
        "universe_definition": ["a"],
        "data_requirements": ["a"],
        "signal_definition": ["a"],
        "holding_period": "5d",
        "rebalancing_rule": ["a"],
        "transaction_cost_model": ["a"],
        "risk_model": ["a"],
        "constraints": ["a"],
        "hyperparameters": ["a"],
        "validation_plan": ["a"],
        "failure_modes": ["a"],
        "minimum_acceptance_criteria": ["a"],
    })
    validate_strategy_spec(spec)
    assert spec.name == "x"


def test_validate_strategy_spec_rejects_invalid_constraints() -> None:
    spec = load_strategy_spec({
        "name": "x",
        "universe_definition": ["a"],
        "data_requirements": ["a"],
        "signal_definition": ["a"],
        "holding_period": "5d",
        "rebalancing_rule": ["a"],
        "transaction_cost_model": ["a"],
        "risk_model": ["a"],
        "constraints": [],
        "hyperparameters": ["a"],
        "validation_plan": ["a"],
        "failure_modes": ["a"],
        "minimum_acceptance_criteria": ["a"],
    })
    try:
        validate_strategy_spec(spec)
    except ValueError:
        assert True
    else:
        assert False
