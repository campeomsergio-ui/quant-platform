from pathlib import Path

from cli import main
from quant_platform.io import load_json


def test_cli_skeleton_accepts_spec_and_plan_paths(tmp_path: Path) -> None:
    strategy = tmp_path / "strategy.json"
    experiment = tmp_path / "experiment.json"
    blueprint = tmp_path / "blueprint.json"
    registry = tmp_path / "registry.json"
    strategy.write_text('{"name":"x","universe_definition":["a"],"data_requirements":["a"],"signal_definition":["a"],"holding_period":"5d","rebalancing_rule":["a"],"transaction_cost_model":["a"],"risk_model":["a"],"constraints":["a"],"hyperparameters":["a"],"validation_plan":["a"],"failure_modes":["a"],"minimum_acceptance_criteria":["a"]}')
    experiment.write_text('{"splits":["a"],"sweeps":["a"],"multiple_testing_control":["a"],"overfitting_estimation":["a"],"metrics":["a"],"diagnostics":["a"],"stop_rules":["a"]}')
    blueprint.write_text('{"files":[],"dependencies":[],"invariants":[],"tests":[]}')
    assert main(["spec-check", "--strategy", str(strategy), "--experiment", str(experiment)]) == 0
    assert main(["run-experiment", "--strategy", str(strategy), "--experiment", str(experiment), "--blueprint", str(blueprint), "--registry", str(registry), "--candidate", "c1"]) == 0
    payload = load_json(str(registry))
    experiment_id = next(iter(payload["experiments"].keys()))
    assert payload["experiments"][experiment_id]["candidates"][0]["candidate_id"] == "c1"


def test_locked_test_enforcement(tmp_path: Path) -> None:
    strategy = tmp_path / "strategy.json"
    experiment = tmp_path / "experiment.json"
    blueprint = tmp_path / "blueprint.json"
    registry = tmp_path / "registry.json"
    strategy.write_text('{"name":"x","universe_definition":["a"],"data_requirements":["a"],"signal_definition":["a"],"holding_period":"5d","rebalancing_rule":["a"],"transaction_cost_model":["a"],"risk_model":["a"],"constraints":["a"],"hyperparameters":["a"],"validation_plan":["a"],"failure_modes":["a"],"minimum_acceptance_criteria":["a"]}')
    experiment.write_text('{"splits":["a"],"sweeps":["a"],"multiple_testing_control":["a"],"overfitting_estimation":["a"],"metrics":["a"],"diagnostics":["a"],"stop_rules":["a"]}')
    blueprint.write_text('{"files":[],"dependencies":[],"invariants":[],"tests":[]}')
    assert main(["run-experiment", "--strategy", str(strategy), "--experiment", str(experiment), "--blueprint", str(blueprint), "--registry", str(registry), "--touch-final-test"]) == 0
    try:
        main(["run-experiment", "--strategy", str(strategy), "--experiment", str(experiment), "--blueprint", str(blueprint), "--registry", str(registry), "--touch-final-test"])
    except ValueError:
        assert True
    else:
        assert False
