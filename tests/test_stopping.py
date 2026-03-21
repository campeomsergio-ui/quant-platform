from stopping import evaluate_stop_rules


def test_evaluate_stop_rules_stops_on_test_touch() -> None:
    decision = evaluate_stop_rules(None, {"final_test_touched": True})
    assert decision.should_stop


def test_evaluate_stop_rules_stops_on_high_pbo() -> None:
    decision = evaluate_stop_rules(None, {"pbo": 0.3})
    assert decision.should_stop
