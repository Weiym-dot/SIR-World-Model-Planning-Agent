from src.agent.goal_parser import GoalParser, MockStructuredLLM, parse_goal_rule_based


def test_aggressive_goal_changes_weights():
    goal = parse_goal_rule_based("Reduce infections aggressively over the next 10 days")
    assert goal.infection_weight > 1.0
    assert goal.intervention_cost_weight < 0.015
    assert goal.planning_horizon == 10


def test_cost_sensitive_goal_changes_weights():
    goal = parse_goal_rule_based("Minimize intervention cost over the next 8 days")
    assert goal.intervention_cost_weight > 0.015
    assert goal.planning_horizon == 8


def test_structured_llm_output_is_validated():
    mock = MockStructuredLLM({
        "infection_weight": 1.2,
        "intervention_cost_weight": 0.01,
        "planning_horizon": 9,
        "target_peak_fraction": None,
        "rationale": "test",
    })
    parsed = GoalParser(mock).parse("balanced control")
    assert parsed.planning_horizon == 9
    assert mock.last_prompt is not None
