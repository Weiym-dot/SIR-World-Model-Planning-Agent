from src.agent.scientific_agent import ScientificPlanningAgent
from src.evaluation.metrics import compute_metrics
from src.world_model.sir import SIRWorldModel


def test_agent_runs_and_preserves_scientific_constraints():
    model = SIRWorldModel(horizon_days=8)
    agent = ScientificPlanningAgent(model, base_iterations=15, seed=3)
    run = agent.run("Balance infection control with intervention cost over the next 5 days", max_steps=5)
    metrics = compute_metrics(run.trajectory)
    assert len(run.trajectory.actions) == 5
    assert metrics.scientific_violations == 0
    assert run.goal.planning_horizon == 5
    assert len(run.logs) == 5
