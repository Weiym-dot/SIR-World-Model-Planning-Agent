from __future__ import annotations

from src.agent.scientific_agent import ScientificPlanningAgent
from src.evaluation.metrics import compute_metrics
from src.planning.greedy import Trajectory, greedy_plan
from src.schemas import InterventionAction
from src.world_model.sir import SIRWorldModel


def rollout_fixed(
    model: SIRWorldModel,
    action: InterventionAction,
    steps: int = 30,
) -> Trajectory:
    state = model.init_state()
    states = [state]
    actions: list[InterventionAction] = []
    rewards: list[float] = []
    for _ in range(steps):
        if model.is_terminal(state):
            break
        result = model.step(state, action)
        state = result.next_state
        states.append(state)
        actions.append(action)
        rewards.append(0.0)
    return Trajectory(states, actions, rewards)


def compare_default_strategies(
    model: SIRWorldModel,
    natural_language_goal: str,
    steps: int = 30,
    mcts_iterations: int = 80,
) -> dict[str, dict]:
    agent = ScientificPlanningAgent(model, base_iterations=mcts_iterations)
    runs = {
        "none": rollout_fixed(model, InterventionAction.NONE, steps),
        "strong": rollout_fixed(model, InterventionAction.STRONG, steps),
        "greedy": greedy_plan(model, max_steps=steps),
        "goal_conditioned_mcts": agent.run(natural_language_goal, max_steps=steps).trajectory,
    }
    return {name: compute_metrics(traj).to_dict() for name, traj in runs.items()}
