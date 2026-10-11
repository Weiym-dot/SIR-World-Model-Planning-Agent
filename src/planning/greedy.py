from __future__ import annotations
from dataclasses import dataclass
from src.schemas import SIRState, InterventionAction
from src.world_model.base import WorldModel
from src.planning.reward import step_reward
from src.grounding.verifier import verify_transition

@dataclass
class Trajectory:
    states: list[SIRState]
    actions: list[InterventionAction]
    rewards: list[float]


def greedy_plan(
    world_model: WorldModel,
    max_steps: int = 60,
    step_days: float = 1.0,
) -> Trajectory:
    state = world_model.init_state()
    states = [state]
    actions: list[InterventionAction] = []
    rewards: list[float] = []

    for _ in range(max_steps):
        if world_model.is_terminal(state):
            break

        candidates = []
        for action in InterventionAction:
            result = world_model.step(state, action, duration=step_days)
            expected_population = state.population
            report = verify_transition(result, expected_population)
            if not report.valid:
                continue
            r = step_reward(result.next_state, action)
            candidates.append((r, action, result.next_state))

        if not candidates:
            raise RuntimeError("No valid actions available")

        best_reward, best_action, best_state = max(candidates, key=lambda x: x[0])
        actions.append(best_action)
        rewards.append(best_reward)
        states.append(best_state)
        state = best_state

    return Trajectory(states=states, actions=actions, rewards=rewards)
