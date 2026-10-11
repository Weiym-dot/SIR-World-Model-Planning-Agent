from __future__ import annotations
from dataclasses import dataclass
from src.schemas import SIRState, InterventionAction, ScientificGoal

INTERVENTION_COST = {
    InterventionAction.NONE: 0.0,
    InterventionAction.MILD: 1.0,
    InterventionAction.STRONG: 3.0,
}

@dataclass
class RewardWeights:
    infected_weight: float = 1.0
    intervention_weight: float = 15.0
    success_bonus: float = 500.0


@dataclass(frozen=True)
class PlanningObjective:
    """Normalized utility parameters used inside MCTS."""

    infection_weight: float = 1.0
    intervention_cost_weight: float = 0.015
    target_peak_fraction: float | None = None
    target_bonus: float = 0.02


def step_reward(
    state: SIRState,
    action: InterventionAction,
    goal: ScientificGoal | None = None,
    weights: RewardWeights | None = None,
) -> float:
    """Higher is better.

    Penalize infections and intervention cost; optionally reward satisfying a target.
    """
    weights = weights or RewardWeights()
    reward = -weights.infected_weight * state.I
    reward -= weights.intervention_weight * INTERVENTION_COST[action]

    if goal is not None and goal.max_final_infected is not None and state.I <= goal.max_final_infected:
        reward += weights.success_bonus
    return reward


def trajectory_reward(states: list[SIRState], actions: list[InterventionAction], weights: RewardWeights | None = None) -> float:
    if len(states) != len(actions) + 1:
        raise ValueError("states must contain the initial state plus one state per action")
    weights = weights or RewardWeights()
    peak_infected = max(s.I for s in states)
    intervention_cost = sum(INTERVENTION_COST[a] for a in actions)
    return -peak_infected - weights.intervention_weight * intervention_cost


def mcts_step_utility(
    state: SIRState,
    action: InterventionAction,
    population: float,
    intervention_cost_weight: float = 0.015,
    infection_weight: float = 1.0,
    target_peak_fraction: float | None = None,
    target_bonus: float = 0.02,
) -> float:
    """Normalized one-step utility for MCTS.

    The objective is intentionally separate from the world model: dynamics say
    what *will* happen; utility says which outcomes the agent *prefers*.
    """
    infection_fraction = state.I / population
    intervention_penalty = intervention_cost_weight * INTERVENTION_COST[action]
    utility = -infection_weight * infection_fraction - intervention_penalty

    if target_peak_fraction is not None and infection_fraction <= target_peak_fraction:
        utility += target_bonus
    return utility


def objective_utility(
    state: SIRState,
    action: InterventionAction,
    population: float,
    objective: PlanningObjective,
) -> float:
    return mcts_step_utility(
        state,
        action,
        population=population,
        intervention_cost_weight=objective.intervention_cost_weight,
        infection_weight=objective.infection_weight,
        target_peak_fraction=objective.target_peak_fraction,
        target_bonus=objective.target_bonus,
    )
