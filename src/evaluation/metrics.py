from __future__ import annotations

from dataclasses import asdict, dataclass

from src.grounding.constraints import check_sir_constraints
from src.planning.greedy import Trajectory
from src.planning.reward import INTERVENTION_COST
from src.schemas import InterventionAction


@dataclass
class TrajectoryMetrics:
    steps: int
    peak_infected: float
    peak_infected_fraction: float
    final_infected: float
    cumulative_infected_fraction: float
    intervention_cost: float
    none_days: int
    mild_days: int
    strong_days: int
    scientific_violations: int

    def to_dict(self) -> dict:
        return asdict(self)


def compute_metrics(trajectory: Trajectory) -> TrajectoryMetrics:
    population = trajectory.states[0].population
    peak = max(s.I for s in trajectory.states)
    cumulative = sum(s.I / population for s in trajectory.states[1:])
    cost = sum(INTERVENTION_COST[a] for a in trajectory.actions)

    violations = 0
    for state in trajectory.states:
        report = check_sir_constraints(state, expected_population=population)
        violations += 0 if report.valid else 1

    return TrajectoryMetrics(
        steps=len(trajectory.actions),
        peak_infected=peak,
        peak_infected_fraction=peak / population,
        final_infected=trajectory.states[-1].I,
        cumulative_infected_fraction=cumulative,
        intervention_cost=cost,
        none_days=trajectory.actions.count(InterventionAction.NONE),
        mild_days=trajectory.actions.count(InterventionAction.MILD),
        strong_days=trajectory.actions.count(InterventionAction.STRONG),
        scientific_violations=violations,
    )
