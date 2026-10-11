from __future__ import annotations

from dataclasses import dataclass

from src.grounding.verifier import verify_transition
from src.schemas import InterventionAction, SIRState, TransitionResult
from src.world_model.base import WorldModel


@dataclass
class SimulationToolResult:
    transition: TransitionResult
    scientifically_valid: bool
    errors: list[str]


def simulate_intervention(
    world_model: WorldModel,
    state: SIRState,
    action: InterventionAction,
    duration: float = 1.0,
) -> SimulationToolResult:
    """A tool-like wrapper around the scientific world model."""
    transition = world_model.step(state, action, duration=duration)
    report = verify_transition(transition, expected_population=state.population)
    return SimulationToolResult(
        transition=transition,
        scientifically_valid=report.valid,
        errors=report.errors,
    )
