from __future__ import annotations
from src.schemas import TransitionResult
from src.grounding.constraints import ConstraintReport, check_sir_constraints


def verify_transition(result: TransitionResult, expected_population: float) -> ConstraintReport:
    return check_sir_constraints(result.next_state, expected_population=expected_population)
