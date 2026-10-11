from __future__ import annotations
from dataclasses import dataclass
from src.schemas import SIRState

@dataclass
class ConstraintReport:
    valid: bool
    errors: list[str]


def check_sir_constraints(state: SIRState, expected_population: float, tol: float = 1e-3) -> ConstraintReport:
    errors: list[str] = []
    if state.S < 0 or state.I < 0 or state.R < 0:
        errors.append("S, I, and R must all be non-negative.")

    if abs(state.population - expected_population) > tol * max(1.0, expected_population):
        errors.append(
            f"Population is not conserved: got {state.population:.6f}, expected {expected_population:.6f}."
        )

    return ConstraintReport(valid=(len(errors) == 0), errors=errors)
