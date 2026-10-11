from __future__ import annotations
from enum import Enum
from pydantic import BaseModel, Field, model_validator

class InterventionAction(str, Enum):
    NONE = "none"
    MILD = "mild"
    STRONG = "strong"

class SIRState(BaseModel):
    t: float = Field(ge=0)
    S: float = Field(ge=0)
    I: float = Field(ge=0)
    R: float = Field(ge=0)

    @property
    def population(self) -> float:
        return self.S + self.I + self.R

class ScientificGoal(BaseModel):
    max_peak_infected: float | None = Field(default=None, ge=0)
    max_final_infected: float | None = Field(default=None, ge=0)
    intervention_budget: float | None = Field(default=None, ge=0)

class TransitionResult(BaseModel):
    previous_state: SIRState
    action: InterventionAction
    next_state: SIRState
    beta_used: float
    gamma_used: float
    duration: float

    @model_validator(mode="after")
    def check_time_increases(self):
        if self.next_state.t <= self.previous_state.t:
            raise ValueError("next_state.t must be greater than previous_state.t")
        return self
