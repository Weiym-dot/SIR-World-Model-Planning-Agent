from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp
from src.schemas import SIRState, InterventionAction, TransitionResult
from src.world_model.base import WorldModel

class SIRWorldModel(WorldModel):
    """Simple scientific world model based on the SIR epidemic equations.

    State: (S, I, R)
    Action: intervention strength, implemented as a multiplier on beta.
    Transition: numerically integrate the SIR ODE over `duration` days.
    """

    ACTION_BETA_MULTIPLIER = {
        InterventionAction.NONE: 1.0,
        InterventionAction.MILD: 0.65,
        InterventionAction.STRONG: 0.35,
    }

    def __init__(
        self,
        population: float = 10_000,
        initial_infected: float = 100,
        initial_recovered: float = 0,
        beta: float = 0.30,
        gamma: float = 0.10,
        horizon_days: float = 60,
    ):
        self.population = float(population)
        self.initial_infected = float(initial_infected)
        self.initial_recovered = float(initial_recovered)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.horizon_days = float(horizon_days)

    def init_state(self) -> SIRState:
        susceptible = self.population - self.initial_infected - self.initial_recovered
        return SIRState(t=0.0, S=susceptible, I=self.initial_infected, R=self.initial_recovered)

    @staticmethod
    def _sir_rhs(_t: float, y: np.ndarray, beta: float, gamma: float, population: float) -> list[float]:
        S, I, R = y
        dS = -beta * S * I / population
        dI = beta * S * I / population - gamma * I
        dR = gamma * I
        return [dS, dI, dR]

    def step(self, state: SIRState, action: InterventionAction, duration: float = 1.0) -> TransitionResult:
        if duration <= 0:
            raise ValueError("duration must be positive")

        beta_used = self.beta * self.ACTION_BETA_MULTIPLIER[action]
        sol = solve_ivp(
            fun=lambda t, y: self._sir_rhs(t, y, beta_used, self.gamma, self.population),
            t_span=(state.t, state.t + duration),
            y0=[state.S, state.I, state.R],
            t_eval=[state.t + duration],
            rtol=1e-8,
            atol=1e-10,
        )
        if not sol.success:
            raise RuntimeError(sol.message)

        S, I, R = (max(0.0, float(x)) for x in sol.y[:, -1])
        next_state = SIRState(t=state.t + duration, S=S, I=I, R=R)
        return TransitionResult(
            previous_state=state,
            action=action,
            next_state=next_state,
            beta_used=beta_used,
            gamma_used=self.gamma,
            duration=duration,
        )

    def is_terminal(self, state: SIRState) -> bool:
        return state.t >= self.horizon_days or state.I < 1.0
