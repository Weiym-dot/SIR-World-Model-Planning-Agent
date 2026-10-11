from __future__ import annotations
from abc import ABC, abstractmethod
from src.schemas import SIRState, InterventionAction, TransitionResult

class WorldModel(ABC):
    @abstractmethod
    def init_state(self) -> SIRState:
        raise NotImplementedError

    @abstractmethod
    def step(self, state: SIRState, action: InterventionAction, duration: float = 1.0) -> TransitionResult:
        raise NotImplementedError

    @abstractmethod
    def is_terminal(self, state: SIRState) -> bool:
        raise NotImplementedError
