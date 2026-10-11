from __future__ import annotations

from dataclasses import dataclass

from src.agent.goal_parser import GoalProfile
from src.schemas import InterventionAction, SIRState


@dataclass
class ActionProposal:
    """Deterministic policy prior used to guide MCTS search.

    `priors` is a probability distribution P(a | state, goal). The proposer
    does not choose the final action; MCTS still evaluates actions through the
    scientific world model and reward function.
    """

    priors: dict[InterventionAction, float]
    rationale: str

    @property
    def ranked_actions(self) -> list[InterventionAction]:
        return sorted(self.priors, key=self.priors.get, reverse=True)


class HeuristicActionProposer:
    """Transparent, deterministic action-prior policy.

    This component is intentionally *not* an LLM. It converts the current SIR
    state and already-parsed GoalProfile into priors over NONE/MILD/STRONG.
    The priors guide MCTS expansion and PUCT selection, but do not replace
    simulation-based planning.
    """

    def propose(self, state: SIRState, goal: GoalProfile) -> ActionProposal:
        infected_fraction = state.I / state.population

        if infected_fraction >= 0.08 or goal.infection_weight >= 1.4:
            priors = {
                InterventionAction.STRONG: 0.60,
                InterventionAction.MILD: 0.30,
                InterventionAction.NONE: 0.10,
            }
            reason = "infection burden or goal priority favors stronger control"
        elif goal.intervention_cost_weight >= 0.03 and infected_fraction < 0.04:
            priors = {
                InterventionAction.NONE: 0.60,
                InterventionAction.MILD: 0.30,
                InterventionAction.STRONG: 0.10,
            }
            reason = "low infection burden and high cost sensitivity favor restraint"
        else:
            priors = {
                InterventionAction.MILD: 0.50,
                InterventionAction.NONE: 0.30,
                InterventionAction.STRONG: 0.20,
            }
            reason = "balanced objective favors a moderate intervention prior"

        return ActionProposal(priors=priors, rationale=reason)
