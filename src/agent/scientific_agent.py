from __future__ import annotations

from dataclasses import dataclass, field

from src.agent.action_proposer import HeuristicActionProposer
from src.agent.goal_parser import GoalParser, GoalProfile
from src.agent.tools import simulate_intervention
from src.llm.openai_compatible import build_llm_client_from_env
from src.planning.greedy import Trajectory
from src.planning.mcts import MCTSConfig, MCTSPlanner
from src.planning.reward import PlanningObjective
from src.schemas import InterventionAction, SIRState
from src.world_model.base import WorldModel


@dataclass
class AgentStepLog:
    day: float
    state: SIRState
    proposal_ranking: list[str]
    proposal_priors: dict[str, float]
    proposal_rationale: str
    chosen_action: InterventionAction
    root_statistics: list[dict[str, float | int | str]]
    next_state: SIRState


@dataclass
class AgentRun:
    goal: GoalProfile
    trajectory: Trajectory
    logs: list[AgentStepLog] = field(default_factory=list)


class ScientificPlanningAgent:
    """LLM-at-goal-only + heuristic-prior + PUCT scientific agent.

    - GoalParser is the only component designed to optionally use an LLM.
    - HeuristicActionProposer deterministically produces P(a|s,g).
    - Expansion uses those priors to order unvisited actions.
    - Selection uses PUCT, so priors guide exploration but backed-up returns
      remain the source of Q values.
    - SIR dynamics and scientific verification stay deterministic.
    """

    def __init__(
        self,
        world_model: WorldModel,
        goal_parser: GoalParser | None = None,
        base_iterations: int = 100,
        puct_constant: float = 1.5,
        discount: float = 0.98,
        seed: int = 7,
    ):
        self.world_model = world_model
        if goal_parser is None:
            goal_parser = GoalParser(llm_client=build_llm_client_from_env())
        self.goal_parser = goal_parser
        self.proposer = HeuristicActionProposer()
        self.base_iterations = base_iterations
        self.puct_constant = puct_constant
        self.discount = discount
        self.seed = seed

    @staticmethod
    def _objective_from_goal(goal: GoalProfile) -> PlanningObjective:
        return PlanningObjective(
            infection_weight=goal.infection_weight,
            intervention_cost_weight=goal.intervention_cost_weight,
            target_peak_fraction=goal.target_peak_fraction,
        )

    def run(self, natural_language_goal: str, max_steps: int = 30) -> AgentRun:
        # Parse once. If a real StructuredLLMClient is configured, this is the
        # only place in the architecture where an LLM is involved.
        goal = self.goal_parser.parse(natural_language_goal)
        objective = self._objective_from_goal(goal)

        config = MCTSConfig(
            iterations=self.base_iterations,
            planning_horizon=goal.planning_horizon,
            puct_constant=self.puct_constant,
            discount=self.discount,
            seed=self.seed,
        )

        # Deterministic prior provider. It can be called at simulated tree
        # nodes cheaply because it does not invoke an LLM.
        def prior_provider(state: SIRState) -> dict[InterventionAction, float]:
            return self.proposer.propose(state, goal).priors

        planner = MCTSPlanner(
            self.world_model,
            config=config,
            objective=objective,
            prior_provider=prior_provider,
        )

        state = self.world_model.init_state()
        states = [state]
        actions: list[InterventionAction] = []
        rewards: list[float] = []
        logs: list[AgentStepLog] = []

        for _ in range(max_steps):
            if self.world_model.is_terminal(state):
                break

            proposal = self.proposer.propose(state, goal)
            chosen_action = planner.choose_action(state)
            root_stats = planner.root_statistics()

            tool_result = simulate_intervention(
                self.world_model,
                state,
                chosen_action,
                duration=config.step_days,
            )
            if not tool_result.scientifically_valid:
                raise RuntimeError(
                    f"Scientific constraint violation: {tool_result.errors}"
                )

            next_state = tool_result.transition.next_state
            _, reward = planner._transition(state, chosen_action)

            logs.append(
                AgentStepLog(
                    day=state.t,
                    state=state,
                    proposal_ranking=[a.value for a in proposal.ranked_actions],
                    proposal_priors={a.value: p for a, p in proposal.priors.items()},
                    proposal_rationale=proposal.rationale,
                    chosen_action=chosen_action,
                    root_statistics=root_stats,
                    next_state=next_state,
                )
            )
            actions.append(chosen_action)
            rewards.append(reward)
            states.append(next_state)
            state = next_state

        return AgentRun(
            goal=goal,
            trajectory=Trajectory(states=states, actions=actions, rewards=rewards),
            logs=logs,
        )
