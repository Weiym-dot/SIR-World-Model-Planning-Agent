from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable

from src.grounding.verifier import verify_transition
from src.planning.greedy import Trajectory
from src.planning.reward import PlanningObjective, objective_utility
from src.schemas import InterventionAction, SIRState
from src.world_model.base import WorldModel

ActionPriorProvider = Callable[[SIRState], dict[InterventionAction, float]]


@dataclass
class MCTSNode:
    """One state in the search tree.

    `action_from_parent` is the action that produced this node.
    `prior_from_parent` is P(action | parent_state, goal), supplied by the
    deterministic action proposer. It guides search but is never treated as a
    reward or a world-model prediction.
    """

    state: SIRState
    parent: "MCTSNode | None" = None
    action_from_parent: InterventionAction | None = None
    reward_from_parent: float = 0.0
    prior_from_parent: float = 0.0
    depth: int = 0
    visits: int = 0
    value_sum: float = 0.0
    children: list["MCTSNode"] = field(default_factory=list)
    untried_actions: list[InterventionAction] = field(
        default_factory=lambda: list(InterventionAction)
    )
    action_priors: dict[InterventionAction, float] | None = None

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0

    @property
    def fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

    def puct_score(self, puct_constant: float) -> float:
        """PUCT = value estimate + prior-guided exploration bonus.

        Q(s,a) is the child's mean backed-up return. P(s,a) is the action
        prior supplied when the child was expanded. N(s) and N(s,a) are visit
        counts. Unlike plain UCT, exploration is weighted by the prior.
        """
        if self.parent is None:
            return self.mean_value

        parent_visits = max(1, self.parent.visits)
        exploration = (
            puct_constant
            * self.prior_from_parent
            * math.sqrt(parent_visits)
            / (1 + self.visits)
        )
        return self.mean_value + exploration

    def best_puct_child(self, puct_constant: float) -> "MCTSNode":
        if not self.children:
            raise ValueError("Cannot select a child from a leaf node")
        return max(
            self.children,
            key=lambda child: child.puct_score(puct_constant),
        )


@dataclass
class MCTSConfig:
    iterations: int = 100
    planning_horizon: int = 7
    puct_constant: float = 1.5
    discount: float = 0.98
    step_days: float = 1.0
    seed: int = 7


class MCTSPlanner:
    """Receding-horizon MCTS with heuristic priors and PUCT selection.

    The planner still derives values from scientific simulation + numeric
    reward. Priors only guide *where to spend search budget*:
      - Expansion: try higher-prior unvisited actions first.
      - Selection: PUCT combines backed-up Q values with prior-guided
        exploration.

    Rollouts remain random so that the heuristic prior does not dominate every
    stage of search.
    """

    def __init__(
        self,
        world_model: WorldModel,
        config: MCTSConfig | None = None,
        objective: PlanningObjective | None = None,
        prior_provider: ActionPriorProvider | None = None,
    ):
        self.world_model = world_model
        self.config = config or MCTSConfig()
        self.objective = objective or PlanningObjective()
        self.prior_provider = prior_provider
        self.rng = random.Random(self.config.seed)
        self.last_root: MCTSNode | None = None

    def _transition(self, state: SIRState, action: InterventionAction):
        result = self.world_model.step(
            state,
            action,
            duration=self.config.step_days,
        )
        report = verify_transition(result, expected_population=state.population)
        if not report.valid:
            raise RuntimeError(f"Invalid world-model transition: {report.errors}")
        utility = objective_utility(
            result.next_state,
            action,
            population=state.population,
            objective=self.objective,
        )
        return result.next_state, utility

    def _ensure_priors(self, node: MCTSNode) -> None:
        """Compute and cache P(a|s,g) once per tree node."""
        if node.action_priors is not None:
            return

        if self.prior_provider is None:
            # Uniform prior reproduces an unguided search baseline.
            uniform = 1.0 / len(list(InterventionAction))
            node.action_priors = {action: uniform for action in InterventionAction}
            return

        raw = self.prior_provider(node.state)
        if set(raw) != set(InterventionAction):
            raise ValueError("prior_provider must return a prior for every action")
        if any(value < 0 for value in raw.values()):
            raise ValueError("action priors must be nonnegative")

        total = sum(raw.values())
        if total <= 0:
            raise ValueError("action priors must have positive total mass")

        # Normalize defensively so callers do not need perfect floating sums.
        node.action_priors = {action: raw[action] / total for action in InterventionAction}

    def _select(self, root: MCTSNode) -> tuple[MCTSNode, float]:
        """Follow PUCT while the current node is fully expanded."""
        node = root
        total_return = 0.0
        discount_power = 0

        while (
            node.fully_expanded
            and node.children
            and node.depth < self.config.planning_horizon
            and not self.world_model.is_terminal(node.state)
        ):
            node = node.best_puct_child(self.config.puct_constant)
            total_return += (self.config.discount ** discount_power) * node.reward_from_parent
            discount_power += 1

        return node, total_return

    def _expand(self, node: MCTSNode) -> tuple[MCTSNode, float]:
        """Expand the highest-prior action that has not yet been tried."""
        if (
            node.depth >= self.config.planning_horizon
            or self.world_model.is_terminal(node.state)
            or not node.untried_actions
        ):
            return node, 0.0

        self._ensure_priors(node)
        assert node.action_priors is not None

        action = max(
            node.untried_actions,
            key=lambda candidate: node.action_priors[candidate],
        )
        node.untried_actions.remove(action)

        next_state, reward = self._transition(node.state, action)
        child = MCTSNode(
            state=next_state,
            parent=node,
            action_from_parent=action,
            reward_from_parent=reward,
            prior_from_parent=node.action_priors[action],
            depth=node.depth + 1,
        )
        node.children.append(child)
        return child, reward

    def _rollout(self, node: MCTSNode) -> float:
        """Simulate to the planning horizon with a random rollout policy."""
        state = node.state
        depth = node.depth
        total_return = 0.0
        local_step = 0

        while (
            depth < self.config.planning_horizon
            and not self.world_model.is_terminal(state)
        ):
            action = self.rng.choice(list(InterventionAction))
            state, reward = self._transition(state, action)
            total_return += (self.config.discount ** local_step) * reward
            local_step += 1
            depth += 1

        return total_return

    @staticmethod
    def _backpropagate(node: MCTSNode, total_return: float) -> None:
        """Update only the nodes on the sampled path from leaf back to root."""
        while node is not None:
            node.visits += 1
            node.value_sum += total_return
            node = node.parent

    def run_search(self, state: SIRState) -> MCTSNode:
        root = MCTSNode(state=state)

        for _ in range(self.config.iterations):
            selected, path_return = self._select(root)

            expanded, expansion_reward = self._expand(selected)
            if expanded is not selected:
                path_return += (
                    self.config.discount ** (expanded.depth - 1)
                ) * expansion_reward

            rollout_return = self._rollout(expanded)
            rollout_return *= self.config.discount ** expanded.depth

            total_return = path_return + rollout_return
            self._backpropagate(expanded, total_return)

        self.last_root = root
        return root

    def choose_action(self, state: SIRState) -> InterventionAction:
        root = self.run_search(state)
        if not root.children:
            raise RuntimeError("MCTS produced no child actions")

        # Robust-child rule: the action receiving the most search visits is the
        # real action. Q is only a tie-breaker; priors guide search, not the
        # final decision directly.
        best_child = max(
            root.children,
            key=lambda child: (child.visits, child.mean_value),
        )
        assert best_child.action_from_parent is not None
        return best_child.action_from_parent

    def root_statistics(self) -> list[dict[str, float | int | str]]:
        if self.last_root is None:
            return []
        rows = []
        for child in sorted(
            self.last_root.children,
            key=lambda c: c.action_from_parent.value if c.action_from_parent else "",
        ):
            rows.append(
                {
                    "action": child.action_from_parent.value,
                    "prior": child.prior_from_parent,
                    "visits": child.visits,
                    "mean_value": child.mean_value,
                    "puct": child.puct_score(self.config.puct_constant),
                }
            )
        return rows


def mcts_plan(
    world_model: WorldModel,
    config: MCTSConfig | None = None,
    max_steps: int = 60,
    objective: PlanningObjective | None = None,
    prior_provider: ActionPriorProvider | None = None,
) -> Trajectory:
    """Execute receding-horizon prior-guided MCTS over a full trajectory."""
    planner = MCTSPlanner(
        world_model,
        config=config,
        objective=objective,
        prior_provider=prior_provider,
    )
    state = world_model.init_state()
    states = [state]
    actions: list[InterventionAction] = []
    rewards: list[float] = []

    for _ in range(max_steps):
        if world_model.is_terminal(state):
            break

        action = planner.choose_action(state)
        next_state, reward = planner._transition(state, action)
        states.append(next_state)
        actions.append(action)
        rewards.append(reward)
        state = next_state

    return Trajectory(states=states, actions=actions, rewards=rewards)
