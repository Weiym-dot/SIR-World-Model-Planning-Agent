from src.planning.mcts import MCTSConfig, MCTSNode, MCTSPlanner
from src.schemas import SIRState, InterventionAction
from src.world_model.sir import SIRWorldModel


def test_puct_prefers_higher_prior_when_values_and_visits_match():
    root = MCTSNode(state=SIRState(t=0, S=9900, I=100, R=0), visits=20)
    child_a = MCTSNode(
        state=root.state,
        parent=root,
        action_from_parent=InterventionAction.NONE,
        prior_from_parent=0.1,
        visits=5,
        value_sum=2.5,
    )
    child_b = MCTSNode(
        state=root.state,
        parent=root,
        action_from_parent=InterventionAction.MILD,
        prior_from_parent=0.7,
        visits=5,
        value_sum=2.5,
    )
    root.children = [child_a, child_b]

    assert child_b.puct_score(1.5) > child_a.puct_score(1.5)


def test_expansion_uses_highest_prior_untried_action():
    model = SIRWorldModel(horizon_days=10)

    def prior_provider(_state):
        return {
            InterventionAction.NONE: 0.1,
            InterventionAction.MILD: 0.2,
            InterventionAction.STRONG: 0.7,
        }

    planner = MCTSPlanner(model, prior_provider=prior_provider)
    root = MCTSNode(state=model.init_state())
    child, _ = planner._expand(root)

    assert child.action_from_parent == InterventionAction.STRONG
    assert child.prior_from_parent == 0.7


def test_backpropagation_updates_only_ancestor_path():
    state = SIRState(t=0, S=9900, I=100, R=0)
    root = MCTSNode(state=state)
    left = MCTSNode(state=state, parent=root)
    right = MCTSNode(state=state, parent=root)
    leaf = MCTSNode(state=state, parent=left)
    root.children = [left, right]
    left.children = [leaf]

    MCTSPlanner._backpropagate(leaf, total_return=2.5)

    assert leaf.visits == 1 and leaf.value_sum == 2.5
    assert left.visits == 1 and left.value_sum == 2.5
    assert root.visits == 1 and root.value_sum == 2.5
    assert right.visits == 0 and right.value_sum == 0.0


def test_mcts_returns_valid_action_and_uses_search_budget():
    model = SIRWorldModel(horizon_days=15)
    config = MCTSConfig(iterations=30, planning_horizon=5, seed=1)
    planner = MCTSPlanner(model, config=config)

    action = planner.choose_action(model.init_state())

    assert action in list(InterventionAction)
    assert planner.last_root is not None
    assert planner.last_root.visits == config.iterations
    assert len(planner.last_root.children) == len(list(InterventionAction))
