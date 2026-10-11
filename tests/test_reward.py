from src.schemas import SIRState, InterventionAction
from src.planning.reward import step_reward


def test_lower_infection_is_better_if_action_cost_same():
    a = SIRState(t=1, S=900, I=80, R=20)
    b = SIRState(t=1, S=920, I=60, R=20)
    assert step_reward(b, InterventionAction.NONE) > step_reward(a, InterventionAction.NONE)


def test_strong_action_has_larger_cost():
    state = SIRState(t=1, S=900, I=80, R=20)
    assert step_reward(state, InterventionAction.NONE) > step_reward(state, InterventionAction.STRONG)
