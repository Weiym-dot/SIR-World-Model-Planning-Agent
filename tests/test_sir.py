from src.schemas import InterventionAction
from src.world_model.sir import SIRWorldModel
from src.grounding.constraints import check_sir_constraints


def test_population_is_conserved():
    model = SIRWorldModel()
    state = model.init_state()
    result = model.step(state, InterventionAction.NONE, duration=5)
    report = check_sir_constraints(result.next_state, model.population)
    assert report.valid, report.errors


def test_strong_intervention_reduces_infections_relative_to_none():
    model = SIRWorldModel()
    state = model.init_state()
    none = model.step(state, InterventionAction.NONE, duration=7).next_state
    strong = model.step(state, InterventionAction.STRONG, duration=7).next_state
    assert strong.I < none.I
