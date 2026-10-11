# Physics-Grounded World-Model Planning Agent

## 1. Research question
Can a hybrid agent translate a natural-language scientific objective into a structured planning objective, then use an explicit world model and MCTS to choose long-horizon interventions while preserving scientific constraints?

## 2. System
The environment is a deterministic SIR epidemic simulator. The agent receives a natural-language goal, converts it to planning weights and a horizon, and uses receding-horizon MCTS to choose among `none`, `mild`, and `strong` interventions. Every proposed real transition is checked for non-negativity and population conservation.

## 3. Why this is a world-model project
The SIR simulator implements the transition function `(state, action) -> next_state`. MCTS never invents epidemic dynamics; it queries this world model to imagine possible futures. Reward is kept separate from dynamics, so changing the goal changes preferences without changing the simulated world.

## 4. Experiments
Run:

```bash
python -m experiments.goal_sensitivity_demo
```

The experiment compares three natural-language objectives: aggressive infection suppression, balanced control, and cost-sensitive intervention. It saves trajectory plots, a stacked action-mix plot, and a CSV of metrics.

## 5. Metrics
- peak infected population / fraction
- final infected population
- cumulative infected fraction
- intervention cost
- number of none/mild/strong days
- scientific constraint violations

## 6. Expected finding
Different natural-language goals should produce different structured reward weights and therefore different intervention policies. This demonstrates a clean separation between language-level intent, planning, and world dynamics.

## 7. Limitations
- SIR is a simplified deterministic model.
- The default goal parser is rule-based so the repo works without an API key; a real structured-output LLM can replace it through the same interface.
- Rollout policy is random rather than learned.
- MCTS uses hand-designed utility rather than a learned reward model.
- The world model is explicit rather than learned from data/video as in PAN.

## 8. Extensions
- replace rule-based goal parsing with a real LLM structured-output client
- use learned or uncertain epidemiological dynamics
- compare random vs heuristic vs learned rollout policies
- add uncertainty-aware planning
- move from SIR to a richer scientific simulator
