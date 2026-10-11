# Architecture

```text
Natural-language goal
        |
        v
Goal Parser (LLM interface or deterministic fallback)
        |
        v
Structured GoalProfile
(infection weight, intervention cost weight, horizon)
        |
        v
Scientific Planning Agent
        |
        +--> high-level action proposal / logging
        |
        v
Heuristic Action Prior --> MCTS Planner -- PUCT --> Selection / Prior-guided Expansion / Rollout / Backpropagation
        |
        v
SIR World Model: (state, action) -> next state
        |
        v
Scientific Constraint Verifier
(population conservation, non-negativity)
        |
        v
Execute ONE action, observe new state, then re-plan
```

## Research connection

- **LAW:** separates agent preferences/intent from world dynamics.
- **RAP:** uses a world model plus reward-guided tree search for planning.
- **PAN:** motivates interactable world simulation: actions condition predicted future states.
- **Scientific grounding:** the simulator and verifier prevent the language layer from inventing physical dynamics.

The project is intentionally hybrid: the language component interprets goals, while the scientific simulator remains the source of truth for state transitions.
