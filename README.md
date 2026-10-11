# SIR World Model Planning Agent

A compact research prototype for **goal-conditioned, world-model-based planning**. The system translates a natural-language scientific objective into a structured planning objective, uses a transparent heuristic prior to guide **PUCT-MCTS**, simulates candidate futures with an explicit **SIR scientific world model**, and verifies scientific constraints before executing each intervention.

This project is inspired by the separation between **language understanding, agent decision-making, world models, and planning** discussed in work such as LAW, RAP, and PAN. It is a simplified scientific planning prototype, not a reproduction of those systems.

## Architecture

```text
Natural-language scientific goal
        ↓
LLM-capable Goal Parser
        ↓
Structured GoalProfile
        ↓
PlanningObjective / numeric reward
        ↓
HeuristicActionProposer
        ↓
P(a | state, goal) action priors
        ↓
PUCT-MCTS
  ├─ Expansion: highest-prior untried action first
  ├─ Selection: Q + prior-guided exploration
  ├─ Rollout: random simulation policy
  └─ Backpropagation: update visits/value on sampled path
        ↓
SIR World Model
        ↓
Scientific Constraint Verifier
        ↓
Execute one action → observe → re-plan
```

The key design choice is that the **LLM does not implement the world model**. Since SIR is a standard model with explicit state-transition equations, the simulator can currently serve as a reliable source of truth without an LLM. Language is only used to interpret high-level goals, while MCTS plans over the resulting dynamics and reward. In future extensions, if the state space or real-world dynamics become more complex and difficult to model explicitly, an LLM-based world model could be introduced.

## Core ideas

### 1. Goal-conditioned planning

A natural-language goal such as:

```text
Balance infection control with intervention cost over the next 10 days.
```

is converted into a structured `GoalProfile` containing:

- infection weight
- intervention-cost weight
- planning horizon
- optional peak-infection target

These parameters define the planner's numeric objective without changing the underlying world dynamics.

### 2. Heuristic action priors

`HeuristicActionProposer` produces a deterministic prior distribution:

```text
P(action | state, goal)
```

For example:

```text
NONE   0.10
MILD   0.30
STRONG 0.60
```

The prior does **not** directly choose the final intervention. It only guides where MCTS spends search effort.

### 3. Prior-guided MCTS with PUCT

Selection uses:

```text
PUCT(s,a) = Q(s,a)
          + c_puct · P(s,a) · sqrt(N(s)) / (1 + N(s,a))
```

where:

- `Q(s,a)` is the backed-up mean return from simulations
- `P(s,a)` is the heuristic action prior
- `N(s)` is the parent visit count
- `N(s,a)` is the child visit count

Expansion tries the highest-prior unvisited action first. Rollout remains random so that the heuristic prior guides search without controlling every simulated decision.

### 4. Explicit scientific world model

The SIR simulator implements:

```text
(state, action) → next_state
```

Available actions are `none`, `mild`, and `strong`, which modify transmission intensity. MCTS queries this simulator repeatedly to imagine possible futures before executing a real intervention.

### 5. Scientific grounding

Every transition is checked for:

- non-negative S/I/R populations
- conservation of total population

The planner therefore cannot accept a simulated future that violates these explicit scientific constraints.

### 6. Receding-horizon agent loop

The agent does not commit to one fixed long-horizon plan. It:

```text
observe → plan → execute one action → observe new state → re-plan
```

This makes the system responsive to the state reached after each intervention.

## Repository structure

```text
.
├── src/
│   ├── agent/
│   │   ├── goal_parser.py
│   │   ├── action_proposer.py
│   │   ├── scientific_agent.py
│   │   └── tools.py
│   ├── llm/
│   │   └── openai_compatible.py
│   ├── planning/
│   │   ├── mcts.py
│   │   ├── reward.py
│   │   └── greedy.py
│   ├── world_model/
│   │   ├── base.py
│   │   └── sir.py
│   ├── grounding/
│   │   ├── constraints.py
│   │   └── verifier.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   └── compare_planners.py
│   └── viz/
│       └── trajectory.py
├── experiments/
│   └── goal_sensitivity_demo.py
├── results/
│   ├── goal_sensitivity_metrics.csv
│   ├── goal_action_mix.png
│   └── trajectory_*.png
├── docs/
│   ├── architecture.md
│   └── project_report.md
├── tests/
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```



## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
python -m experiments.goal_sensitivity_demo
```

The experiment writes plots and metrics to `results/`.

## Optional LLM goal parsing

The project runs without an API key by default. With `LLM_ENABLED=false`, a deterministic rule-based parser produces the same `GoalProfile` interface used by the LLM path.

To enable a real model:

```bash
cp .env.example .env
```

Then configure any provider exposing an OpenAI-compatible `chat/completions` endpoint:

```dotenv
LLM_ENABLED=true
LLM_BASE_URL=https://your-provider.example/v1
LLM_API_KEY=your_key
LLM_MODEL=your_model
LLM_TIMEOUT_SECONDS=45
LLM_TEMPERATURE=0
```

For example, an OpenRouter configuration can use:

```dotenv
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=openrouter/free
```

Provider-specific networking code is isolated in `src/llm/`. The rest of the research code depends only on a minimal interface:

```python
class StructuredLLMClient(Protocol):
    def complete_json(self, prompt: str) -> dict:
        ...
```

If a future provider does not expose an OpenAI-compatible endpoint, a new adapter can implement this interface without changing the agent, planner, reward, or world model.

## Experiment

`experiments/goal_sensitivity_demo.py` evaluates three high-level objectives:

- aggressive infection suppression
- balanced infection control and intervention cost
- cost-sensitive intervention

For each goal, the experiment records:

- parsed planning objective
- peak infected population/fraction
- final infected population
- cumulative infection burden
- intervention cost
- number of `none`, `mild`, and `strong` intervention days
- scientific constraint violations

The purpose is to test the full causal chain:

```text
language goal
→ structured objective
→ action priors + reward
→ PUCT search behavior
→ intervention policy
→ simulated scientific trajectory
```



## Testing

The test suite covers:

- SIR dynamics and population conservation
- reward behavior
- goal parsing and validation
- provider-agnostic LLM adapter behavior
- PUCT preference under matched value/visit conditions
- prior-guided expansion ordering
- MCTS backpropagation behavior
- end-to-end scientific agent execution

Run:

```bash
pytest -q
```



## Design boundaries

The components intentionally have different responsibilities:


| Component       | Responsibility                                    |
| --------------- | ------------------------------------------------- |
| Goal Parser     | Translate language into a structured objective    |
| Reward          | Define numerically what outcomes are desirable    |
| Heuristic Prior | Guide search toward promising actions             |
| PUCT-MCTS       | Allocate search effort and choose an intervention |
| World Model     | Predict scientific state transitions              |
| Verifier        | Reject scientifically invalid transitions         |


In particular, **prior ≠ reward ≠ value**. Priors guide exploration, rewards score simulated consequences, and backed-up mean values summarize observed long-term returns.

## Limitations

- The SIR model is deliberately simple and deterministic.
- The action prior is hand-designed rather than learned.
- Rollout uses a random policy.
- The reward is hand-designed rather than learned from preferences or data.
- The scientific world model is explicit rather than learned from multimodal observations.
- Natural-language goal parsing can introduce ambiguity when a real LLM backend is enabled.



## Possible extensions

- compare uniform-prior UCT with heuristic-prior PUCT
- study sensitivity to prior quality and `c_puct`
- replace random rollout with a heuristic or learned rollout policy
- introduce stochastic/uncertain epidemiological dynamics
- add model uncertainty to planning
- extend the environment beyond SIR
- learn action priors from data while keeping the scientific simulator grounded



## Research inspiration

- **LAW:** *Language Models, Agent Models, and World Models: The LAW for Machine Reasoning and Planning*
- **RAP:** *Reasoning with Language Model is Planning with World Model*
- **PAN:** *A World Model for General, Interactable, and Long-Horizon World Simulation*

These works motivate the conceptual separation used here; this repository is a small independent prototype rather than an implementation of those papers.