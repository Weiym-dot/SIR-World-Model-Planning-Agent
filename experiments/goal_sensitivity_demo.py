from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

from src.agent.scientific_agent import ScientificPlanningAgent
from src.evaluation.metrics import compute_metrics
from src.viz.trajectory import plot_trajectory
from src.world_model.sir import SIRWorldModel

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)

GOALS = {
    "aggressive": "Reduce infections aggressively over the next 10 days.",
    "balanced": "Balance infection control with intervention cost over the next 10 days.",
    "cost_sensitive": "Keep interventions light and minimize intervention cost over the next 10 days.",
}


def main() -> None:
    model = SIRWorldModel(horizon_days=30)
    rows: list[dict] = []

    for name, natural_goal in GOALS.items():
        agent = ScientificPlanningAgent(model, base_iterations=70, seed=7)
        run = agent.run(natural_goal, max_steps=30)
        metrics = compute_metrics(run.trajectory)

        print(f"\n[{name}] {natural_goal}")
        print("parsed goal:")
        print(json.dumps(run.goal.model_dump(), indent=2))
        print("metrics:")
        print(json.dumps(metrics.to_dict(), indent=2))
        print("first 10 actions:", [a.value for a in run.trajectory.actions[:10]])

        plot_trajectory(
            run.trajectory.states,
            f"Goal-conditioned MCTS: {name}",
            str(OUT / f"trajectory_{name}.png"),
        )

        row = {"goal_name": name, "natural_language_goal": natural_goal}
        row.update(run.goal.model_dump())
        row.update(metrics.to_dict())
        rows.append(row)

    csv_path = OUT / "goal_sensitivity_metrics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Simple research-ready comparison: action mix under different goals.
    labels = [r["goal_name"] for r in rows]
    none = [r["none_days"] for r in rows]
    mild = [r["mild_days"] for r in rows]
    strong = [r["strong_days"] for r in rows]

    x = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, none, label="none")
    ax.bar(x, mild, bottom=none, label="mild")
    bottoms = [a + b for a, b in zip(none, mild)]
    ax.bar(x, strong, bottom=bottoms, label="strong")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Days")
    ax.set_title("Intervention mix changes with natural-language goal")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "goal_action_mix.png", dpi=180)
    plt.close(fig)

    print(f"\nSaved outputs to {OUT}")


if __name__ == "__main__":
    main()
