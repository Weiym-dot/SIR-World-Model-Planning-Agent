from __future__ import annotations
import matplotlib.pyplot as plt
from src.schemas import SIRState


def plot_trajectory(states: list[SIRState], title: str, output_path: str | None = None):
    t = [s.t for s in states]
    S = [s.S for s in states]
    I = [s.I for s in states]
    R = [s.R for s in states]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t, S, label="Susceptible")
    ax.plot(t, I, label="Infected")
    ax.plot(t, R, label="Recovered")
    ax.set_xlabel("Day")
    ax.set_ylabel("Population")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=160)
    return fig, ax
