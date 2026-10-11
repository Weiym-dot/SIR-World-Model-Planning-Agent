from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel, Field


class GoalProfile(BaseModel):
    """Structured objective used by the planner.

    The LLM (or deterministic fallback) maps a natural-language scientific goal
    into these parameters. The world model itself remains independent of this
    preference representation.
    """

    original_goal: str
    infection_weight: float = Field(default=1.0, gt=0)
    intervention_cost_weight: float = Field(default=0.015, ge=0)
    planning_horizon: int = Field(default=7, ge=2, le=30)
    target_peak_fraction: float | None = Field(default=None, gt=0, lt=1)
    rationale: str = ""


class StructuredLLMClient(Protocol):
    """Minimal provider-agnostic interface for structured LLM output."""

    def complete_json(self, prompt: str) -> dict:
        ...


def build_goal_parser_prompt(goal: str) -> str:
    """Prompt for a real LLM client.

    Kept provider-agnostic so the research code is not tied to one API.
    """
    return f"""You are converting a scientific planning objective into a compact control profile.

The environment is an SIR epidemic simulator. Available interventions are none, mild, and strong.
Return JSON only with these keys:
- infection_weight: positive float
- intervention_cost_weight: nonnegative float
- planning_horizon: integer from 2 to 30
- target_peak_fraction: float in (0,1) or null
- rationale: one short sentence

Interpretation:
- Larger infection_weight prioritizes suppressing infections.
- Larger intervention_cost_weight avoids costly interventions.
- Longer planning_horizon values favor longer-term planning.

User goal: {goal!r}
"""


def _number_before_percent(text: str) -> float | None:
    match = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
    if not match:
        return None
    value = float(match.group(1)) / 100.0
    return value if 0 < value < 1 else None


def parse_goal_rule_based(goal: str) -> GoalProfile:
    """Deterministic fallback used for tests and no-API demos.

    This is deliberately transparent: it lets the full agent run locally while
    preserving the same structured interface a real LLM would fill.
    """
    text = goal.lower().strip()

    infection_weight = 1.0
    intervention_cost_weight = 0.015
    planning_horizon = 7
    rationale_bits: list[str] = []

    aggressive_terms = (
        "aggressive", "aggressively", "minimize infections", "suppress infections",
        "prioritize infection", "keep infections low", "reduce infections",
    )
    economy_terms = (
        "minimize intervention", "avoid intervention", "low intervention",
        "economic", "cost", "light intervention", "least intervention",
    )
    long_terms = ("long term", "long-term", "long horizon", "over the next month")

    if any(term in text for term in aggressive_terms):
        infection_weight = 1.6
        intervention_cost_weight = 0.006
        rationale_bits.append("prioritize infection suppression")

    is_balanced = "balance" in text or "balanced" in text or "trade-off" in text or "tradeoff" in text

    if any(term in text for term in economy_terms) and not is_balanced:
        infection_weight = 0.8
        intervention_cost_weight = 0.035
        rationale_bits.append("penalize intervention cost more strongly")

    if is_balanced:
        infection_weight = 1.0
        intervention_cost_weight = 0.015
        rationale_bits.append("balance health burden and intervention cost")

    if any(term in text for term in long_terms):
        planning_horizon = 14
        rationale_bits.append("use a longer planning horizon")

    day_match = re.search(r"(?:next|over)\s+(\d+)\s+days", text)
    if day_match:
        planning_horizon = min(30, max(2, int(day_match.group(1))))
        rationale_bits.append(f"match the stated {planning_horizon}-day horizon")

    target_peak_fraction = _number_before_percent(text)
    if target_peak_fraction is not None:
        rationale_bits.append("track the stated infection-percentage target")

    return GoalProfile(
        original_goal=goal,
        infection_weight=infection_weight,
        intervention_cost_weight=intervention_cost_weight,
        planning_horizon=planning_horizon,
        target_peak_fraction=target_peak_fraction,
        rationale="; ".join(rationale_bits) or "use the default balanced planning objective",
    )


@dataclass
class GoalParser:
    """Natural-language goal parser with an optional real LLM backend."""

    llm_client: StructuredLLMClient | None = None

    def parse(self, goal: str) -> GoalProfile:
        if not goal.strip():
            raise ValueError("goal must not be empty")

        if self.llm_client is None:
            return parse_goal_rule_based(goal)

        raw = self.llm_client.complete_json(build_goal_parser_prompt(goal))
        # Validate model output with Pydantic instead of trusting arbitrary JSON.
        return GoalProfile(original_goal=goal, **raw)


class MockStructuredLLM:
    """Tiny test double demonstrating the structured-output interface."""

    def __init__(self, response: dict):
        self.response = response
        self.last_prompt: str | None = None

    def complete_json(self, prompt: str) -> dict:
        self.last_prompt = prompt
        # round-trip keeps the mock JSON-like and avoids shared mutable objects
        return json.loads(json.dumps(self.response))
