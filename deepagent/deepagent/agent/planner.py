"""Planner agent — decomposes goals into executable sub-tasks."""

from __future__ import annotations

import json
from typing import Optional

from deepagent.config import DeepAgentConfig
from deepagent.llm.backend import LLMBackend
from deepagent.llm.prompts import PLANNER_SYSTEM
from deepagent.utils.logger import get_logger

log = get_logger(__name__)


class Planner:
    """Takes a high-level goal and produces a structured plan of sub-tasks."""

    def __init__(self, llm: LLMBackend, config: DeepAgentConfig):
        self.llm = llm
        self.config = config

    def create_plan(self, goal: str, context: str = "") -> dict:
        """Generate a structured plan for the given goal.

        Returns:
            dict with "plan_summary" and "sub_tasks" list.
        """
        messages = [
            {"role": "system", "content": PLANNER_SYSTEM},
        ]
        if context:
            messages.append({"role": "user", "content": f"Context:\n{context}"})
        messages.append({"role": "user", "content": f"Goal: {goal}"})

        raw = self.llm.generate(messages, temperature=0.4)

        plan = self._parse_plan(raw)
        log.info("Plan created: %s (%d sub-tasks)", plan.get("plan_summary", "?")[:80], len(plan.get("sub_tasks", [])))
        return plan

    def revise_plan(self, original_plan: dict, feedback: str) -> dict:
        """Revise a plan based on feedback from the Critic or failures."""
        messages = [
            {"role": "system", "content": PLANNER_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Original plan:\n{json.dumps(original_plan, indent=2)}\n\n"
                    f"Feedback / issues:\n{feedback}\n\n"
                    "Please create a REVISED plan that addresses the feedback."
                ),
            },
        ]

        raw = self.llm.generate(messages, temperature=0.4)
        return self._parse_plan(raw)

    @staticmethod
    def _parse_plan(raw: str) -> dict:
        """Extract JSON plan from the LLM output."""
        # Try to find JSON block
        import re

        json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
        text_to_parse = json_match.group(1) if json_match else raw

        try:
            plan = json.loads(text_to_parse.strip())
            if "sub_tasks" not in plan:
                plan = {"plan_summary": "Parsed plan", "sub_tasks": [plan] if isinstance(plan, dict) else []}
            return plan
        except json.JSONDecodeError:
            # Fallback: treat the whole output as a single-step plan
            log.warning("Could not parse plan JSON, creating single-step fallback.")
            return {
                "plan_summary": "Single-step execution",
                "sub_tasks": [
                    {
                        "id": 1,
                        "description": raw.strip()[:500],
                        "depends_on": [],
                        "tools_hint": [],
                        "estimated_complexity": "medium",
                    }
                ],
            }
