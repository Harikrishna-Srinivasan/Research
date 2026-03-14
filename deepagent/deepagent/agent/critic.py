"""Critic agent — reviews work and detects breakthroughs."""

from __future__ import annotations

import json
import re
from typing import Optional

from deepagent.config import DeepAgentConfig
from deepagent.llm.backend import LLMBackend
from deepagent.llm.prompts import CRITIC_SYSTEM
from deepagent.utils import alerts
from deepagent.utils.logger import get_logger

log = get_logger(__name__)


class Critic:
    """Reviews completed work for quality and potential breakthroughs."""

    def __init__(self, llm: LLMBackend, config: DeepAgentConfig):
        self.llm = llm
        self.config = config

    def review(self, task_description: str, result: str, context: str = "") -> dict:
        """Review work and return a structured assessment.

        Returns dict with:
            quality_score (1-10), is_breakthrough (bool),
            breakthrough_description (str|None), issues (list),
            suggestions (list), verdict (PASS|FAIL|NEEDS_REVISION)
        """
        messages = [
            {"role": "system", "content": CRITIC_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Task: {task_description}\n\n"
                    f"Result:\n{result[:3000]}\n\n"
                    f"Context: {context[:1000] if context else 'None'}\n\n"
                    "Provide your review in the specified JSON format."
                ),
            },
        ]

        raw = self.llm.generate(messages, temperature=0.3)
        review = self._parse_review(raw)

        # Handle breakthrough alert
        if review.get("is_breakthrough") and self.config.agent.enable_breakthrough_detection:
            desc = review.get("breakthrough_description", "Potential breakthrough detected!")
            alerts.breakthrough(f"Task: {task_description}\n\n{desc}")
            log.info("⚡ BREAKTHROUGH: %s", desc[:200])

        return review

    @staticmethod
    def _parse_review(raw: str) -> dict:
        """Parse the critic's JSON review."""
        json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
        text = json_match.group(1) if json_match else raw

        try:
            review = json.loads(text.strip())
            return review
        except json.JSONDecodeError:
            log.warning("Could not parse critic review, using defaults.")
            return {
                "quality_score": 5,
                "is_breakthrough": False,
                "breakthrough_description": None,
                "issues": ["Could not parse critic output"],
                "suggestions": [],
                "verdict": "NEEDS_REVISION",
            }
