"""Executor agent — runs sub-tasks using the ReAct loop."""

from __future__ import annotations

from deepagent.agent.core import AgentCore
from deepagent.config import DeepAgentConfig
from deepagent.llm.backend import LLMBackend
from deepagent.llm.prompts import EXECUTOR_SYSTEM
from deepagent.memory.long_term import LongTermMemory
from deepagent.tools.registry import ToolRegistry
from deepagent.utils.logger import get_logger

log = get_logger(__name__)


class Executor:
    """Executes individual sub-tasks via the ReAct agent core."""

    def __init__(
        self,
        llm: LLMBackend,
        tools: ToolRegistry,
        config: DeepAgentConfig,
        long_term_memory: LongTermMemory | None = None,
    ):
        self.llm = llm
        self.tools = tools
        self.config = config
        self.long_term_memory = long_term_memory

    def execute_subtask(
        self,
        subtask: dict,
        plan_context: str = "",
        max_iterations: int | None = None,
    ) -> dict:
        """Execute a single sub-task.

        Args:
            subtask: Dict with at least "id" and "description".
            plan_context: Summary of the overall plan for context.
            max_iterations: Override for max ReAct iterations.

        Returns:
            dict with "status" ("success"|"failed"|"timeout"),
            "result", and "steps_used".
        """
        task_desc = subtask.get("description", str(subtask))
        task_id = subtask.get("id", "?")
        tools_hint = subtask.get("tools_hint", [])

        system = EXECUTOR_SYSTEM
        if plan_context:
            system += f"\n\nOverall plan context:\n{plan_context}"
        if tools_hint:
            system += f"\n\nSuggested tools: {', '.join(tools_hint)}"

        agent = AgentCore(
            llm=self.llm,
            tools=self.tools,
            config=self.config,
            long_term_memory=self.long_term_memory,
            system_prompt=system,
            role_name=f"Executor-{task_id}",
        )

        log.info("Executing sub-task %s: %s", task_id, task_desc[:100])
        result = agent.run(task_desc, max_iterations=max_iterations)

        if result.startswith("FAILED:"):
            status = "failed"
        elif result.startswith("TIMEOUT:") or result.startswith("MAX_ITERATIONS:"):
            status = "timeout"
        else:
            status = "success"

        return {
            "task_id": task_id,
            "status": status,
            "result": result,
            "steps_used": agent.step_count,
        }
