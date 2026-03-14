"""Agent Core — the ReAct reasoning loop."""

from __future__ import annotations

import time
from typing import Optional

from deepagent.config import DeepAgentConfig
from deepagent.llm.backend import LLMBackend
from deepagent.memory.short_term import ShortTermMemory
from deepagent.memory.long_term import LongTermMemory
from deepagent.tools.registry import ToolRegistry
from deepagent.utils import alerts
from deepagent.utils.logger import get_logger

log = get_logger(__name__)


class AgentCore:
    """ReAct agent loop — Think → Act → Observe, repeat.

    This is the fundamental execution unit. The Planner, Executor, and
    Critic are all built on top of this.
    """

    def __init__(
        self,
        llm: LLMBackend,
        tools: ToolRegistry,
        config: DeepAgentConfig,
        memory: ShortTermMemory | None = None,
        long_term_memory: LongTermMemory | None = None,
        system_prompt: str = "",
        role_name: str = "Agent",
    ):
        self.llm = llm
        self.tools = tools
        self.config = config
        self.memory = memory or ShortTermMemory(
            max_tokens=config.memory.short_term_tokens,
            token_counter=llm.count_tokens,
        )
        self.long_term_memory = long_term_memory
        self.role_name = role_name
        self._step_count = 0

        if system_prompt:
            self.memory.add("system", system_prompt, priority=100)

    def run(self, task: str, max_iterations: int | None = None) -> str:
        """Execute a task using the ReAct loop.

        Returns the final result string.
        """
        max_iter = max_iterations or self.config.agent.max_iterations
        self.memory.add("user", f"Task: {task}", priority=90)

        # Look up relevant long-term memories
        if self.long_term_memory and self.long_term_memory.count > 0:
            relevant = self.long_term_memory.search_formatted(task, n_results=3)
            if "No memories found" not in relevant:
                self.memory.add("system", f"Relevant past knowledge:\n{relevant}", priority=50)

        start_time = time.time()
        timeout = self.config.agent.max_time_minutes * 60

        for iteration in range(1, max_iter + 1):
            self._step_count = iteration
            elapsed = time.time() - start_time
            if elapsed > timeout:
                return f"TIMEOUT: Task exceeded {self.config.agent.max_time_minutes} minute time limit."

            log.info("[%s] Step %d/%d (%.0fs elapsed)", self.role_name, iteration, max_iter, elapsed)

            # --- Generate thought + action ---
            messages = self.memory.get_messages()
            tool_schemas = self.tools.get_tool_schemas()

            response = self.llm.generate_with_tools(messages, tool_schemas)

            if response["tool_calls"]:
                tool_call = response["tool_calls"][0]
                thought = response.get("content") or ""
                tool_name = tool_call["name"]
                tool_args = tool_call["arguments"]

                # Log the thought
                if thought:
                    log.info("[%s] Thought: %s", self.role_name, thought[:200])
                log.info("[%s] Action: %s(%s)", self.role_name, tool_name, _truncate_args(tool_args))

                # Store in memory
                action_text = f"Thought: {thought}\nAction: {tool_name}({_truncate_args(tool_args)})"
                self.memory.add("assistant", action_text, priority=30)

                # --- Check for completion signals ---
                if tool_name == "task_complete":
                    result = tool_args.get("result", "Task completed.")
                    log.info("[%s] ✅ Complete: %s", self.role_name, result[:200])

                    # Store in long-term memory
                    if self.long_term_memory:
                        self.long_term_memory.store(
                            f"Task: {task}\nResult: {result}",
                            category="task_result",
                        )
                    return result

                if tool_name == "task_failed":
                    reason = tool_args.get("reason", "Unknown failure.")
                    log.warning("[%s] ❌ Failed: %s", self.role_name, reason[:200])
                    return f"FAILED: {reason}"

                # --- Execute the tool ---
                observation = self.tools.call(tool_name, tool_args)
                log.info("[%s] Observation: %s", self.role_name, observation[:200])
                self.memory.add("user", f"Observation:\n{observation}", priority=20)

            else:
                # No tool call — the model just produced text
                text = response.get("content", "")
                if text:
                    self.memory.add("assistant", text, priority=30)
                    # Nudge it to use a tool
                    self.memory.add(
                        "user",
                        "Please use a tool to make progress, or call task_complete if done.",
                        priority=10,
                    )

        return f"MAX_ITERATIONS: Reached {max_iter} iterations without completing the task."

    @property
    def step_count(self) -> int:
        return self._step_count


def _truncate_args(args: dict, max_len: int = 200) -> str:
    """Truncate tool arguments for display."""
    text = str(args)
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text
