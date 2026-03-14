"""Orchestrator — coordinates the Planner → Executor → Critic pipeline."""

from __future__ import annotations

import json
import time
from typing import Optional

from deepagent.config import DeepAgentConfig, load_config
from deepagent.llm.backend import LLMBackend, create_backend
from deepagent.agent.planner import Planner
from deepagent.agent.executor import Executor
from deepagent.agent.critic import Critic
from deepagent.memory.long_term import LongTermMemory
from deepagent.tools.registry import ToolRegistry
from deepagent.utils import alerts
from deepagent.utils.cleanup import get_tracker
from deepagent.utils.logger import get_logger

log = get_logger(__name__)


class Orchestrator:
    """Top-level coordinator.

    Plan → Execute all sub-tasks → Critic reviews →
    Re-plan if needed → repeat.
    """

    def __init__(self, config: DeepAgentConfig | None = None):
        self.config = config or load_config()

        alerts.info(f"Initializing DeepAgent with model: {self.config.llm['model_id']}")

        # Core components
        self.llm: LLMBackend = create_backend(self.config.llm)
        self.tools = ToolRegistry(self.config)
        self.long_term_memory = LongTermMemory(
            db_path=self.config.memory["long_term_db_path"],
            embedding_model=self.config.memory["embedding_model"],
        )

        # Agents
        self.planner = Planner(self.llm, self.config)
        self.executor = Executor(self.llm, self.tools, self.config, self.long_term_memory)
        self.critic = Critic(self.llm, self.config) if self.config.agent["enable_critic"] else None

        alerts.success(
            f"DeepAgent ready!\n"
            f"  Model: {self.config.llm['model_id']}\n"
            f"  Tools: {len(self.tools)} available\n"
            f"  Memory: {self.long_term_memory.count} stored entries\n"
            f"  Critic: {'enabled' if self.critic else 'disabled'}",
            title="Initialized",
        )

    def run(self, goal: str, max_revisions: int = 3) -> str:
        """Execute a complete task from goal to finish.

        1. Planner creates a plan
        2. Executor runs each sub-task
        3. Critic reviews each result
        4. If critic says FAIL, re-plan and retry (up to max_revisions)
        5. Return final summary
        """
        start_time = time.time()
        alerts.info(f"Goal: {goal}", title="🚀 Starting Task")

        # --- Phase 1: Plan ---
        plan = self.planner.create_plan(goal)
        alerts.info(
            f"Plan: {plan.get('plan_summary', '?')}\n"
            f"Sub-tasks: {len(plan.get('sub_tasks', []))}",
            title="📋 Plan Created",
        )

        all_results = []
        revision = 0

        while revision <= max_revisions:
            sub_tasks = plan.get("sub_tasks", [])
            total = len(sub_tasks)
            failed_tasks = []

            # --- Phase 2: Execute ---
            for i, subtask in enumerate(sub_tasks, 1):
                desc = subtask.get("description", str(subtask))
                alerts.progress_update("Executing", i, total, desc[:80])

                result = self.executor.execute_subtask(
                    subtask,
                    plan_context=plan.get("plan_summary", ""),
                )
                all_results.append(result)

                # --- Phase 3: Critic review ---
                if self.critic and result["status"] == "success":
                    review = self.critic.review(desc, result["result"])
                    result["review"] = review

                    if review.get("verdict") == "FAIL":
                        failed_tasks.append({
                            "task": subtask,
                            "result": result,
                            "review": review,
                        })
                        alerts.warning(
                            f"Sub-task {subtask.get('id', '?')} failed review "
                            f"(score: {review.get('quality_score', '?')}/10)\n"
                            f"Issues: {', '.join(review.get('issues', []))}",
                            title="Critic Review",
                        )
                elif result["status"] in ("failed", "timeout"):
                    failed_tasks.append({
                        "task": subtask,
                        "result": result,
                        "review": None,
                    })

            # --- Phase 4: Re-plan if needed ---
            if failed_tasks and revision < max_revisions:
                revision += 1
                feedback = self._format_failure_feedback(failed_tasks)
                alerts.warning(
                    f"{len(failed_tasks)} sub-task(s) need revision (attempt {revision}/{max_revisions})",
                    title="Re-planning",
                )
                plan = self.planner.revise_plan(plan, feedback)
            else:
                break

        # --- Final summary ---
        elapsed = time.time() - start_time
        summary = self._build_summary(goal, all_results, elapsed)

        # Store in long-term memory
        self.long_term_memory.store(summary, category="completed_task")

        # Cleanup
        if self.config.safety["auto_cleanup"]:
            tracker = get_tracker()
            cleanup_actions = tracker.cleanup()
            if cleanup_actions:
                alerts.info(f"Cleaned up {len(cleanup_actions)} resources", title="🧹 Cleanup")

        alerts.success(summary, title="✅ Task Complete")
        return summary

    def run_interactive(self) -> None:
        """Run in interactive mode — chat-style interface."""
        alerts.info(
            "Type your task or question. Type 'quit' to exit.\n"
            "Commands: /tools, /memory <query>, /plan <goal>",
            title="🤖 DeepAgent Interactive Mode",
        )

        while True:
            try:
                user_input = input("\n🧑 You: ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                break

            # Special commands
            if user_input.startswith("/tools"):
                print("\nAvailable tools:")
                for name in self.tools.list_tools():
                    print(f"  • {name}")
                continue

            if user_input.startswith("/memory"):
                query = user_input[7:].strip() or "recent"
                result = self.long_term_memory.search_formatted(query)
                print(f"\n{result}")
                continue

            if user_input.startswith("/plan"):
                goal = user_input[5:].strip()
                if goal:
                    plan = self.planner.create_plan(goal)
                    print(f"\n{json.dumps(plan, indent=2)}")
                else:
                    print("Usage: /plan <goal>")
                continue

            # Regular task
            result = self.run(user_input)
            print(f"\n🤖 Agent: {result}")

    @staticmethod
    def _format_failure_feedback(failed: list[dict]) -> str:
        lines = []
        for f in failed:
            task = f["task"]
            result = f["result"]
            review = f.get("review")
            lines.append(f"Sub-task {task.get('id', '?')}: {task.get('description', '?')[:100]}")
            lines.append(f"  Status: {result['status']}")
            lines.append(f"  Result: {result['result'][:200]}")
            if review:
                lines.append(f"  Critic issues: {review.get('issues', [])}")
                lines.append(f"  Suggestions: {review.get('suggestions', [])}")
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _build_summary(goal: str, results: list[dict], elapsed: float) -> str:
        successes = len([r for r in results if r["status"] == "success"])
        failures = len([r for r in results if r["status"] != "success"])
        total_steps = sum(r.get("steps_used", 0) for r in results)
        mins = elapsed / 60

        lines = [
            f"Goal: {goal}",
            f"Completed: {successes}/{successes + failures} sub-tasks",
            f"Total steps: {total_steps}",
            f"Time: {mins:.1f} minutes",
            "",
            "Results:",
        ]
        for r in results:
            status_icon = "✅" if r["status"] == "success" else "❌"
            lines.append(f"  {status_icon} Task {r.get('task_id', '?')}: {r['result'][:150]}")

        return "\n".join(lines)
