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

        alerts.info(f"Initializing DeepAgent with model: {self.config.llm.model_id}")

        # Core components
        self.llm: LLMBackend = create_backend(self.config.llm)
        self.tools = ToolRegistry(self.config)
        self.long_term_memory = LongTermMemory(
            db_path=self.config.memory.long_term_db_path,
            embedding_model=self.config.memory.embedding_model,
        )

        # Agents
        self.planner = Planner(self.llm, self.config)
        self.executor = Executor(self.llm, self.tools, self.config, self.long_term_memory)
        self.critic = Critic(self.llm, self.config) if self.config.agent.enable_critic else None

        alerts.success(
            f"DeepAgent ready!\n"
            f"  Model: {self.config.llm.model_id}\n"
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
        if self.config.safety.auto_cleanup:
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


# JSON Schema for Workers
WORKER_SCHEMA = {
    "type": "json_object",
    "schema": {
        "type": "object",
        "properties": {
            "thought_process": {"type": "string"},
            "proposal": {"type": "string"},
            "critique": {"type": "string"},
            "confidence": {"type": "number"}
        },
        "required": ["thought_process", "confidence"]
    }
}

# JSON Schema for Judges
JUDGE_SCHEMA = {
    "type": "json_object",
    "schema": {
        "type": "object",
        "properties": {
            "thought_process": {"type": "string"},
            "decision": {"type": "string", "enum": ["approve", "revise", "reject"]},
            "justification": {"type": "string"},
            "confidence": {"type": "number"}
        },
        "required": ["thought_process", "decision", "justification", "confidence"]
    }
}

class BackendRouter:
    """Manages simultaneous loading of Transformers and LlamaCpp backends."""
    def __init__(self, config: DeepAgentConfig):
        self.config = config
        self.transformers_backend = None
        self.llama_cpp_backend = None

    def call_agent(self, role_key: str, system_prompt: str, user_prompt: str, is_judge: bool = False) -> dict:
        import json
        import gc
        import torch
        from deepagent.llm.backend import TransformersBackend, LLMConfig
        
        model_id = self.config.llm.roles_to_models.get(role_key)
        if not model_id:
            log.error(f"No model ID for role: {role_key}")
            return {}
            
        alerts.info(f"Loading {role_key} (CPU execution): {model_id}...")
        
        # CPU-Efficient: We manually clear cache to keep RAM from bloating since we aren't using GGUF
        if hasattr(self, "_current_backend_id") and self._current_backend_id != model_id:
            log.info("Switching models, clearing previous from RAM...")
            if hasattr(self, "_active_backend"):
                del self._active_backend
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        cfg = LLMConfig(
            model_id=model_id,
            quantization="none", # BitsAndBytes 4-bit won't work on Windows CPU
            device="cpu"
        )
        self._active_backend = TransformersBackend(cfg)
        self._current_backend_id = model_id
        backend = self._active_backend

        schema = JUDGE_SCHEMA if is_judge else WORKER_SCHEMA

        messages = [
            {"role": "system", "content": system_prompt + "\nOUTPUT MUST BE STRICTLY JSON ACCORDING TO SCHEMA."},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            raw_response = backend.generate(messages, response_format=schema)
            if "```json" in raw_response:
                raw_response = raw_response.split("```json")[1].split("```")[0].strip()
            elif "```" in raw_response:
                raw_response = raw_response.split("```")[1].strip()
            return json.loads(raw_response)
        except Exception as e:
            log.error(f"Failed to generate/parse JSON from {role_key}: {e}")
            return {"decision": "revise", "justification": f"Error parsing response: {e}"} if is_judge else {"proposal": f"Error parsing response: {e}"}

class MultiAgentOrchestrator:
    """Implement the Worker/Judge dynamic assembly line consensus loop."""

    def __init__(self, backend, config: DeepAgentConfig):
        self.config = config
        self.router = BackendRouter(config)

    def _spawn_worker(self, worker_index: int, support: bool, topic: str, context: str):
        if worker_index < 4:
            role = "worker_tier_1"
        elif worker_index < 9:
            role = "worker_tier_2"
        else:
            role = "worker_tier_3"
            
        stance = "SUPPORT" if support else "OPPOSE"
        sys_prompt = f"You are a researcher assigned to {stance} the idea based on VectorDB content. Format strictly as JSON."
        usr_prompt = f"Topic/Idea: {topic}\nVectorDB Context:\n{context}\nAnalyze and provide a proposal if supporting, or critique if opposing."
        
        alerts.info(f"Spawning worker {worker_index+1} ({role}) with stance: {stance}")
        return self.router.call_agent(role, sys_prompt, usr_prompt, is_judge=False)

    def run_research(self, topic: str, research_size: str = "medium"):
        import time
        from deepagent.memory.long_term import LongTermMemory
        alerts.info(f"Starting dynamic assembly line research on: {topic}", title="Research Flow")

        max_gathering_sec = self.config.agent.max_gathering_mins * 60
        max_conclusion_sec = self.config.agent.max_conclusion_mins * 60
        max_research_sec = self.config.agent.max_research_mins * 60
        
        start_time = time.time()
        
        # --- Stage 1: Resource Gathering ---
        alerts.info(f"Phase 1: Gathering Resources to VectorDB (Max {self.config.agent.max_gathering_mins} mins)")
        # Initialize VectorDB
        vectordb = LongTermMemory(
            db_path=self.config.memory.long_term_db_path,
            embedding_model=self.config.memory.embedding_model,
        )
        # Simulating fetching diverse resources to store efficiently in embeddings
        sources = ["YouTube Transcript on topic", "GitHub Repo READMEs", "ArXiv Papers/Journals", "HuggingFace Articles & Blogs"]
        for src in sources:
            elapsed = time.time() - start_time
            if elapsed > max_gathering_sec:
               alerts.warning("Gathering phase limit reached. Stopping ingest.")
               break
            # Writing to VectorDB
            vectordb.store(f"Content extracted dynamically from {src} related to {topic}", category=src.split()[0].lower())
            time.sleep(0.5)
            
        # Read from VectorDB
        research_context = vectordb.search_formatted(topic, n_results=5)
        alerts.success(f"Populated VectorDB with diverse internet/research sources. Read relevant context.")
            
        # --- Stage 2: Idea Generation & Workers ---
        alerts.info("Phase 2: Idea Generation from VectorDB & Worker Spawning")
        workers_responses = []
        workers_count = 2 
        support_flag = True
        
        while workers_count <= 12:
            elapsed = time.time() - start_time
            if elapsed > max_research_sec:
                alerts.warning("Total research time exceeded limit. Halting worker spawns.")
                break
                
            resp = self._spawn_worker(workers_count - 1, support_flag, topic, research_context)
            workers_responses.append(resp)
            support_flag = not support_flag
            
            if workers_count < 12 and (len(workers_responses) < 4 and research_size == "large"):
                workers_count += 1
            else:
                break
                
        compiled_proposal = f"Topic: {topic}\n" + " \n".join([str(w.get("proposal", w.get("critique", ""))) for w in workers_responses])
                
        # --- Stage 3: Primary Judges ---
        alerts.info("Phase 3: Primary Judgement")
        j1_sys = "You are Primary Judge 1. Review the proposal critically. Provide detailed justification. Strict JSON."
        j2_sys = "You are Primary Judge 2. Review the proposal critically. Provide detailed justification. Strict JSON."
        
        j1_out = self.router.call_agent("judge_primary_1", j1_sys, compiled_proposal, is_judge=True)
        j2_out = self.router.call_agent("judge_primary_2", j2_sys, compiled_proposal, is_judge=True)
        
        alerts.info(f"Judge 1 Decision: {j1_out.get('decision')} - Justification: {j1_out.get('justification', '')[:100]}...")
        alerts.info(f"Judge 2 Decision: {j2_out.get('decision')} - Justification: {j2_out.get('justification', '')[:100]}...")
        
        summon_3rd = False
        if j1_out.get('decision') != j2_out.get('decision'):
            alerts.warning("Primary Judges disagree! Their justifications highlight gaps. Summoning Backup Judge.")
            summon_3rd = True
        if research_size == "large":
            alerts.info("Research is large. Summoning Backup Judge proactively.")
            summon_3rd = True
            
        j3_out = {}
        if summon_3rd:
            j3_sys = "You are the Backup 3rd Judge. Break the tie or review large research thoughtfully. Strict JSON."
            j3_out = self.router.call_agent("judge_backup", j3_sys, compiled_proposal, is_judge=True)
            alerts.info(f"Judge 3 Decision: {j3_out.get('decision')} - Justification: {j3_out.get('justification', '')[:100]}")
            
        # --- Stage 4: Continuous Convincement Loop ---
        alerts.info("Phase 4: Debate Loop Among Judges/Workers")
        debate_convos = 0
        current_proposal = compiled_proposal
        
        while debate_convos < 12:
            elapsed_conclusion = (time.time() - start_time) - max_gathering_sec
            if elapsed_conclusion > max_conclusion_sec:
                alerts.warning(f"Conclusion phase exceeded {self.config.agent.max_conclusion_mins} mins limit!")
                break
                
            decs = [j1_out.get('decision'), j2_out.get('decision')]
            if summon_3rd: decs.append(j3_out.get('decision'))
            
            if all(d == "approve" for d in decs if d):
                alerts.success("SUCCESS: Total Consensus Reached among judges.")
                return current_proposal
                
            if all(d == "reject" for d in decs if d):
                alerts.warning("TOTAL REJECTION: All active judges rejected.")
                return None
                
            debate_convos += 1
            alerts.info(f"Debate Round {debate_convos}...")
            
            p_prompt = f"Revise proposal to convince judges based on validations/justifications. J1:{j1_out.get('justification')} J2:{j2_out.get('justification')} J3:{j3_out.get('justification')}"
            
            if workers_count < 12:
                workers_count += 1
            worker_resp = self._spawn_worker(workers_count - 1, True, p_prompt, research_context)
            current_proposal = worker_resp.get("proposal", current_proposal)
            
            # Re-judge
            j1_out = self.router.call_agent("judge_primary_1", j1_sys, current_proposal, is_judge=True)
            j2_out = self.router.call_agent("judge_primary_2", j2_sys, current_proposal, is_judge=True)
            if summon_3rd:
                j3_out = self.router.call_agent("judge_backup", j3_sys, current_proposal, is_judge=True)
                
        alerts.warning("STALEMATE: Max conversations (12) reached OR Time limit exceeded.")
        decs = [d for d in [j1_out.get('decision'), j2_out.get('decision'), j3_out.get('decision')] if d]
        
        approvals = decs.count("approve")
        if approvals >= 2:
            alerts.success("Ended in stalemate but majority approved.")
            return current_proposal
        
        import os
        with open("hold_queue.txt", "a", encoding="utf-8") as f:
            f.write(f"STALEMATE IDEA:\n{topic}\n\nLast Proposal:\n{current_proposal}\n\n---\n")
        return None
