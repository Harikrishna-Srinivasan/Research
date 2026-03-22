import sys
import os
import logging

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepagent.config import load_config
from deepagent.llm.backend import create_backend
from deepagent.agent.orchestrator import MultiAgentOrchestrator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def main():
    print("=====================================================")
    print("  DeepAgent: Dynamic Multi-Agent Research Assembly   ")
    print("=====================================================\n")

    # 1. Load config
    config = load_config()

    print("Roles loaded from config:")
    for role, path in config.llm.roles_to_models.items():
         print(f"  - {role}: {path}")
         
    print("\nNote: Models will be dynamically downloaded via HuggingFace if not local.")
    print("Executing research with mixed backends (Transformers + LlamaCpp).")

    # 2. Instantiate Orchestrator (Backend is now internally routed)
    # The first parameter `backend` is deprecated in usage but kept for signature.
    dummy_backend = None 
    orchestrator = MultiAgentOrchestrator(dummy_backend, config)

    # 3. Give the Worker a starting topic
    topic = "The potential architectural advantages of routing separate sub-tasks of an LLM loop to distinct specialized quantization variants on limited hardware constraints."
    size = "medium"

    print(f"\n[Orchestrator] Sending topic to the Dynamic Assembly Line: {topic}\n")
    final_idea = orchestrator.run_research(topic, research_size=size)

    if final_idea:
        print("\n=====================================================")
        print("                CONSENSUS REACHED                    ")
        print("=====================================================")
        print(f"FINAL PROPOSAL:\n{final_idea}")
    else:
        print("\n=====================================================")
        print("             PROPOSAL FAILED OR STALLED              ")
        print("=====================================================")
        print("Check hold_queue.txt for details.")

if __name__ == "__main__":
    main()
