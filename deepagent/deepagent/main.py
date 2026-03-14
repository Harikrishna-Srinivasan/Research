"""DeepAgent CLI — entry point for the autonomous research agent."""

from __future__ import annotations

import argparse
import sys

from deepagent.config import load_config


def main():
    parser = argparse.ArgumentParser(
        prog="deepagent",
        description="DeepAgent — Autonomous Research Agent powered by open-source LLMs & MCP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  deepagent run "Build a full-stack todo app with React and FastAPI"
  deepagent run "Find recent papers on mixture-of-experts on ArXiv and summarize them"
  deepagent run "Find primes using optimized Lucas-Lehmer tests"
  deepagent interactive
  deepagent tools
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # --- run ---
    run_parser = subparsers.add_parser("run", help="Execute a task autonomously")
    run_parser.add_argument("task", type=str, help="Task description in natural language")
    run_parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    run_parser.add_argument("--model", type=str, default=None, help="Override model ID")
    run_parser.add_argument("--no-cleanup", action="store_true", help="Keep all created files")
    run_parser.add_argument("--no-critic", action="store_true", help="Disable the Critic agent")
    run_parser.add_argument("--max-iterations", type=int, default=None, help="Max iterations per sub-task")

    # --- interactive ---
    inter_parser = subparsers.add_parser("interactive", help="Interactive chat mode")
    inter_parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    inter_parser.add_argument("--model", type=str, default=None, help="Override model ID")

    # --- tools ---
    tools_parser = subparsers.add_parser("tools", help="List all available tools")
    tools_parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")

    # --- mcp-server ---
    mcp_parser = subparsers.add_parser("mcp-server", help="Run as an MCP server (stdio transport)")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # --- Execute commands ---

    if args.command == "run":
        config = load_config(args.config)
        if args.model:
            config.llm["model_id"] = args.model
        if args.no_cleanup:
            config.safety["auto_cleanup"] = False
        if args.no_critic:
            config.agent["enable_critic"] = False
        if args.max_iterations:
            config.agent["max_iterations"] = args.max_iterations

        from deepagent.agent.orchestrator import Orchestrator

        orchestrator = Orchestrator(config)
        result = orchestrator.run(args.task)
        print(f"\n{'=' * 60}")
        print(result)

    elif args.command == "interactive":
        config = load_config(args.config)
        if args.model:
            config.llm.model_id = args.model

        from deepagent.agent.orchestrator import Orchestrator

        orchestrator = Orchestrator(config)
        orchestrator.run_interactive()

    elif args.command == "tools":
        config = load_config(args.config)
        from deepagent.tools.registry import ToolRegistry

        registry = ToolRegistry(config)
        print(f"\n🔧 DeepAgent Tools ({len(registry)} available):\n")
        for schema in registry.get_tool_schemas():
            print(f"  • {schema['name']}")
            print(f"    {schema['description']}")
            if schema.get("parameters"):
                params = ", ".join(
                    f"{k}: {v.get('type', '?')}" for k, v in schema["parameters"].items()
                )
                print(f"    Params: ({params})")
            print()

    elif args.command == "mcp-server":
        from deepagent.tools.mcp_server import run_mcp_server

        run_mcp_server()


if __name__ == "__main__":
    main()
