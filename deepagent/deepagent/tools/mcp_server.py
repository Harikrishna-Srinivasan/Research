"""MCP Server — exposes all DeepAgent tools via Model Context Protocol."""

from __future__ import annotations

from typing import Any


def create_mcp_server():
    """Create a FastMCP server with all built-in tools registered.

    This server can be run standalone (for external MCP clients) or
    used internally by the agent.
    """
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        raise ImportError("MCP SDK not installed. Run: pip install mcp")

    server = FastMCP("DeepAgent Tools")

    # Register all built-in tools
    from deepagent.tools.file_ops import FILE_TOOLS
    from deepagent.tools.shell import SHELL_TOOLS
    from deepagent.tools.web import WEB_TOOLS
    from deepagent.tools.code_exec import CODE_EXEC_TOOLS
    from deepagent.tools.research import RESEARCH_TOOLS

    all_tools = FILE_TOOLS + SHELL_TOOLS + WEB_TOOLS + CODE_EXEC_TOOLS + RESEARCH_TOOLS

    for tool_def in all_tools:
        fn = tool_def["function"]
        name = tool_def["name"]
        desc = tool_def["description"]
        # Use MCP's @server.tool() registration
        server.tool(name=name, description=desc)(fn)

    return server


def run_mcp_server():
    """Run the MCP server (stdio transport)."""
    server = create_mcp_server()
    server.run()


if __name__ == "__main__":
    run_mcp_server()
