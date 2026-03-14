"""MCP Client — connects to external MCP servers for additional tools."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Optional

from deepagent.utils.logger import get_logger

log = get_logger(__name__)


class MCPClient:
    """Connect to an external MCP server and call tools on it.

    This allows the agent to use tools provided by *any* MCP-compliant
    server — not just our built-in tools.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, Any] = {}  # server_name -> session
        self._available_tools: dict[str, dict] = {}  # tool_name -> {server, schema}

    async def connect_stdio(self, name: str, command: str, args: list[str] | None = None):
        """Connect to an MCP server via stdio transport."""
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client

            params = StdioServerParameters(
                command=command,
                args=args or [],
            )

            transport = stdio_client(params)
            read, write = await transport.__aenter__()
            session = ClientSession(read, write)
            await session.__aenter__()
            await session.initialize()

            self._sessions[name] = {
                "session": session,
                "transport": transport,
            }

            # Discover tools
            tool_list = await session.list_tools()
            for tool in tool_list.tools:
                self._available_tools[tool.name] = {
                    "server": name,
                    "description": tool.description or "",
                    "schema": tool.inputSchema if hasattr(tool, "inputSchema") else {},
                }

            log.info(
                "Connected to MCP server '%s' — %d tools available",
                name, len(tool_list.tools),
            )
        except Exception as e:
            log.error("Failed to connect to MCP server '%s': %s", name, e)
            raise

    def list_external_tools(self) -> list[dict]:
        """List all tools from connected MCP servers."""
        tools = []
        for name, info in self._available_tools.items():
            tools.append({
                "name": name,
                "description": info["description"],
                "parameters": info.get("schema", {}),
                "source": f"mcp:{info['server']}",
            })
        return tools

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        """Call a tool on an external MCP server."""
        if tool_name not in self._available_tools:
            return f"ERROR: Tool '{tool_name}' not found in any connected MCP server."

        info = self._available_tools[tool_name]
        session = self._sessions[info["server"]]["session"]

        try:
            result = await session.call_tool(tool_name, arguments)
            # MCP returns content objects
            if hasattr(result, "content"):
                parts = []
                for item in result.content:
                    if hasattr(item, "text"):
                        parts.append(item.text)
                    else:
                        parts.append(str(item))
                return "\n".join(parts)
            return str(result)
        except Exception as e:
            return f"ERROR calling MCP tool '{tool_name}': {e}"

    async def disconnect_all(self):
        """Disconnect from all MCP servers."""
        for name, info in self._sessions.items():
            try:
                await info["session"].__aexit__(None, None, None)
                await info["transport"].__aexit__(None, None, None)
            except Exception:
                pass
        self._sessions.clear()
        self._available_tools.clear()
