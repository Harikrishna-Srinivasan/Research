"""Shell command execution tool — sandboxed with safety checks."""

from __future__ import annotations

import os
import subprocess
import signal
from pathlib import Path
from typing import Optional

from deepagent.utils.cleanup import get_tracker
from deepagent.utils.logger import get_logger

log = get_logger(__name__)


def run_command(
    command: str,
    cwd: str = ".",
    timeout: int = 300,
    allowed_commands: list[str] | None = None,
    blocked_patterns: list[str] | None = None,
) -> str:
    """Execute a shell command with safety guardrails.

    Returns combined stdout + stderr, truncated to ~8000 chars.
    """
    # ---- Safety checks ----
    if blocked_patterns:
        for pat in blocked_patterns:
            if pat.lower() in command.lower():
                return f"BLOCKED: Command contains dangerous pattern: '{pat}'"

    if allowed_commands:
        cmd_base = command.split()[0] if command.strip() else ""
        # Allow compound commands like 'pip install' if 'pip' is allowed
        if not any(cmd_base.startswith(ac) for ac in allowed_commands):
            return (
                f"BLOCKED: '{cmd_base}' is not in allowed commands: {allowed_commands}. "
                f"If needed, ask the user to add it to config.yaml."
            )

    # ---- Execute ----
    resolved_cwd = str(Path(cwd).resolve())
    log.info("Running: %s (cwd=%s, timeout=%ds)", command, resolved_cwd, timeout)

    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=resolved_cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )

        output_parts = []
        if result.stdout:
            output_parts.append(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            output_parts.append(f"STDERR:\n{result.stderr}")

        output = "\n".join(output_parts) or "(no output)"

        # Truncate if very long
        if len(output) > 8000:
            output = output[:4000] + "\n\n... [truncated] ...\n\n" + output[-4000:]

        status = f"Exit code: {result.returncode}"
        return f"{status}\n{output}"

    except subprocess.TimeoutExpired:
        return f"ERROR: Command timed out after {timeout}s: {command}"
    except Exception as e:
        return f"ERROR executing command: {e}"


def install_package(package: str, manager: str = "pip") -> str:
    """Install a package and track it for cleanup."""
    if manager == "pip":
        result = run_command(f"pip install {package}", allowed_commands=["pip"])
        if "Successfully installed" in result or "already satisfied" in result:
            get_tracker().track_pip_package(package.split("==")[0].split(">=")[0])
    elif manager == "npm":
        result = run_command(f"npm install {package}", allowed_commands=["npm"])
        if "added" in result:
            get_tracker().track_npm_package(package)
    else:
        return f"Unknown package manager: {manager}"
    return result


SHELL_TOOLS = [
    {
        "name": "run_command",
        "description": "Execute a shell command. Has safety guardrails against dangerous operations.",
        "parameters": {
            "command": {"type": "string", "description": "Shell command to execute"},
            "cwd": {"type": "string", "description": "Working directory", "default": "."},
            "timeout": {"type": "integer", "description": "Max seconds", "default": 300},
        },
        "function": run_command,
    },
    {
        "name": "install_package",
        "description": "Install a pip or npm package. Automatically tracked for cleanup.",
        "parameters": {
            "package": {"type": "string", "description": "Package name (e.g. 'flask', 'react')"},
            "manager": {"type": "string", "description": "'pip' or 'npm'", "default": "pip"},
        },
        "function": install_package,
    },
]
