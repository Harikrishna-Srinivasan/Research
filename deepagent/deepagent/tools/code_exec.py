"""Sandboxed Python code execution tool."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import os
from pathlib import Path

from deepagent.utils.cleanup import get_tracker


def execute_python(code: str, timeout: int = 120) -> str:
    """Execute Python code in a sandboxed subprocess.

    The code is written to a temp file and executed with the current
    Python interpreter.  stdout and stderr are captured.
    """
    # Write to temp file
    tmp_dir = tempfile.mkdtemp(prefix="deepagent_exec_")
    get_tracker().track_directory(tmp_dir)
    script_path = os.path.join(tmp_dir, "script.py")

    with open(script_path, "w", encoding="utf-8") as f:
        f.write(code)

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=tmp_dir,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )

        parts = []
        if result.stdout:
            parts.append(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            parts.append(f"STDERR:\n{result.stderr}")

        output = "\n".join(parts) or "(no output)"

        if len(output) > 8000:
            output = output[:4000] + "\n... [truncated] ...\n" + output[-4000:]

        return f"Exit code: {result.returncode}\n{output}"

    except subprocess.TimeoutExpired:
        return f"ERROR: Code execution timed out after {timeout}s"
    except Exception as e:
        return f"ERROR executing code: {e}"


def execute_python_expression(expression: str) -> str:
    """Evaluate a single Python expression and return the result."""
    code = f"print(repr({expression}))"
    return execute_python(code, timeout=30)


CODE_EXEC_TOOLS = [
    {
        "name": "execute_python",
        "description": "Execute a Python script in a sandboxed subprocess. Returns stdout/stderr.",
        "parameters": {
            "code": {"type": "string", "description": "Python code to execute"},
            "timeout": {"type": "integer", "description": "Max seconds", "default": 120},
        },
        "function": execute_python,
    },
    {
        "name": "execute_python_expression",
        "description": "Evaluate a single Python expression and return its value.",
        "parameters": {
            "expression": {"type": "string", "description": "Python expression to evaluate"},
        },
        "function": execute_python_expression,
    },
]
