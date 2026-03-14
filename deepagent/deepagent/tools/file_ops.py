"""File operations tools — read, write, search, list."""

from __future__ import annotations

import os
import fnmatch
from pathlib import Path
from typing import Optional

from deepagent.utils.cleanup import get_tracker


def _safe_check(path: str, workspace_dirs: list[str] | None = None) -> str:
    """Resolve path and check it's within an allowed workspace."""
    resolved = str(Path(path).resolve())
    if workspace_dirs:
        if not any(resolved.startswith(str(Path(w).resolve())) for w in workspace_dirs):
            raise PermissionError(
                f"Path {resolved} is outside allowed workspace dirs: {workspace_dirs}"
            )
    return resolved


def read_file(path: str, start_line: int = 0, end_line: int = -1) -> str:
    """Read a file and return its contents (or a line range)."""
    path = str(Path(path).resolve())
    if not os.path.isfile(path):
        return f"ERROR: File not found: {path}"
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        if end_line == -1:
            end_line = len(lines)
        selected = lines[start_line:end_line]
        # Add line numbers
        numbered = [f"{start_line + i + 1:>4} | {l}" for i, l in enumerate(selected)]
        return f"File: {path} ({len(lines)} lines total)\n" + "".join(numbered)
    except Exception as e:
        return f"ERROR reading {path}: {e}"


def write_file(path: str, content: str, workspace_dirs: list[str] | None = None) -> str:
    """Write content to a file, creating parent directories if needed."""
    try:
        resolved = _safe_check(path, workspace_dirs)
        os.makedirs(os.path.dirname(resolved), exist_ok=True)
        with open(resolved, "w", encoding="utf-8") as f:
            f.write(content)
        get_tracker().track_file(resolved)
        return f"Successfully wrote {len(content)} chars to {resolved}"
    except Exception as e:
        return f"ERROR writing {path}: {e}"


def append_file(path: str, content: str, workspace_dirs: list[str] | None = None) -> str:
    """Append content to an existing file."""
    try:
        resolved = _safe_check(path, workspace_dirs)
        with open(resolved, "a", encoding="utf-8") as f:
            f.write(content)
        return f"Appended {len(content)} chars to {resolved}"
    except Exception as e:
        return f"ERROR appending to {path}: {e}"


def list_directory(path: str = ".", recursive: bool = False, max_depth: int = 3) -> str:
    """List files and directories."""
    resolved = str(Path(path).resolve())
    if not os.path.isdir(resolved):
        return f"ERROR: Not a directory: {resolved}"
    
    entries = []
    if recursive:
        for root, dirs, files in os.walk(resolved):
            depth = root.replace(resolved, "").count(os.sep)
            if depth >= max_depth:
                dirs.clear()
                continue
            indent = "  " * depth
            entries.append(f"{indent}📁 {os.path.basename(root)}/")
            for f in sorted(files)[:50]:  # Cap per directory
                size = os.path.getsize(os.path.join(root, f))
                entries.append(f"{indent}  📄 {f} ({_human_size(size)})")
    else:
        for item in sorted(os.listdir(resolved))[:100]:
            full = os.path.join(resolved, item)
            if os.path.isdir(full):
                entries.append(f"📁 {item}/")
            else:
                size = os.path.getsize(full)
                entries.append(f"📄 {item} ({_human_size(size)})")
    
    return f"Directory: {resolved}\n" + "\n".join(entries) if entries else f"Empty directory: {resolved}"


def find_files(directory: str, pattern: str = "*", max_results: int = 50) -> str:
    """Find files matching a glob pattern."""
    resolved = str(Path(directory).resolve())
    results = []
    for root, _, files in os.walk(resolved):
        for f in files:
            if fnmatch.fnmatch(f, pattern):
                full = os.path.join(root, f)
                results.append(full)
                if len(results) >= max_results:
                    break
        if len(results) >= max_results:
            break
    
    if results:
        return f"Found {len(results)} files matching '{pattern}':\n" + "\n".join(results)
    return f"No files matching '{pattern}' in {resolved}"


def search_in_files(directory: str, query: str, extensions: str = ".py,.js,.ts,.md,.txt,.yaml,.json") -> str:
    """Search for text within files."""
    resolved = str(Path(directory).resolve())
    ext_list = [e.strip() for e in extensions.split(",")]
    matches = []
    
    for root, _, files in os.walk(resolved):
        for f in files:
            if not any(f.endswith(ext) for ext in ext_list):
                continue
            full = os.path.join(root, f)
            try:
                with open(full, "r", encoding="utf-8", errors="ignore") as fh:
                    for i, line in enumerate(fh, 1):
                        if query.lower() in line.lower():
                            matches.append(f"{full}:{i}: {line.strip()}")
                            if len(matches) >= 50:
                                break
            except Exception:
                continue
        if len(matches) >= 50:
            break
    
    if matches:
        return f"Found {len(matches)} matches for '{query}':\n" + "\n".join(matches)
    return f"No matches for '{query}' in {resolved}"


def _human_size(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.0f}{unit}" if unit == "B" else f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


# ---- Tool schema registry ----
FILE_TOOLS = [
    {
        "name": "read_file",
        "description": "Read contents of a file. Returns line-numbered content.",
        "parameters": {
            "path": {"type": "string", "description": "File path to read"},
            "start_line": {"type": "integer", "description": "Start line (0-indexed, optional)", "default": 0},
            "end_line": {"type": "integer", "description": "End line (-1 for all, optional)", "default": -1},
        },
        "function": read_file,
    },
    {
        "name": "write_file",
        "description": "Write content to a file. Creates parent directories automatically.",
        "parameters": {
            "path": {"type": "string", "description": "File path to write to"},
            "content": {"type": "string", "description": "Content to write"},
        },
        "function": write_file,
    },
    {
        "name": "append_file",
        "description": "Append content to an existing file.",
        "parameters": {
            "path": {"type": "string", "description": "File path"},
            "content": {"type": "string", "description": "Content to append"},
        },
        "function": append_file,
    },
    {
        "name": "list_directory",
        "description": "List files and directories at a path.",
        "parameters": {
            "path": {"type": "string", "description": "Directory path", "default": "."},
            "recursive": {"type": "boolean", "description": "List recursively", "default": False},
        },
        "function": list_directory,
    },
    {
        "name": "find_files",
        "description": "Find files matching a glob pattern (e.g. '*.py').",
        "parameters": {
            "directory": {"type": "string", "description": "Root directory to search"},
            "pattern": {"type": "string", "description": "Glob pattern like '*.py'"},
        },
        "function": find_files,
    },
    {
        "name": "search_in_files",
        "description": "Search for a text string inside files (case-insensitive grep).",
        "parameters": {
            "directory": {"type": "string", "description": "Root directory"},
            "query": {"type": "string", "description": "Text to search for"},
        },
        "function": search_in_files,
    },
]
