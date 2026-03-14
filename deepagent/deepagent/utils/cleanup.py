"""Resource tracking and cleanup manager."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from deepagent.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class _Resource:
    kind: str  # "file", "directory", "pip_package", "npm_package", "process"
    path_or_name: str
    keep: bool = False


class ResourceTracker:
    """Track and optionally clean up all resources the agent creates."""

    def __init__(self) -> None:
        self._resources: list[_Resource] = []

    # ---- registration ----

    def track_file(self, path: str, keep: bool = False) -> None:
        self._resources.append(_Resource("file", str(Path(path).resolve()), keep))

    def track_directory(self, path: str, keep: bool = False) -> None:
        self._resources.append(_Resource("directory", str(Path(path).resolve()), keep))

    def track_pip_package(self, name: str, keep: bool = False) -> None:
        self._resources.append(_Resource("pip_package", name, keep))

    def track_npm_package(self, name: str, keep: bool = False) -> None:
        self._resources.append(_Resource("npm_package", name, keep))

    def track_process(self, pid: int) -> None:
        self._resources.append(_Resource("process", str(pid)))

    def mark_keep(self, path_or_name: str) -> None:
        """Mark a resource to be kept on cleanup."""
        for r in self._resources:
            if r.path_or_name == path_or_name:
                r.keep = True

    # ---- cleanup ----

    def cleanup(self, force: bool = False) -> list[str]:
        """Remove all tracked resources that are not marked as 'keep'.

        Returns list of actions taken.
        """
        actions: list[str] = []
        for r in reversed(self._resources):
            if r.keep and not force:
                continue
            try:
                if r.kind == "file" and os.path.isfile(r.path_or_name):
                    os.remove(r.path_or_name)
                    actions.append(f"Deleted file: {r.path_or_name}")
                elif r.kind == "directory" and os.path.isdir(r.path_or_name):
                    shutil.rmtree(r.path_or_name)
                    actions.append(f"Deleted directory: {r.path_or_name}")
                elif r.kind == "pip_package":
                    subprocess.run(
                        ["pip", "uninstall", "-y", r.path_or_name],
                        capture_output=True, timeout=60,
                    )
                    actions.append(f"Uninstalled pip package: {r.path_or_name}")
                elif r.kind == "npm_package":
                    subprocess.run(
                        ["npm", "uninstall", r.path_or_name],
                        capture_output=True, timeout=60,
                    )
                    actions.append(f"Uninstalled npm package: {r.path_or_name}")
                elif r.kind == "process":
                    pid = int(r.path_or_name)
                    os.kill(pid, 9)
                    actions.append(f"Killed process: {pid}")
            except Exception as exc:
                log.warning("Cleanup failed for %s %s: %s", r.kind, r.path_or_name, exc)

        self._resources = [r for r in self._resources if r.keep and not force]
        return actions

    def summary(self) -> dict:
        return {
            "total_tracked": len(self._resources),
            "files": len([r for r in self._resources if r.kind == "file"]),
            "directories": len([r for r in self._resources if r.kind == "directory"]),
            "packages": len([r for r in self._resources if r.kind in ("pip_package", "npm_package")]),
            "processes": len([r for r in self._resources if r.kind == "process"]),
        }


# Global singleton
_tracker: Optional[ResourceTracker] = None


def get_tracker() -> ResourceTracker:
    global _tracker
    if _tracker is None:
        _tracker = ResourceTracker()
    return _tracker
