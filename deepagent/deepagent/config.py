"""Configuration loader for DeepAgent."""

import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Any


@dataclass
class LLMConfig:
    model_id: str = "Qwen/Qwen3.5-9B"
    quantization: str = "none"  # "4bit", "8bit", "none"
    context_length: int = 8192
    temperature: float = 0.6
    max_new_tokens: int = 2048
    device: str = "auto"


@dataclass
class AgentConfig:
    max_iterations: int = 50
    max_time_minutes: int = 180
    enable_critic: bool = True
    enable_breakthrough_detection: bool = True
    enable_tool_creation: bool = True


@dataclass
class MemoryConfig:
    short_term_tokens: int = 4096
    long_term_db_path: str = "./memory_store"
    embedding_model: str = "all-MiniLM-L6-v2"


@dataclass
class ToolsConfig:
    workspace_dirs: list = field(default_factory=lambda: ["./workspace"])
    allowed_commands: list = field(
        default_factory=lambda: ["python", "pip", "node", "npm", "npx", "git", "curl", "wget"]
    )
    blocked_patterns: list = field(
        default_factory=lambda: ["rm -rf /", "format c:", "del /s /q c:"]
    )
    web_search_enabled: bool = True
    max_command_timeout: int = 300
    custom_tools_dir: str = "./custom_tools"


@dataclass
class SafetyConfig:
    confirm_destructive: bool = True
    auto_cleanup: bool = True
    max_file_size_mb: int = 100


@dataclass
class DeepAgentConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)

    def update(self, attr: str, val: Any):
        if val is not None:
            setattr(self, attr, val)


def load_config(config_path: Optional[str] = None) -> DeepAgentConfig:
    """Load configuration from YAML file, falling back to defaults."""
    if config_path is None:
        # Search upward from CWD for config.yaml
        search = Path.cwd()
        for parent in [search] + list(search.parents):
            candidate = parent / "config.yaml"
            if candidate.exists():
                config_path = str(candidate)
                break

    config = DeepAgentConfig()

    if config_path and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        for attr in ["llm", "agent", "memory", "tools", "safety"]:
            config.update(attr, raw.get(attr))

    return config
