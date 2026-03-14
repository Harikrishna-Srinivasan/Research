"""Short-term memory — sliding window over conversation with summarization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Message:
    role: str  # "system", "user", "assistant", "observation"
    content: str
    priority: int = 0  # higher = kept longer when context is compressed
    tokens: int = 0


class ShortTermMemory:
    """Manages the conversation context window with intelligent truncation."""

    def __init__(self, max_tokens: int = 4096, token_counter=None):
        self.max_tokens = max_tokens
        self._messages: list[Message] = []
        self._token_counter = token_counter or (lambda x: len(x) // 4)  # rough estimate

    def add(self, role: str, content: str, priority: int = 0) -> None:
        tokens = self._token_counter(content)
        self._messages.append(Message(role, content, priority, tokens))
        self._maybe_compress()

    def get_messages(self) -> list[dict]:
        """Return messages as dicts for the LLM."""
        return [{"role": m.role, "content": m.content} for m in self._messages]

    def get_total_tokens(self) -> int:
        return sum(m.tokens for m in self._messages)

    def clear(self) -> None:
        self._messages.clear()

    def _maybe_compress(self) -> None:
        """If we exceed max_tokens, remove low-priority older messages."""
        total = self.get_total_tokens()
        if total <= self.max_tokens:
            return

        # Always keep system messages and the last few messages
        keep_last = 4
        if len(self._messages) <= keep_last + 1:
            return

        # Sort middle messages by priority (low priority removed first)
        system_msgs = [m for m in self._messages if m.role == "system"]
        tail = self._messages[-keep_last:]
        middle = self._messages[len(system_msgs) : -keep_last]

        # Remove lowest priority messages from middle until under budget
        middle.sort(key=lambda m: m.priority, reverse=True)
        budget = self.max_tokens - sum(m.tokens for m in system_msgs) - sum(m.tokens for m in tail)

        kept_middle = []
        used = 0
        for m in middle:
            if used + m.tokens <= budget:
                kept_middle.append(m)
                used += m.tokens

        # If we had to drop messages, add a summary marker
        dropped_count = len(middle) - len(kept_middle)
        if dropped_count > 0:
            summary_msg = Message(
                role="system",
                content=f"[{dropped_count} older messages compressed to fit context window]",
                priority=0,
                tokens=15,
            )
            self._messages = system_msgs + [summary_msg] + kept_middle + tail
        else:
            self._messages = system_msgs + kept_middle + tail

    @property
    def message_count(self) -> int:
        return len(self._messages)
