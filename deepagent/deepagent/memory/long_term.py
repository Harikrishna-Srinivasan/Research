"""Long-term memory — persistent vector store using ChromaDB."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

from deepagent.utils.logger import get_logger

log = get_logger(__name__)


class LongTermMemory:
    """Persistent semantic memory backed by ChromaDB.

    Stores research findings, code snippets, task results, and
    breakthroughs across sessions.
    """

    def __init__(self, db_path: str = "./memory_store", embedding_model: str = "all-MiniLM-L6-v2"):
        self.db_path = str(Path(db_path).resolve())
        self._client = None
        self._collection = None
        self._embedding_model = embedding_model

    def _ensure_initialized(self) -> None:
        if self._client is not None:
            return
        try:
            import chromadb

            os.makedirs(self.db_path, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self.db_path)
            self._collection = self._client.get_or_create_collection(
                name="deepagent_memory",
                metadata={"hnsw:space": "cosine"},
            )
            log.info("Long-term memory initialized at %s (%d entries)", self.db_path, self._collection.count())
        except ImportError:
            log.warning("ChromaDB not available — long-term memory disabled.")

    def store(
        self,
        content: str,
        metadata: dict | None = None,
        category: str = "general",
    ) -> str:
        """Store a piece of information in long-term memory."""
        self._ensure_initialized()
        if self._collection is None:
            return "WARN: Long-term memory not available."

        doc_id = f"{category}_{int(time.time() * 1000)}"
        meta = {
            "category": category,
            "timestamp": time.time(),
            **(metadata or {}),
        }
        # Ensure all metadata values are primitives
        clean_meta = {k: str(v) if not isinstance(v, (str, int, float, bool)) else v for k, v in meta.items()}

        self._collection.add(
            documents=[content],
            metadatas=[clean_meta],
            ids=[doc_id],
        )
        return f"Stored in memory: {doc_id}"

    def search(self, query: str, n_results: int = 5, category: str | None = None) -> list[dict]:
        """Search memory semantically."""
        self._ensure_initialized()
        if self._collection is None or self._collection.count() == 0:
            return []

        where = {"category": category} if category else None
        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=min(n_results, self._collection.count()),
                where=where,
            )
        except Exception as e:
            log.warning("Memory search failed: %s", e)
            return []

        entries = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else None
                entries.append({
                    "content": doc,
                    "metadata": meta,
                    "relevance": 1 - distance if distance is not None else None,
                })
        return entries

    def search_formatted(self, query: str, n_results: int = 5) -> str:
        """Search and format results as a readable string."""
        entries = self.search(query, n_results)
        if not entries:
            return f"No memories found for '{query}'."

        lines = [f"Found {len(entries)} relevant memories for '{query}':\n"]
        for i, e in enumerate(entries, 1):
            relevance = f" (relevance: {e['relevance']:.2f})" if e['relevance'] is not None else ""
            cat = e["metadata"].get("category", "general")
            lines.append(f"--- Memory {i} [{cat}]{relevance} ---\n{e['content'][:500]}\n")
        return "\n".join(lines)

    @property
    def count(self) -> int:
        self._ensure_initialized()
        if self._collection is None:
            return 0
        return self._collection.count()
