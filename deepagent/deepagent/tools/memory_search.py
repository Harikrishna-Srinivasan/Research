"""Memory search tools — semantic search over long-term memory."""

from __future__ import annotations

from typing import Optional


def search_memory(query: str, n_results: int = 5, category: str | None = None) -> str:
    """Search long-term memory semantically for relevant information.
    
    This tool allows the agent to retrieve stored knowledge from previous tasks,
    research findings, and breakthroughs.
    
    Args:
        query: The search query describing what information you're looking for.
        n_results: Maximum number of results to return (default: 5).
        category: Optional category filter (e.g., 'research', 'task_result', 'completed_task').
    
    Returns:
        Formatted search results or a message if no memories are found.
    """
    try:
        from deepagent.memory.long_term import LongTermMemory
        from deepagent.config import load_config
        
        config = load_config()
        memory = LongTermMemory(
            db_path=config.memory.long_term_db_path,
            embedding_model=config.memory.embedding_model,
        )
        
        if memory.count == 0:
            return "No memories stored yet. Complete some tasks first to build up knowledge."
        
        entries = memory.search(query, n_results=n_results, category=category)
        
        if not entries:
            return f"No relevant memories found for '{query}'."
        
        lines = [f"🧠 Found {len(entries)} relevant memories for '{query}':\n"]
        for i, e in enumerate(entries, 1):
            relevance = f" (relevance: {e['relevance']:.2f})" if e['relevance'] is not None else ""
            cat = e["metadata"].get("category", "general")
            content = e["content"][:400]
            if len(e["content"]) > 400:
                content += "..."
            lines.append(f"--- Memory {i} [{cat}]{relevance} ---\n{content}\n")
        
        return "\n".join(lines)
    
    except ImportError as e:
        return f"ERROR: Required package not installed: {e}"
    except Exception as e:
        return f"ERROR searching memory: {type(e).__name__}: {e}"


def store_memory(content: str, category: str = "general", metadata: dict | None = None) -> str:
    """Store important information in long-term memory for future retrieval.
    
    Use this tool to save key findings, insights, or results that might be useful
    for future tasks.
    
    Args:
        content: The information to store (be concise but complete).
        category: Category label for organization (e.g., 'research', 'code_snippet', 'insight').
        metadata: Optional additional metadata as a JSON-like dict.
    
    Returns:
        Confirmation message with the stored memory ID.
    """
    try:
        from deepagent.memory.long_term import LongTermMemory
        from deepagent.config import load_config
        
        config = load_config()
        memory = LongTermMemory(
            db_path=config.memory.long_term_db_path,
            embedding_model=config.memory.embedding_model,
        )
        
        result = memory.store(content, metadata=metadata, category=category)
        return f"✅ {result}"
    
    except ImportError as e:
        return f"ERROR: Required package not installed: {e}"
    except Exception as e:
        return f"ERROR storing memory: {type(e).__name__}: {e}"


def list_memories(category: str | None = None, limit: int = 10) -> str:
    """List recent memories, optionally filtered by category.
    
    Args:
        category: Optional category filter. If None, lists all recent memories.
        limit: Maximum number of memories to list (default: 10).
    
    Returns:
        Formatted list of recent memories.
    """
    try:
        from deepagent.memory.long_term import LongTermMemory
        from deepagent.config import load_config
        import chromadb
        
        config = load_config()
        memory = LongTermMemory(
            db_path=config.memory.long_term_db_path,
            embedding_model=config.memory.embedding_model,
        )
        
        if memory.count == 0:
            return "No memories stored yet."
        
        # Get raw data from ChromaDB
        memory._ensure_initialized()
        if memory._collection is None:
            return "Memory system not available."
        
        # Fetch recent entries
        where = {"category": category} if category else None
        results = memory._collection.get(
            where=where,
            limit=limit,
            include=["documents", "metadatas"],
        )
        
        if not results["documents"]:
            return f"No memories found{' in category ' + category if category else ''}."
        
        lines = [f"📚 Recent memories{' in category [' + category + ']' if category else ''}:\n"]
        for i, (doc, meta) in enumerate(zip(results["documents"], results["metadatas"]), 1):
            cat = meta.get("category", "general")
            timestamp = meta.get("timestamp", "")
            if timestamp:
                from datetime import datetime
                timestamp = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")
            
            preview = doc[:150] + "..." if len(doc) > 150 else doc
            lines.append(f"{i}. [{cat}] {timestamp}\n   {preview}\n")
        
        total = memory.count
        lines.append(f"\nTotal memories: {total}")
        return "\n".join(lines)
    
    except ImportError as e:
        return f"ERROR: Required package not installed: {e}"
    except Exception as e:
        return f"ERROR listing memories: {type(e).__name__}: {e}"


def clear_memories(category: str | None = None, confirm: bool = False) -> str:
    """Clear memories from long-term storage.
    
    ⚠️ WARNING: This operation is irreversible!
    
    Args:
        category: If provided, only clear memories in this category.
                  If None, clears ALL memories.
        confirm: Must be True to proceed with deletion.
    
    Returns:
        Confirmation of deletion or warning if not confirmed.
    """
    try:
        from deepagent.memory.long_term import LongTermMemory
        from deepagent.config import load_config
        
        if not confirm:
            return "⚠️ CONFIRMATION REQUIRED: Set confirm=True to proceed with deletion."
        
        config = load_config()
        memory = LongTermMemory(
            db_path=config.memory.long_term_db_path,
            embedding_model=config.memory.embedding_model,
        )
        
        memory._ensure_initialized()
        if memory._collection is None:
            return "Memory system not available."
        
        initial_count = memory.count
        
        if category:
            # Delete only specific category
            results = memory._collection.get(where={"category": category}, include=["ids"])
            ids_to_delete = results["ids"][0] if results["ids"] else []
            if ids_to_delete:
                memory._collection.delete(ids=ids_to_delete)
                deleted = len(ids_to_delete)
            else:
                deleted = 0
        else:
            # Delete all
            memory._collection.delete(where={})
            deleted = initial_count
        
        return f"✅ Deleted {deleted} memories. Remaining: {memory.count}"
    
    except ImportError as e:
        return f"ERROR: Required package not installed: {e}"
    except Exception as e:
        return f"ERROR clearing memories: {type(e).__name__}: {e}"


MEMORY_TOOLS = [
    {
        "name": "search_memory",
        "description": "Search long-term memory semantically for relevant information from past tasks and research.",
        "parameters": {
            "query": {"type": "string", "description": "Search query describing the information needed"},
            "n_results": {"type": "integer", "description": "Max results to return", "default": 5},
            "category": {"type": "string", "description": "Optional category filter (e.g., 'research', 'task_result')", "default": None},
        },
        "function": search_memory,
    },
    {
        "name": "store_memory",
        "description": "Store important information in long-term memory for future retrieval.",
        "parameters": {
            "content": {"type": "string", "description": "The information to store"},
            "category": {"type": "string", "description": "Category label for organization", "default": "general"},
        },
        "function": store_memory,
    },
    {
        "name": "list_memories",
        "description": "List recent memories, optionally filtered by category.",
        "parameters": {
            "category": {"type": "string", "description": "Optional category filter", "default": None},
            "limit": {"type": "integer", "description": "Max memories to list", "default": 10},
        },
        "function": list_memories,
    },
    {
        "name": "clear_memories",
        "description": "⚠️ Clear memories from storage. Requires confirm=True. Use with caution!",
        "parameters": {
            "category": {"type": "string", "description": "Category to clear (None for all)", "default": None},
            "confirm": {"type": "boolean", "description": "Must be true to proceed", "default": False},
        },
        "function": clear_memories,
    },
]
