"""Research tools — ArXiv, HuggingFace Hub, RSS feeds."""

from __future__ import annotations

from typing import Optional


def search_arxiv(query: str, max_results: int = 10) -> str:
    """Search ArXiv for research papers."""
    try:
        import arxiv

        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        results = []
        for paper in search.results():
            authors = ", ".join(a.name for a in paper.authors[:3])
            if len(paper.authors) > 3:
                authors += f" et al. ({len(paper.authors)} authors)"
            results.append(
                f"**{paper.title}**\n"
                f"  Authors: {authors}\n"
                f"  Published: {paper.published.strftime('%Y-%m-%d')}\n"
                f"  URL: {paper.entry_id}\n"
                f"  Abstract: {paper.summary[:300]}...\n"
            )
        if results:
            return f"ArXiv results for '{query}':\n\n" + "\n---\n".join(results)
        return f"No ArXiv papers found for '{query}'"
    except ImportError:
        return "ERROR: arxiv package not installed. Run: pip install arxiv"
    except Exception as e:
        return f"ERROR searching ArXiv: {e}"


def search_arxiv_recent(topic: str, days: int = 7, max_results: int = 10) -> str:
    """Search for the most recent ArXiv papers on a topic."""
    try:
        import arxiv
        from datetime import datetime, timedelta

        search = arxiv.Search(
            query=topic,
            max_results=max_results * 3,  # Fetch more, then filter by date
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )
        cutoff = datetime.now(tz=None) - timedelta(days=days)
        results = []
        for paper in search.results():
            pub_date = paper.published.replace(tzinfo=None)
            if pub_date < cutoff:
                continue
            authors = ", ".join(a.name for a in paper.authors[:3])
            results.append(
                f"**{paper.title}**\n"
                f"  Authors: {authors}\n"
                f"  Published: {paper.published.strftime('%Y-%m-%d')}\n"
                f"  URL: {paper.entry_id}\n"
                f"  Abstract: {paper.summary[:250]}...\n"
            )
            if len(results) >= max_results:
                break

        if results:
            return f"Recent ArXiv papers on '{topic}' (last {days} days):\n\n" + "\n---\n".join(results)
        return f"No recent papers found for '{topic}' in the last {days} days."
    except Exception as e:
        return f"ERROR: {e}"


def search_huggingface(query: str, kind: str = "model", max_results: int = 10) -> str:
    """Search HuggingFace Hub for models, datasets, or spaces."""
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        results = []

        if kind == "model":
            models = api.list_models(search=query, limit=max_results, sort="downloads")
            for m in models:
                results.append(
                    f"**{m.modelId}**  (⬇ {m.downloads:,})\n"
                    f"  Tags: {', '.join(m.tags[:5]) if m.tags else 'none'}\n"
                    f"  URL: https://huggingface.co/{m.modelId}\n"
                )
        elif kind == "dataset":
            datasets = api.list_datasets(search=query, limit=max_results, sort="downloads")
            for d in datasets:
                results.append(
                    f"**{d.id}**  (⬇ {d.downloads:,})\n"
                    f"  URL: https://huggingface.co/datasets/{d.id}\n"
                )
        elif kind == "space":
            spaces = api.list_spaces(search=query, limit=max_results, sort="likes")
            for s in spaces:
                results.append(
                    f"**{s.id}**  (❤ {s.likes})\n"
                    f"  URL: https://huggingface.co/spaces/{s.id}\n"
                )

        if results:
            return f"HuggingFace {kind}s for '{query}':\n\n" + "\n".join(results)
        return f"No {kind}s found for '{query}'"
    except ImportError:
        return "ERROR: huggingface-hub not installed. Run: pip install huggingface-hub"
    except Exception as e:
        return f"ERROR searching HuggingFace: {e}"


def read_rss_feed(url: str, max_entries: int = 10) -> str:
    """Read an RSS/Atom feed and return entries."""
    try:
        import feedparser

        feed = feedparser.parse(url)
        if feed.bozo and not feed.entries:
            return f"ERROR: Could not parse feed at {url}"

        entries = []
        for entry in feed.entries[:max_entries]:
            title = entry.get("title", "Untitled")
            link = entry.get("link", "")
            published = entry.get("published", "")
            summary = entry.get("summary", "")[:200]
            entries.append(f"**{title}**\n  {link}\n  {published}\n  {summary}\n")

        feed_title = feed.feed.get("title", url)
        if entries:
            return f"Feed: {feed_title}\n\n" + "\n---\n".join(entries)
        return f"No entries in feed: {url}"
    except ImportError:
        return "ERROR: feedparser not installed. Run: pip install feedparser"
    except Exception as e:
        return f"ERROR reading feed: {e}"


RESEARCH_TOOLS = [
    {
        "name": "search_arxiv",
        "description": "Search ArXiv for research papers by topic/keyword.",
        "parameters": {
            "query": {"type": "string", "description": "Search query"},
            "max_results": {"type": "integer", "description": "Max results", "default": 10},
        },
        "function": search_arxiv,
    },
    {
        "name": "search_arxiv_recent",
        "description": "Find the most recent ArXiv papers on a topic (last N days).",
        "parameters": {
            "query": {"type": "string", "description": "Topic to search"},
            "days": {"type": "integer", "description": "Look back N days", "default": 7},
        },
        "function": search_arxiv_recent,
    },
    {
        "name": "search_huggingface",
        "description": "Search HuggingFace Hub for models, datasets, or spaces.",
        "parameters": {
            "query": {"type": "string", "description": "Search query"},
            "kind": {"type": "string", "description": "'model', 'dataset', or 'space'", "default": "model"},
        },
        "function": search_huggingface,
    },
    {
        "name": "read_rss_feed",
        "description": "Read an RSS/Atom feed (for news, blogs, etc.).",
        "parameters": {
            "url": {"type": "string", "description": "Feed URL"},
            "max_entries": {"type": "integer", "description": "Max entries", "default": 10},
        },
        "function": read_rss_feed,
    },
]
