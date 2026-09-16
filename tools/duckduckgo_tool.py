"""
tools/duckduckgo_tool.py
----------------------------
Adds multi_search(), matching the reference project's pattern of
running several targeted queries per agent (e.g. 5 competitor
queries) instead of one broad query, then combining results.
"""

from ddgs import DDGS
from concurrent.futures import ThreadPoolExecutor


class SearchResult:
    """Lightweight wrapper matching the reference's to_context_string() pattern."""

    def __init__(self, query: str, hits: list):
        self.query = query
        self.hits = hits

    def to_context_string(self) -> str:
        lines = [f"Query: {self.query}"]
        for h in self.hits:
            lines.append(f"- {h.get('title', '')}: {h.get('snippet', '')} ({h.get('url', '')})")
        return "\n".join(lines)


class DuckDuckGoTool:
    def search(self, query: str, max_results: int = 10) -> list:
        results = []
        try:
            with DDGS() as ddgs:
                response = ddgs.text(query, max_results=max_results)
                for item in response:
                    results.append({
                        "title": item.get("title"),
                        "url": item.get("href"),
                        "snippet": item.get("body"),
                    })
        except Exception as e:
            print(f"DuckDuckGo search error: {e}")
        return results

    def multi_search(self, queries: list, max_results_per_query: int = 7) -> list:
        """
        Runs multiple targeted queries (matches the reference project's
        pattern) and returns a list of SearchResult objects, one per
        query, each with its own hits - not merged, so the LLM can see
        which results came from which angle of research.
        """
        # Each query creates its own short-lived DDGS client in search(), so
        # they are independent and can run concurrently. Preserve the input
        # order so downstream context remains deterministic.
        with ThreadPoolExecutor(max_workers=min(4, len(queries) or 1)) as executor:
            hits_by_query = list(
                executor.map(lambda query: self.search(query, max_results_per_query), queries)
            )
        return [SearchResult(query, hits) for query, hits in zip(queries, hits_by_query)]
