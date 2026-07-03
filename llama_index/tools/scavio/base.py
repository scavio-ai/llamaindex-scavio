"""Scavio tool spec for LlamaIndex.

Wraps the Scavio real-time search API (Google, Google News, Reddit, YouTube,
Amazon) and exposes each provider as a LlamaIndex agent tool that returns
``Document`` objects ready to drop into a RAG pipeline or an agent.

    from llama_index.tools.scavio import ScavioToolSpec

    tools = ScavioToolSpec(api_key="sk_...").to_tool_list()

The API key falls back to the ``SCAVIO_API_KEY`` environment variable.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from llama_index.core.schema import Document
from llama_index.core.tools.tool_spec.base import BaseToolSpec


def _records(resp: Dict[str, Any]) -> List[Any]:
    """Pull the most relevant list of items out of a Scavio response.

    Scavio returns different shapes per endpoint (Google SERP puts organic
    results at the top level; Amazon/Reddit/YouTube nest them under ``data``),
    so probe the known keys in priority order.
    """
    if not isinstance(resp, dict):
        return []
    # Top-level result lists (Google SERP full mode uses organic_results;
    # light mode uses results; News uses news_results).
    for key in ("results", "organic_results", "news_results"):
        if isinstance(resp.get(key), list) and resp[key]:
            return resp[key]
    data = resp.get("data")
    if isinstance(data, dict):
        for key in (
            "products",
            "posts",
            "results",
            "organic_results",
            "videos",
            "news_results",
            "items",
        ):
            if isinstance(data.get(key), list) and data[key]:
                return data[key]
    if isinstance(data, list):
        return data
    for key in ("posts", "videos", "items"):
        if isinstance(resp.get(key), list) and resp[key]:
            return resp[key]
    return []


def _flatten_text(value: Any) -> str:
    """Flatten a field to a string.

    Handles YouTube-style rich text (``{"runs": [{"text": ...}]}`` /
    ``{"simpleText": ...}``) that Scavio passes through from the source.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        runs = value.get("runs")
        if isinstance(runs, list):
            return "".join(r.get("text", "") for r in runs if isinstance(r, dict))
        for key in ("text", "simpleText", "content"):
            if isinstance(value.get(key), str):
                return value[key]
    return "" if value is None else str(value)


def _url_of(record: Dict[str, Any]) -> Optional[str]:
    """Derive a usable URL from a result record across provider shapes."""
    for key in ("url", "link", "permalink"):
        value = record.get(key)
        if isinstance(value, str) and value:
            if value.startswith("/"):
                asin = record.get("asin")
                return f"https://www.amazon.com/dp/{asin}" if asin else f"https://www.amazon.com{value}"
            return value
    video_id = record.get("videoId") or record.get("video_id")
    if video_id:
        return f"https://www.youtube.com/watch?v={video_id}"
    return None


def _to_documents(resp: Dict[str, Any], source: str) -> List[Document]:
    """Convert a raw Scavio response into LlamaIndex ``Document`` objects."""
    records = _records(resp)
    if not records:
        return [
            Document(
                text=json.dumps(resp, ensure_ascii=False)[:8000],
                extra_info={"source": source},
            )
        ]
    docs: List[Document] = []
    for record in records:
        if isinstance(record, dict):
            title = _flatten_text(record.get("title") or record.get("name") or "")
            body = _flatten_text(
                record.get("description")
                or record.get("snippet")
                or record.get("text")
                or record.get("content")
                or record.get("selftext")
                or ""
            )
            text = f"{title}\n{body}".strip() or json.dumps(record, ensure_ascii=False)[:2000]
            url = _url_of(record)
            extra = {"source": source}
            if url:
                extra["url"] = url
            docs.append(Document(text=text, extra_info=extra))
        else:
            docs.append(Document(text=str(record), extra_info={"source": source}))
    return docs


class ScavioToolSpec(BaseToolSpec):
    """Scavio tool spec: real-time search across Google, Reddit, YouTube, Amazon."""

    spec_functions = [
        "search",
        "news",
        "reddit_search",
        "youtube_search",
        "amazon_search",
    ]

    def __init__(self, api_key: Optional[str] = None) -> None:
        """Initialize with a Scavio API key (or the SCAVIO_API_KEY env var)."""
        try:
            from scavio import ScavioClient
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "`scavio` not installed. Please install using `pip install scavio`"
            ) from exc

        self.client = ScavioClient(api_key=api_key or os.getenv("SCAVIO_API_KEY"))

    def search(
        self,
        query: str,
        gl: Optional[str] = None,
        hl: Optional[str] = None,
        max_results: Optional[int] = 10,
    ) -> List[Document]:
        """Search Google for real-time web results.

        Args:
            query: The search query.
            gl: Two-letter country code of the search, e.g. 'us'.
            hl: Two-letter UI language code, e.g. 'en'.
            max_results: Maximum number of results to return.

        """
        params: Dict[str, Any] = {"gl": gl, "hl": hl}
        resp = self.client.google.search(
            query, **{k: v for k, v in params.items() if v is not None}
        )
        docs = _to_documents(resp, "google")
        return docs[:max_results] if max_results else docs

    def news(
        self,
        query: str,
        gl: Optional[str] = None,
        hl: Optional[str] = None,
        max_results: Optional[int] = 10,
    ) -> List[Document]:
        """Search Google News for recent news articles on a topic.

        Args:
            query: The news search query.
            gl: Two-letter country code, e.g. 'us'.
            hl: Two-letter language code, e.g. 'en'.
            max_results: Maximum number of results to return.

        """
        params: Dict[str, Any] = {"query": query, "gl": gl, "hl": hl}
        resp = self.client.google.news(**{k: v for k, v in params.items() if v is not None})
        docs = _to_documents(resp, "google_news")
        return docs[:max_results] if max_results else docs

    def reddit_search(
        self,
        query: str,
        sort: Optional[str] = None,
        max_results: Optional[int] = 10,
    ) -> List[Document]:
        """Search Reddit posts for community discussion and sentiment.

        Args:
            query: The Reddit search query.
            sort: Sort order: new, relevance, hot, top, or comments.
            max_results: Maximum number of results to return.

        """
        params: Dict[str, Any] = {"sort": sort}
        resp = self.client.reddit.search(
            query, **{k: v for k, v in params.items() if v is not None}
        )
        docs = _to_documents(resp, "reddit")
        return docs[:max_results] if max_results else docs

    def youtube_search(
        self,
        query: str,
        max_results: Optional[int] = 10,
    ) -> List[Document]:
        """Search YouTube for videos, channels, or playlists.

        Args:
            query: The video search query.
            max_results: Maximum number of results to return.

        """
        resp = self.client.youtube.search(query)
        docs = _to_documents(resp, "youtube")
        return docs[:max_results] if max_results else docs

    def amazon_search(
        self,
        query: str,
        domain: Optional[str] = None,
        max_results: Optional[int] = 10,
    ) -> List[Document]:
        """Search Amazon for products matching a query.

        Args:
            query: The product search query.
            domain: Amazon domain, e.g. 'amazon.com'.
            max_results: Maximum number of results to return.

        """
        params: Dict[str, Any] = {"domain": domain}
        resp = self.client.amazon.search(
            query, **{k: v for k, v in params.items() if v is not None}
        )
        docs = _to_documents(resp, "amazon")
        return docs[:max_results] if max_results else docs
