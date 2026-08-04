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

    Scavio returns different shapes per endpoint (Google v2 puts organic
    results at the top level; Amazon/Reddit/YouTube nest them under ``data``),
    so probe the known keys in priority order.
    """
    if not isinstance(resp, dict):
        return []
    # Top-level result lists (Google v2 SERP uses organic_results;
    # News uses news_results).
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
            "comments",
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
            # Reddit post bodies come back as `text` (not Reddit's own
            # `selftext`); Google v2 uses `snippet`.
            body = _flatten_text(
                record.get("description")
                or record.get("snippet")
                or record.get("text")
                or record.get("content")
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
        "youtube_video",
        "youtube_transcript",
        "youtube_comments",
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
        cursor: Optional[str] = None,
        max_results: Optional[int] = 10,
    ) -> List[Document]:
        """Search Reddit posts for community discussion and sentiment.

        Results are relevance-ordered and cannot be sorted or filtered by type.

        Args:
            query: The Reddit search query.
            cursor: Pagination cursor: the next_cursor of a previous response.
            max_results: Maximum number of results to return.

        """
        params: Dict[str, Any] = {"cursor": cursor}
        resp = self.client.reddit.search(
            query, **{k: v for k, v in params.items() if v is not None}
        )
        docs = _to_documents(resp, "reddit")
        return docs[:max_results] if max_results else docs

    def youtube_search(
        self,
        query: str,
        upload_date: Optional[str] = None,
        type: Optional[str] = None,
        duration: Optional[str] = None,
        sort_by: Optional[str] = None,
        max_results: Optional[int] = 10,
    ) -> List[Document]:
        """Search YouTube for videos, channels, or playlists.

        Args:
            query: The video search query.
            upload_date: Upload date filter: last_hour, today, this_week, this_month, this_year.
            type: Result type: video, channel, playlist, movie.
            duration: Duration filter: short, medium, long.
            sort_by: Sort order: relevance, date, view_count, rating.
            max_results: Maximum number of results to return.

        """
        params: Dict[str, Any] = {
            "upload_date": upload_date,
            "type": type,
            "duration": duration,
            "sort_by": sort_by,
        }
        resp = self.client.youtube.search(
            query, **{k: v for k, v in params.items() if v is not None}
        )
        docs = _to_documents(resp, "youtube")
        return docs[:max_results] if max_results else docs

    def youtube_video(self, video_id: str) -> List[Document]:
        """Fetch full details for a YouTube video (title, author, description, view count, chapters).

        Args:
            video_id: YouTube video id or a full watch URL.

        """
        resp = self.client.youtube.video(video_id)
        return _to_documents(resp, "youtube_video")

    def youtube_transcript(
        self,
        video_id: str,
        language: Optional[str] = None,
        format: Optional[str] = None,
    ) -> List[Document]:
        """Fetch a YouTube video's transcript as a Document ready for a RAG pipeline.

        Args:
            video_id: YouTube video id or a full watch URL.
            language: Transcript language code (default 'en').
            format: 'text' for a plain transcript, 'srt' for timed subtitles.

        """
        params: Dict[str, Any] = {"language": language, "format": format}
        resp = self.client.youtube.transcript(
            video_id, **{k: v for k, v in params.items() if v is not None}
        )
        payload = resp.get("data") if isinstance(resp.get("data"), dict) else resp
        content = _flatten_text(payload.get("content")) if isinstance(payload, dict) else ""
        if content:
            extra = {"source": "youtube_transcript"}
            vid = payload.get("video_id") if isinstance(payload, dict) else None
            if vid:
                extra["url"] = f"https://www.youtube.com/watch?v={vid}"
            return [Document(text=content, extra_info=extra)]
        return _to_documents(resp, "youtube_transcript")

    def youtube_comments(
        self,
        video_id: str,
        max_results: Optional[int] = 10,
    ) -> List[Document]:
        """List comments on a YouTube video as Documents.

        Args:
            video_id: YouTube video id or a full watch URL.
            max_results: Maximum number of comments to return.

        """
        resp = self.client.youtube.comments(video_id)
        docs = _to_documents(resp, "youtube_comments")
        return docs[:max_results] if max_results else docs

    def amazon_search(
        self,
        query: str,
        country: Optional[str] = None,
        max_results: Optional[int] = 10,
    ) -> List[Document]:
        """Search Amazon for products matching a query.

        Results are unsorted and cannot be filtered.

        Args:
            query: The product search query.
            country: Marketplace country code (ISO 3166-1 alpha-2), not a
                domain: 'us' (default), 'gb' (the UK is gb, not uk), 'de', 'jp'.
                An unknown code falls back to 'us'.
            max_results: Maximum number of results to return.

        """
        params: Dict[str, Any] = {"country": country}
        resp = self.client.amazon.search(
            query, **{k: v for k, v in params.items() if v is not None}
        )
        docs = _to_documents(resp, "amazon")
        return docs[:max_results] if max_results else docs
