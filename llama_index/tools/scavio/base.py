"""Scavio tool spec for LlamaIndex.

Wraps the Scavio real-time search API (Google, Google News, Reddit, YouTube,
Amazon) plus ``extract``, which reads any URL, and exposes each one as a
LlamaIndex agent tool that returns ``Document`` objects ready to drop into a
RAG pipeline or an agent. ``enable_crawl=True`` adds the multi-page members of
the Extract family: map a site's URLs, crawl a site, and batch-extract a list
of URLs.

    from llama_index.tools.scavio import ScavioToolSpec

    tools = ScavioToolSpec(api_key="sk_...").to_tool_list()
    crawl_tools = ScavioToolSpec(enable_crawl=True).to_tool_list()

The API key falls back to the ``SCAVIO_API_KEY`` environment variable.
"""

from __future__ import annotations

import json
import os
import time
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


def _extract_document(resp: Dict[str, Any], requested_url: str) -> List[Document]:
    """Build a single ``Document`` from an ``/api/v1/extract`` response.

    Extract is the one endpoint here that returns page CONTENT rather than a
    list of records, so ``_to_documents`` is the wrong shape for it: there is no
    result array to iterate, and its JSON fallback would truncate the page to
    8000 characters. ``youtube_transcript`` special-cases its content string the
    same way. Falls back to ``_to_documents`` only if the content field is
    missing, so an unexpected shape still reaches the caller instead of
    vanishing.
    """
    payload = resp.get("data") if isinstance(resp.get("data"), dict) else resp
    content = payload.get("content") if isinstance(payload, dict) else None
    if not isinstance(content, str) or not content:
        return _to_documents(resp, "extract")
    extra: Dict[str, Any] = {"source": "extract", "url": payload.get("url") or requested_url}
    for key in ("format", "mode", "content_length"):
        value = payload.get(key)
        if value is not None:
            extra[key] = value
    return [Document(text=content, extra_info=extra)]


# Crawl and batch run as background jobs. The tools start the job, then poll
# its (free) status until it finishes or the wait budget runs out.
JOB_TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})
JOB_POLL_INTERVAL = 3.0
JOB_MAX_WAIT = 300
JOB_DEFAULT_WAIT = 45
JOB_DEFAULT_LIMIT = 10
_JOB_SUMMARY_KEYS = (
    "id",
    "type",
    "status",
    "stopped_reason",
    "error",
    "url",
    "pages",
    "credits_used",
    "credits_per_page",
    "expires_at",
    "next_cursor",
    "urls_count",
    "invalid_urls",
)


def _clean(params: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in params.items() if v is not None}


def _map_document(resp: Dict[str, Any], requested_url: str) -> List[Document]:
    """One Document listing a site's URLs, one per line, with the counts as metadata."""
    data = resp.get("data") if isinstance(resp.get("data"), dict) else {}
    links = data.get("links") if isinstance(data.get("links"), list) else []
    urls = [link.get("url") for link in links if isinstance(link, dict) and link.get("url")]
    extra: Dict[str, Any] = {"source": "extract_map", "url": data.get("url") or requested_url}
    for key in ("total_found", "returned", "truncated", "sitemaps_read"):
        if data.get(key) is not None:
            extra[key] = data[key]
    text = "\n".join(urls) if urls else json.dumps(resp, ensure_ascii=False)[:8000]
    return [Document(text=text, extra_info=extra)]


def _job_documents(resp: Dict[str, Any], kind: str) -> List[Document]:
    """A job summary Document first, then one Document per returned page.

    The summary carries the job id, status, page counts, credits and
    ``next_cursor``, so an agent can read further pages (or a job that is still
    running) with the matching status tool.
    """
    data = resp.get("data") if isinstance(resp.get("data"), dict) else {}
    summary = {k: data[k] for k in _JOB_SUMMARY_KEYS if k in data}
    docs = [
        Document(
            text=json.dumps(summary, ensure_ascii=False),
            extra_info={"source": f"extract_{kind}_job", "job_id": data.get("id"), "status": data.get("status")},
        )
    ]
    results = data.get("results") if isinstance(data.get("results"), list) else []
    for result in results:
        if not isinstance(result, dict):
            continue
        extra: Dict[str, Any] = {"source": f"extract_{kind}", "job_id": data.get("id")}
        for key in ("url", "status", "depth", "status_code", "content_length", "truncated", "error", "skip_reason", "credits"):
            if result.get(key) is not None:
                extra[key] = result[key]
        content = result.get("content")
        text = content if isinstance(content, str) and content else json.dumps(extra, ensure_ascii=False)
        docs.append(Document(text=text, extra_info=extra))
    return docs


class ScavioToolSpec(BaseToolSpec):
    """Scavio tool spec: real-time search across Google, Reddit, YouTube, Amazon, plus URL extraction."""

    spec_functions = [
        "search",
        "news",
        "reddit_search",
        "youtube_search",
        "youtube_video",
        "youtube_transcript",
        "youtube_comments",
        "amazon_search",
        "extract",
        "heb_search",
        "heb_product",
        "heb_stores",
        "kroger_search",
        "kroger_product",
    ]

    # Opt-in: the multi-page Extract tools. They can bill per page, so they are
    # off unless enable_crawl=True.
    crawl_functions = [
        "map_website",
        "crawl_website",
        "crawl_status",
        "batch_extract",
        "batch_status",
    ]

    def __init__(self, api_key: Optional[str] = None, enable_crawl: bool = False) -> None:
        """Initialize with a Scavio API key (or the SCAVIO_API_KEY env var).

        Args:
            api_key: Scavio API key. Falls back to SCAVIO_API_KEY.
            enable_crawl: Also expose map_website, crawl_website, crawl_status,
                batch_extract and batch_status.

        """
        if enable_crawl:
            self.spec_functions = [*type(self).spec_functions, *self.crawl_functions]
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

    def extract(
        self,
        url: str,
        format: Optional[str] = None,
        mode: Optional[str] = None,
    ) -> List[Document]:
        """Read any web page and return it as one Document ready for a RAG pipeline.

        Use this after a search to read the page behind a result, or on any URL
        the user names. Returns a single Document holding the whole page, not a
        list of records.

        Args:
            url: Page to read (1-2048 characters). http(s) only; a bare host is
                upgraded to https, and loopback, private, link-local and
                metadata hosts are rejected.
            format: 'markdown' (default) is a readability extraction with the
                page chrome stripped, 'text' is that flattened to plain text,
                'html' is the raw page.
            mode: Fetch tier, and the parameter that sets the price: 'normal'
                (default) is the standard fetch and costs 1 credit, 'advanced'
                renders JavaScript-built pages and costs 1, 'ultra' reaches the
                hardest pages and costs 2. Start on 'normal'.

        """
        params: Dict[str, Any] = {"format": format, "mode": mode}
        resp = self.client.extract(url, **{k: v for k, v in params.items() if v is not None})
        return _extract_document(resp, url)

    # -- HEB and Kroger grocery tools ------------------------------------------

    def heb_search(
        self,
        query: str,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
        max_results: Optional[int] = 10,
    ) -> List[Document]:
        """Search HEB grocery products by keyword.

        Returns price (cents), sale status, unit price, size, UPC, stock,
        EBT/SNAP eligibility, aisle location and product URL. Costs 2 credits.

        Args:
            query: Search keywords (1-200 characters).
            page: Results page, 1-based (1-100).
            page_size: Results per page (1-60; server default 30).
            max_results: Maximum number of results to return.

        """
        params: Dict[str, Any] = {"page": page, "page_size": page_size}
        resp = self.client.heb.search(
            query, **{k: v for k, v in params.items() if v is not None}
        )
        docs = _to_documents(resp, "heb")
        return docs[:max_results] if max_results else docs

    def heb_product(
        self,
        product_url: Optional[str] = None,
        product_id: Optional[str] = None,
        slug: Optional[str] = None,
    ) -> List[Document]:
        """Fetch full HEB product detail.

        Nutrition facts, ingredients, description, images, similar products,
        coupons, EBT/SNAP eligibility and breadcrumbs. Pass product_url (from
        search) or slug + product_id. Costs 2 credits.

        Args:
            product_url: Full or relative product URL from heb.com. Preferred.
            product_id: HEB product id (numeric id at the end of a product URL). Requires slug.
            slug: Product slug (readable part of the URL). Required with product_id.

        """
        params: Dict[str, Any] = {"product_url": product_url, "product_id": product_id, "slug": slug}
        resp = self.client.heb.product(**{k: v for k, v in params.items() if v is not None})
        return _to_documents(resp, "heb_product")

    def heb_stores(
        self,
        zip: Optional[str] = None,
        city: Optional[str] = None,
    ) -> List[Document]:
        """Find HEB stores near a US ZIP code or city.

        Returns store number, address, phone, hours, pharmacy availability,
        fulfillment channels, distance and store URL. Costs 2 credits.

        Args:
            zip: 5-digit US ZIP code (e.g. '78701').
            city: City name (e.g. 'Austin'). Pass zip or city.

        """
        params: Dict[str, Any] = {"zip": zip, "city": city}
        resp = self.client.heb.stores(**{k: v for k, v in params.items() if v is not None})
        return _to_documents(resp, "heb_stores")

    def kroger_search(
        self,
        query: str,
        page: Optional[int] = None,
        max_results: Optional[int] = 10,
    ) -> List[Document]:
        """Search Kroger grocery products by keyword.

        Returns price (cents), promo/n-for deals, unit price, UPC, stock,
        SNAP eligibility, dietary info, rating and category. Kroger operates
        Kroger, Ralphs, Fred Meyer, Harris Teeter, King Soopers, Fry's and
        other banners. Costs 2 credits.

        Args:
            query: Search keywords (1-200 characters).
            page: Results page, 1-based (1-100).
            max_results: Maximum number of results to return.

        """
        params: Dict[str, Any] = {"page": page}
        resp = self.client.kroger.search(
            query, **{k: v for k, v in params.items() if v is not None}
        )
        docs = _to_documents(resp, "kroger")
        return docs[:max_results] if max_results else docs

    def kroger_product(
        self,
        product_url: Optional[str] = None,
        upc: Optional[str] = None,
        slug: Optional[str] = None,
    ) -> List[Document]:
        """Fetch full Kroger product detail.

        Nutrition facts with allergens and ingredients, rating distribution,
        store aisle location, dietary flags, SNAP eligibility, dimensions and
        country of origin. Pass product_url (from search) or upc. Costs 2
        credits.

        Args:
            product_url: Full or relative product URL from kroger.com. Preferred.
            upc: Kroger product UPC (from search or product URLs).
            slug: Product slug for readable URL. Optional with upc.

        """
        params: Dict[str, Any] = {"product_url": product_url, "upc": upc, "slug": slug}
        resp = self.client.kroger.product(**{k: v for k, v in params.items() if v is not None})
        return _to_documents(resp, "kroger_product")

    # -- map, crawl and batch (enable_crawl=True) ---------------------------

    def map_website(
        self,
        url: str,
        search: Optional[str] = None,
        limit: Optional[int] = None,
        sitemap: Optional[str] = None,
        include_subdomains: Optional[bool] = None,
        include_paths: Optional[List[str]] = None,
        exclude_paths: Optional[List[str]] = None,
        ignore_query_parameters: Optional[bool] = None,
        mode: Optional[str] = None,
    ) -> List[Document]:
        """List a website's page URLs from its sitemaps and start page, deduplicated and filtered.

        Use it to find the right pages before extracting or crawling them. Pages
        the site's robots.txt disallows are left out. Returns one Document whose
        text is the URLs, one per line, with total_found, returned and truncated
        in its metadata. Costs 1 credit in normal mode; advanced costs up to 2
        and ultra up to 10, priced by what the call needed.

        Args:
            url: The site to map. A bare host is upgraded to https.
            search: Only return URLs containing every word of this text.
            limit: Maximum links to return, 1-5000. Default 1000 (or the plan's
                limit if lower: Free 500, PAYG 1,000, paid plans 5,000).
            sitemap: 'include' (default) combines the site's sitemaps with the
                start page's links, 'only' uses sitemaps alone, 'skip' page links alone.
            include_subdomains: Also return URLs on subdomains. Default false.
            include_paths: Path globs to keep, e.g. ["/blog/**"] (max 20).
            exclude_paths: Path globs to drop (max 20).
            ignore_query_parameters: Treat URLs differing only by query string
                as one URL. Default true.
            mode: 'normal' is the standard fetch, 'advanced' renders
                JavaScript-built pages, 'ultra' reaches the hardest pages.

        """
        params = _clean(
            {
                "search": search,
                "limit": limit,
                "sitemap": sitemap,
                "include_subdomains": include_subdomains,
                "include_paths": include_paths,
                "exclude_paths": exclude_paths,
                "ignore_query_parameters": ignore_query_parameters,
                "mode": mode,
            }
        )
        return _map_document(self.client.extract.map(url, **params), url)

    def crawl_website(
        self,
        url: str,
        max_pages: Optional[int] = None,
        max_depth: Optional[int] = None,
        include_paths: Optional[List[str]] = None,
        exclude_paths: Optional[List[str]] = None,
        allow_subdomains: Optional[bool] = None,
        sitemap: Optional[str] = None,
        format: Optional[str] = None,
        mode: Optional[str] = None,
        wait_seconds: int = JOB_DEFAULT_WAIT,
        limit: int = JOB_DEFAULT_LIMIT,
    ) -> List[Document]:
        """Crawl a website from one URL and return its pages as Documents.

        Same-site pages are discovered and extracted breadth-first within the
        page, depth and path limits; robots.txt is obeyed. Starts the crawl and
        waits up to wait_seconds for it to finish. The first Document is the job
        summary (id, status, pages, credits_used, next_cursor); each further
        Document is one page. If status is still queued or running, call
        crawl_status with the id later. Billed per SUCCESSFUL page at extract's
        price for the mode: normal and advanced 1 credit, ultra 2 credits.
        Failed and skipped pages are free, and the crawl stops cleanly if the
        balance runs out.

        Args:
            url: Where the crawl starts. Only pages on the same site are crawled.
            max_pages: Maximum pages to fetch. Default 100. Plan limits: Free 10,
                PAYG 100, Project 500, Bootstrap 2,000, Startup 5,000, Growth 10,000.
            max_depth: Links away from the start page to follow, 0-10 (default 3).
            include_paths: Path globs to crawl, e.g. ["/docs/**"] (max 20).
            exclude_paths: Path globs to skip (max 20).
            allow_subdomains: Also crawl subdomains. Default false.
            sitemap: 'include' (default) also starts from the sitemap, 'skip'
                follows links only, 'only' fetches sitemap pages without following links.
            format: 'markdown' (default), 'text' or 'html' for each page.
            mode: 'normal' is the standard fetch, 'advanced' renders
                JavaScript-built pages, 'ultra' reaches the hardest pages.
            wait_seconds: Seconds to wait for the crawl to finish, 0-300
                (default 45). 0 returns the job id at once.
            limit: Pages returned in this response, 1-100 (default 10).

        """
        body = _clean(
            {
                "max_pages": max_pages,
                "max_depth": max_depth,
                "include_paths": include_paths,
                "exclude_paths": exclude_paths,
                "allow_subdomains": allow_subdomains,
                "sitemap": sitemap,
                "format": format,
                "mode": mode,
            }
        )
        created = self.client.extract.crawl(url, **body)
        return _job_documents(self._wait("crawl", created, wait_seconds, limit), "crawl")

    def crawl_status(
        self,
        id: str,
        cursor: Optional[str] = None,
        limit: int = JOB_DEFAULT_LIMIT,
        status: Optional[str] = None,
        include_content: Optional[bool] = None,
    ) -> List[Document]:
        """Read a crawl's progress and its pages by job id, paged by cursor. Free.

        Returns the job summary Document (status, pages, credits_used,
        next_cursor) followed by one Document per page. Results are kept for 3
        days after the crawl finishes.

        Args:
            id: The crawl's job id.
            cursor: next_cursor from the previous response.
            limit: Pages per response, 1-100 (default 10).
            status: Only pages with this status: succeeded, failed or skipped.
            include_content: False lists pages without their content.

        """
        params = _clean({"cursor": cursor, "limit": limit, "status": status, "include_content": include_content})
        return _job_documents(self.client.extract.crawl_status(id, **params), "crawl")

    def batch_extract(
        self,
        urls: List[str],
        format: Optional[str] = None,
        mode: Optional[str] = None,
        wait_seconds: int = JOB_DEFAULT_WAIT,
        limit: int = JOB_DEFAULT_LIMIT,
    ) -> List[Document]:
        """Extract a list of URLs in one job, each page returned as a Document.

        Starts the batch and waits up to wait_seconds for it to finish. The first
        Document is the job summary (id, status, pages, credits_used,
        next_cursor); each further Document is one page, in input order. If
        status is still queued or running, call batch_status with the id later.
        Billed per SUCCESSFUL page at extract's price for the mode: normal and
        advanced 1 credit, ultra 2 credits. Failed pages are free.

        Args:
            urls: The pages to extract. Plan limits: Free 10, PAYG 50, Project
                100, Bootstrap 500, Startup and Growth 1,000.
            format: 'markdown' (default), 'text' or 'html' for each page.
            mode: 'normal' is the standard fetch, 'advanced' renders
                JavaScript-built pages, 'ultra' reaches the hardest pages.
            wait_seconds: Seconds to wait for the batch to finish, 0-300
                (default 45). 0 returns the job id at once.
            limit: Pages returned in this response, 1-100 (default 10).

        """
        created = self.client.extract.batch(urls, **_clean({"format": format, "mode": mode}))
        return _job_documents(self._wait("batch", created, wait_seconds, limit), "batch")

    def batch_status(
        self,
        id: str,
        cursor: Optional[str] = None,
        limit: int = JOB_DEFAULT_LIMIT,
        status: Optional[str] = None,
        include_content: Optional[bool] = None,
    ) -> List[Document]:
        """Read a batch's progress and its pages by job id, paged by cursor. Free.

        Returns the job summary Document followed by one Document per page, in
        input order. Results are kept for 3 days after the batch finishes.

        Args:
            id: The batch's job id.
            cursor: next_cursor from the previous response.
            limit: Pages per response, 1-100 (default 10).
            status: Only pages with this status: succeeded, failed or skipped.
            include_content: False lists pages without their content.

        """
        params = _clean({"cursor": cursor, "limit": limit, "status": status, "include_content": include_content})
        return _job_documents(self.client.extract.batch_status(id, **params), "batch")

    def _wait(self, kind: str, created: Dict[str, Any], wait_seconds: int, limit: int) -> Dict[str, Any]:
        """Poll a started job until it finishes or the wait runs out, then read its results."""
        job_id = (created.get("data") or {}).get("id")
        if not job_id or wait_seconds <= 0:
            return created
        status_fn = self.client.extract.crawl_status if kind == "crawl" else self.client.extract.batch_status
        deadline = time.monotonic() + min(wait_seconds, JOB_MAX_WAIT)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(JOB_POLL_INTERVAL, remaining))
            probe = status_fn(job_id, limit=1, include_content=False)
            if (probe.get("data") or {}).get("status") in JOB_TERMINAL_STATUSES:
                break
        return status_fn(job_id, limit=limit)
