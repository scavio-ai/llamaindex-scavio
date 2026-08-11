"""Tests for the Scavio LlamaIndex tool spec.

Unit tests replace the Scavio SDK client with a fake that returns canned
responses mirroring the real wire shapes, so they run offline. The
``integration``-marked tests hit the live API and require SCAVIO_API_KEY.
"""

import os

import pytest
from llama_index.core.schema import Document

from llama_index.tools.scavio import ScavioToolSpec
from llama_index.tools.scavio.base import (
    _extract_document,
    _flatten_text,
    _records,
    _to_documents,
    _url_of,
)

MOCK_API_KEY = "sk_live_test_key_12345"


# --- response fixtures (mirror the real Scavio wire shapes) ----------------

# Google v2 (/api/v2/google): flat response, organic_results with link/snippet.
GOOGLE_RESPONSE = {
    "search_parameters": {"q": "openai"},
    "credits_used": 1,
    "organic_results": [
        {
            "title": f"Result {i}",
            "link": f"https://example.com/{i}",
            "snippet": f"Description {i}",
            "position": i,
        }
        for i in range(1, 13)
    ],
}

# YouTube search (/api/v1/youtube/search): 2 credits, results under data.
YOUTUBE_RESPONSE = {
    "data": {
        "results": [
            {
                "videoId": f"vid{i}",
                "title": {"runs": [{"text": f"Result {i}"}]},
                "description": f"Description {i}",
            }
            for i in range(1, 13)
        ]
    },
    "credits_used": 2,
}

AMAZON_RESPONSE = {
    "data": {
        "page": 1,
        "products": [
            {
                "asin": f"B00{i}",
                "title": f"Product {i}",
                "url": f"/dp/B00{i}",
                "image": f"https://m.media-amazon.com/images/I/{i}.jpg",
            }
            for i in range(1, 6)
        ],
    },
    "credits_used": 1,
}

# Reddit search (/api/v1/reddit/search): 1 credit since Reddit changed upstream source.
# data.results (not data.posts) of flat post objects keyed `text` (not selftext).
REDDIT_RESPONSE = {
    "data": {
        "results": [
            {
                "post_id": f"t3_abc{i}",
                "title": f"Post {i}",
                "url": f"https://www.reddit.com/r/x/comments/abc{i}/post_{i}/",
                "text": f"body {i}",
                "subreddit": "x",
                "author": f"user{i}",
                "score": i * 10,
                "num_comments": i,
            }
            for i in range(1, 4)
        ],
        "next_cursor": "t3_abc3",
        "has_more": True,
    },
    "credits_used": 1,
}


# Extract (/api/v1/extract): a CORE endpoint, not a platform. It returns page
# content under `data`, with no result list to iterate, and is tier-priced by
# `mode` (normal 1, advanced 1, ultra 2) rather than a flat per-call constant.
EXTRACT_RESPONSE = {
    "data": {
        "url": "https://example.com/pricing",
        "format": "markdown",
        "mode": "normal",
        "content": "# Pricing\n\nStarts at $9 a month.",
        "content_length": 31,
    },
    "response_time": 812,
    "credits_used": 1,
    "credits_remaining": 4999,
}


class _FakeNamespace:
    def __init__(self, response, calls=None):
        self._response = response
        self._calls = calls if calls is not None else []

    def __getattr__(self, name):
        def _call(*args, **kwargs):
            self._calls.append((name, args, kwargs))
            return self._response

        return _call


class _FakeClient:
    def __init__(self):
        self.calls = []
        self.google = _FakeNamespace(GOOGLE_RESPONSE, self.calls)
        self.amazon = _FakeNamespace(AMAZON_RESPONSE, self.calls)
        self.reddit = _FakeNamespace(REDDIT_RESPONSE, self.calls)
        self.youtube = _FakeNamespace(YOUTUBE_RESPONSE, self.calls)
        self.extract_response = EXTRACT_RESPONSE

    # extract is a TOP-LEVEL method on the client, never a namespace, so it is
    # a plain method here rather than a _FakeNamespace. A tool that called
    # client.extract.extract() would fail against this fake, which is the point.
    def extract(self, url, **kwargs):
        self.calls.append(("extract", (url,), kwargs))
        return self.extract_response


@pytest.fixture()
def spec():
    tool_spec = ScavioToolSpec(api_key=MOCK_API_KEY)
    tool_spec.client = _FakeClient()
    return tool_spec


# --- curated-surface guards ------------------------------------------------

CURATED_TOOLS = [
    "search",
    "news",
    "reddit_search",
    "youtube_search",
    "youtube_video",
    "youtube_transcript",
    "youtube_comments",
    "amazon_search",
    "extract",
]


def test_surface_is_exactly_nine_tools():
    """Curated by design: 8 endpoints over Google, Reddit, YouTube, Amazon, plus extract.

    Guards the README's counts. A new tool here means the README table, the platform
    counts and the credit table all need updating in the same change.
    """
    assert ScavioToolSpec.spec_functions == CURATED_TOOLS
    assert len(ScavioToolSpec.spec_functions) == 9
    assert all(callable(getattr(ScavioToolSpec, name)) for name in CURATED_TOOLS)


def test_to_tool_list_exposes_the_same_nine(spec):
    """to_tool_list is what an agent actually sees; it must match spec_functions."""
    assert [t.metadata.name for t in spec.to_tool_list()] == CURATED_TOOLS


def test_uncovered_platforms_are_absent(spec):
    """Walmart, TikTok, TikTok Shop, Instagram, X and LinkedIn are not exposed here."""
    for absent in ("walmart", "tiktok", "tiktok_shop", "instagram", "x_", "linkedin"):
        assert not any(name.startswith(absent) for name in ScavioToolSpec.spec_functions)


def test_deprecated_youtube_metadata_alias_not_exposed():
    """/youtube/metadata is a deprecated alias of /youtube/video, not a peer tool."""
    assert "youtube_metadata" not in ScavioToolSpec.spec_functions
    assert "youtube_video" in ScavioToolSpec.spec_functions


def test_every_tool_has_a_docstring():
    """Agent-facing tools are selected off their docstrings; none may be blank."""
    for name in ScavioToolSpec.spec_functions:
        doc = getattr(ScavioToolSpec, name).__doc__
        assert doc and doc.strip(), name


def test_google_tools_use_v2_vocabulary(spec):
    """/api/v1/google is retired (410). Only gl/hl reach the SDK, never v1 params."""
    spec.search("openai", gl="us", hl="en")
    assert spec.client.calls[-1] == ("search", ("openai",), {"gl": "us", "hl": "en"})
    spec.news("openai", gl="us")
    assert spec.client.calls[-1] == ("news", (), {"query": "openai", "gl": "us"})
    import inspect

    for name in ("search", "news"):
        params = set(inspect.signature(getattr(ScavioToolSpec, name)).parameters)
        assert not params & {"light_request", "country_code", "language", "search_type", "page"}


# --- helper unit tests -----------------------------------------------------

def test_records_google_top_level():
    assert len(_records(GOOGLE_RESPONSE)) == 12


def test_records_amazon_nested():
    assert len(_records(AMAZON_RESPONSE)) == 5


def test_records_reddit_nested_results():
    """Reddit search nests its posts under data.results, not data.posts."""
    assert len(_records(REDDIT_RESPONSE)) == 3


def test_to_documents_falls_back_to_json():
    docs = _to_documents({"weird": "shape"}, "google")
    assert len(docs) == 1
    assert "weird" in docs[0].text


def test_to_documents_sets_url_and_source():
    docs = _to_documents(GOOGLE_RESPONSE, "google")
    assert docs[0].metadata["url"] == "https://example.com/1"
    assert docs[0].metadata["source"] == "google"


# --- tool method tests -----------------------------------------------------

def test_search_returns_documents(spec):
    docs = spec.search("openai")
    assert all(isinstance(d, Document) for d in docs)
    assert len(docs) == 10  # default max_results
    assert "Result 1" in docs[0].text


def test_search_respects_max_results(spec):
    assert len(spec.search("openai", max_results=3)) == 3


def test_amazon_search_uses_nested_products(spec):
    docs = spec.amazon_search("laptop")
    assert docs[0].metadata["source"] == "amazon"
    assert "Product 1" in docs[0].text


def test_amazon_search_sends_country_not_domain(spec):
    """The marketplace param is country (a 2-letter code), never domain."""
    spec.amazon_search("laptop", country="gb")
    assert spec.client.calls[-1] == ("search", ("laptop",), {"country": "gb"})


def test_reddit_search_reads_flat_post_objects(spec):
    docs = spec.reddit_search("best search api")
    assert docs[0].metadata["url"].startswith("https://www.reddit.com/")
    assert docs[0].metadata["source"] == "reddit"
    assert "body 1" in docs[0].text


def test_reddit_search_only_sends_query_and_cursor(spec):
    """/reddit/search takes query + cursor only; sort and type do not exist."""
    spec.reddit_search("best search api", cursor="t3_abc3")
    assert spec.client.calls[-1] == ("search", ("best search api",), {"cursor": "t3_abc3"})


def test_reddit_search_signature_has_no_sort_or_type():
    import inspect

    params = inspect.signature(ScavioToolSpec.reddit_search).parameters
    assert "sort" not in params
    assert "type" not in params
    assert "cursor" in params


def test_to_tool_list_exposes_all_functions(spec):
    tools = spec.to_tool_list()
    names = {t.metadata.name for t in tools}
    assert names == {
        "search",
        "news",
        "reddit_search",
        "youtube_search",
        "youtube_video",
        "youtube_transcript",
        "youtube_comments",
        "amazon_search",
        "extract",
    }


# --- extract tests ---------------------------------------------------------

def test_extract_returns_one_document_with_the_whole_page(spec):
    """Extract returns CONTENT, not a record list: one Document holding the page."""
    docs = spec.extract("https://example.com/pricing")
    assert len(docs) == 1
    assert docs[0].text == "# Pricing\n\nStarts at $9 a month."
    assert docs[0].metadata["source"] == "extract"
    assert docs[0].metadata["url"] == "https://example.com/pricing"
    assert docs[0].metadata["format"] == "markdown"
    assert docs[0].metadata["mode"] == "normal"


def test_extract_is_a_top_level_method_not_a_namespace(spec):
    """scavio.extract(url), never scavio.extract.extract()."""
    spec.extract("https://example.com/pricing")
    assert spec.client.calls[-1] == ("extract", ("https://example.com/pricing",), {})


def test_extract_sends_format_and_mode_and_drops_none(spec):
    spec.extract("https://example.com", format="text", mode="ultra")
    assert spec.client.calls[-1] == (
        "extract",
        ("https://example.com",),
        {"format": "text", "mode": "ultra"},
    )
    spec.extract("https://example.com", mode="advanced")
    assert spec.client.calls[-1] == ("extract", ("https://example.com",), {"mode": "advanced"})


def test_extract_signature_offers_only_url_format_mode():
    """The route takes url, format and mode; anything else is stripped server-side."""
    import inspect

    params = inspect.signature(ScavioToolSpec.extract).parameters
    assert set(params) == {"self", "url", "format", "mode"}


def test_extract_document_falls_back_when_content_is_missing():
    """An unexpected shape must still reach the caller rather than vanish."""
    docs = _extract_document({"weird": "shape"}, "https://example.com")
    assert len(docs) == 1
    assert "weird" in docs[0].text
    assert docs[0].metadata["source"] == "extract"


def test_extract_document_does_not_truncate_a_long_page():
    """_to_documents caps its JSON fallback at 8000 chars; a real page must survive whole."""
    body = "x" * 50_000
    docs = _extract_document(
        {"data": {"url": "https://example.com", "content": body, "content_length": len(body)}},
        "https://example.com",
    )
    assert len(docs[0].text) == 50_000


# --- real wire-shape regression tests (offline) ----------------------------

# Google v2: organic_results at top level with link/snippet keys.
GOOGLE_V2_RESPONSE = {
    "search_parameters": {"q": "coffee"},
    "organic_results": [
        {"position": 1, "title": "Best Coffee", "link": "https://x.com/a", "snippet": "great"},
        {"position": 2, "title": "More Coffee", "link": "https://x.com/b", "snippet": "good"},
    ],
    "credits_used": 1,
}

# YouTube search: data.results with rich-text title and videoId (no url). 2 credits.
YOUTUBE_REAL_RESPONSE = {
    "data": {
        "results": [
            {
                "videoId": "aywZrzNaKjs",
                "title": {"runs": [{"text": "LangChain Explained"}]},
            }
        ]
    },
    "credits_used": 2,
}

# Amazon: data.products with relative url + asin, image (not url_image).
AMAZON_REAL_RESPONSE = {
    "data": {
        "products": [
            {
                "asin": "B088NRLMPV",
                "title": "Anker Cable",
                "url": "/Anker/dp/B088NRLMPV/ref=sr",
                "image": "https://m.media-amazon.com/images/I/61.jpg",
                "badge": "Amazon's Choice",
            }
        ]
    },
    "credits_used": 1,
}


def test_records_google_v2_organic_results():
    assert len(_records(GOOGLE_V2_RESPONSE)) == 2


def test_documents_google_v2_uses_link_and_snippet():
    docs = _to_documents(GOOGLE_V2_RESPONSE, "google")
    assert docs[0].metadata["url"] == "https://x.com/a"
    assert "great" in docs[0].text


def test_flatten_text_handles_youtube_runs():
    assert _flatten_text({"runs": [{"text": "Lang"}, {"text": "Chain"}]}) == "LangChain"


def test_url_of_builds_youtube_url_from_videoid():
    assert _url_of({"videoId": "aywZrzNaKjs"}) == "https://www.youtube.com/watch?v=aywZrzNaKjs"


def test_url_of_builds_amazon_url_from_asin():
    assert _url_of({"asin": "B088NRLMPV", "url": "/Anker/dp/B088NRLMPV"}) == "https://www.amazon.com/dp/B088NRLMPV"


def test_documents_youtube_real_shape():
    docs = _to_documents(YOUTUBE_REAL_RESPONSE, "youtube")
    assert docs[0].text.strip() == "LangChain Explained"
    assert docs[0].metadata["url"] == "https://www.youtube.com/watch?v=aywZrzNaKjs"


def test_documents_amazon_real_shape():
    docs = _to_documents(AMAZON_REAL_RESPONSE, "amazon")
    assert docs[0].metadata["url"] == "https://www.amazon.com/dp/B088NRLMPV"


# --- YouTube tool tests ----------------------------------------------------

def test_youtube_search_returns_documents(spec):
    docs = spec.youtube_search("langchain", max_results=3)
    assert len(docs) == 3
    assert docs[0].metadata["source"] == "youtube"
    assert docs[0].metadata["url"] == "https://www.youtube.com/watch?v=vid1"


def test_youtube_video_returns_documents(spec):
    docs = spec.youtube_video("aywZrzNaKjs")
    assert docs[0].metadata["source"] == "youtube_video"


def test_youtube_comments_respects_max_results(spec):
    assert len(spec.youtube_comments("aywZrzNaKjs", max_results=2)) == 2


def test_youtube_transcript_returns_single_document(spec):
    spec.client.youtube = _FakeNamespace(
        {
            "data": {
                "video_id": "aywZrzNaKjs",
                "language": "en",
                "content": "hello world transcript",
            },
            "credits_used": 8,
        },
        spec.client.calls,
    )
    docs = spec.youtube_transcript("aywZrzNaKjs")
    assert len(docs) == 1
    assert docs[0].text == "hello world transcript"
    assert docs[0].metadata["url"] == "https://www.youtube.com/watch?v=aywZrzNaKjs"


# --- live integration tests (require SCAVIO_API_KEY) -----------------------

@pytest.mark.integration
def test_live_search():
    key = os.getenv("SCAVIO_API_KEY")
    if not key:
        pytest.skip("SCAVIO_API_KEY not set")
    docs = ScavioToolSpec(api_key=key).search("what is an AI agent", max_results=5)
    assert len(docs) >= 1
    assert isinstance(docs[0], Document)
    assert docs[0].text.strip()
