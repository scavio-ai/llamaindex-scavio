"""Tests for the Scavio LlamaIndex tool spec.

Unit tests replace the Scavio SDK client with a fake that returns canned
responses mirroring the real wire shapes, so they run offline. The
``integration``-marked tests hit the live API and require SCAVIO_API_KEY.
"""

import os

import pytest
from llama_index.core.schema import Document

from llama_index.tools.scavio import ScavioToolSpec
from llama_index.tools.scavio.base import _flatten_text, _records, _to_documents, _url_of

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

# Reddit search (/api/v1/reddit/search): 1 credit since Reddit moved to TikHub.
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


@pytest.fixture()
def spec():
    tool_spec = ScavioToolSpec(api_key=MOCK_API_KEY)
    tool_spec.client = _FakeClient()
    return tool_spec


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
    }


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
