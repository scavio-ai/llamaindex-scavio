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

GOOGLE_RESPONSE = {
    "query": "openai",
    "credits_used": 1,
    "results": [
        {
            "title": f"Result {i}",
            "url": f"https://example.com/{i}",
            "description": f"Description {i}",
            "position": i,
        }
        for i in range(1, 13)
    ],
}

AMAZON_RESPONSE = {
    "data": {
        "page": 1,
        "products": [
            {"asin": f"B00{i}", "title": f"Product {i}", "url": f"/dp/B00{i}"}
            for i in range(1, 6)
        ],
    },
    "credits_used": 1,
}

REDDIT_RESPONSE = {
    "data": {
        "posts": [
            {"title": f"Post {i}", "permalink": f"https://reddit.com/r/x/{i}",
             "selftext": f"body {i}"}
            for i in range(1, 4)
        ]
    },
    "credits_used": 2,
}


class _FakeNamespace:
    def __init__(self, response):
        self._response = response

    def __getattr__(self, _name):
        def _call(*_args, **_kwargs):
            return self._response

        return _call


class _FakeClient:
    def __init__(self):
        self.google = _FakeNamespace(GOOGLE_RESPONSE)
        self.amazon = _FakeNamespace(AMAZON_RESPONSE)
        self.reddit = _FakeNamespace(REDDIT_RESPONSE)
        self.youtube = _FakeNamespace(GOOGLE_RESPONSE)


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


def test_records_reddit_nested_posts():
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


def test_reddit_search_uses_permalink(spec):
    docs = spec.reddit_search("best search api")
    assert docs[0].metadata["url"].startswith("https://reddit.com/")
    assert docs[0].metadata["source"] == "reddit"


def test_to_tool_list_exposes_all_functions(spec):
    tools = spec.to_tool_list()
    names = {t.metadata.name for t in tools}
    assert names == {
        "search",
        "news",
        "reddit_search",
        "youtube_search",
        "amazon_search",
    }


# --- real wire-shape regression tests (offline) ----------------------------

# Google full mode: organic_results at top level with link/snippet keys.
GOOGLE_FULL_RESPONSE = {
    "search_parameters": {"q": "coffee"},
    "organic_results": [
        {"position": 1, "title": "Best Coffee", "link": "https://x.com/a", "snippet": "great"},
        {"position": 2, "title": "More Coffee", "link": "https://x.com/b", "snippet": "good"},
    ],
    "credits_used": 1,
}

# YouTube: data.results with rich-text title and videoId (no url).
YOUTUBE_REAL_RESPONSE = {
    "data": {
        "results": [
            {
                "videoId": "aywZrzNaKjs",
                "title": {"runs": [{"text": "LangChain Explained"}]},
            }
        ]
    },
    "credits_used": 1,
}

# Amazon: data.products with relative url + asin.
AMAZON_REAL_RESPONSE = {
    "data": {
        "products": [{"asin": "B088NRLMPV", "title": "Anker Cable", "url": "/Anker/dp/B088NRLMPV/ref=sr"}]
    },
    "credits_used": 1,
}


def test_records_google_full_organic_results():
    assert len(_records(GOOGLE_FULL_RESPONSE)) == 2


def test_documents_google_full_uses_link_and_snippet():
    docs = _to_documents(GOOGLE_FULL_RESPONSE, "google")
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
