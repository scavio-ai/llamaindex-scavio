# llama-index-tools-scavio

[![PyPI version](https://img.shields.io/pypi/v/llama-index-tools-scavio.svg)](https://pypi.org/project/llama-index-tools-scavio/)
[![Python versions](https://img.shields.io/pypi/pyversions/llama-index-tools-scavio.svg)](https://pypi.org/project/llama-index-tools-scavio/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[Scavio](https://scavio.dev) real-time search and page-extraction tools for [LlamaIndex](https://www.llamaindex.ai/) — Google, Google News, Reddit, YouTube, Amazon, and `extract` (read any URL) as clean `Document`s, plus opt-in tools to map a site's URLs, crawl a whole site, and batch-extract a URL list, with one API key. A drop-in [Tavily alternative](https://scavio.dev/docs) and [SerpAPI alternative](https://scavio.dev/docs) for LlamaIndex RAG pipelines and agents.

## Install

```bash
pip install llama-index-tools-scavio
```

## Setup

Get a Scavio API key from the [Scavio Dashboard](https://dashboard.scavio.dev) (new accounts get 50 free signup credits, one-time, no credit card). Set `SCAVIO_API_KEY` or pass `api_key=` to the tool spec.

## Usage

```python
from llama_index.tools.scavio import ScavioToolSpec

tool_spec = ScavioToolSpec()  # reads SCAVIO_API_KEY

# Direct call — returns a list of llama_index Document objects
docs = tool_spec.search("best real-time search API for AI agents", max_results=5)
for doc in docs:
    print(doc.text, doc.metadata["url"])

# Read the page behind a result — one Document holding the whole page
page = tool_spec.extract(docs[0].metadata["url"])
print(page[0].text)
```

Hand the tools to an agent:

```python
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI

agent = FunctionAgent(
    tools=ScavioToolSpec().to_tool_list(),
    llm=OpenAI(model="gpt-5.5"),
    system_prompt="You are a research assistant. Use Scavio for fresh web data.",
)
response = await agent.run("What are people on Reddit saying about Tavily alternatives?")
```

## Map, crawl and batch (opt-in)

`ScavioToolSpec(enable_crawl=True)` adds five tools for whole sites. They are off by default because crawl and batch bill per page, so an existing tool list is unchanged until you opt in.

| Tool | Endpoint | What it returns | Credits |
|---|---|---|---|
| `map_website` | `POST /api/v1/extract/map` | One Document: the site's page URLs, one per line, from its sitemaps and start page; `total_found` / `truncated` in metadata | 1 on `normal` (up to 2 on `advanced`, up to 10 on `ultra`) |
| `crawl_website` | `POST /api/v1/extract/crawl` | A job-summary Document, then one Document per page | per successful page: 1 / 1 / 2 by `mode` |
| `crawl_status` | `GET /api/v1/extract/crawl/{id}` | The same, for a crawl that outlived the wait or the next `cursor` page | free |
| `batch_extract` | `POST /api/v1/extract/batch` | A job-summary Document, then one Document per URL, in input order | per successful page: 1 / 1 / 2 by `mode` |
| `batch_status` | `GET /api/v1/extract/batch/{id}` | The same, for a batch | free |

Crawl and batch run as background jobs. `crawl_website` and `batch_extract` start the job and **wait up to `wait_seconds` (default 45, max 300)** for it to finish, then return the job summary (`id`, `status`, `pages`, `credits_used`, `next_cursor`) as the first Document and up to `limit` pages (default 10) after it. A job still `queued` or `running` when the wait ends keeps going; read the rest with `crawl_status` / `batch_status` and its `id`. That keeps an agent step bounded while a large crawl (up to 6 hours) runs on.

```python
import json

tool_spec = ScavioToolSpec(enable_crawl=True)

urls = tool_spec.map_website("https://hono.dev/docs", limit=50)[0].text.splitlines()

docs = tool_spec.crawl_website("https://hono.dev/docs", max_pages=25, include_paths=["/docs/**"])
job = json.loads(docs[0].text)        # {"id", "status", "pages", "credits_used", "next_cursor", ...}
pages = docs[1:]                      # one Document per page, metadata url/status/depth
if job["next_cursor"]:
    more = tool_spec.crawl_status(job["id"], cursor=job["next_cursor"])

batch = tool_spec.batch_extract(["https://example.com/", "https://hono.dev/docs/getting-started/basic"])
```

Crawl and batch bill **per successful page** at extract's price for the mode (normal and advanced 1 credit, ultra 2); failed and skipped pages are free, and a job stops cleanly with `stopped_reason: "insufficient_credits"` if the balance runs out. Reading a job is free. Crawl and map obey the site's robots.txt and stay on the start site (subdomains optional); results are kept for 3 days. Plan limits (pages per crawl / active jobs / URLs per batch): Free 10/1/10, PAYG 100/1/50, Project 500/2/100, Bootstrap 2,000/3/500, Startup 5,000/5/1,000, Growth 10,000/10/1,000. Cancellation, webhooks and idempotency keys exist on the [REST API](https://scavio.dev/docs/extract) and in the `scavio` SDK (`client.extract.cancel_crawl(...)`), but are not agent tools here.

## Scope: 9 endpoints, by design

`ScavioToolSpec` is a **curated subset** of the Scavio API, not a wrapper around all of it. By default it exposes **9** of Scavio's 195 endpoints (plus the five opt-in map/crawl/batch tools above) — 8 across **4** of its 31 platforms, plus the core `extract` endpoint — the ones that map cleanly onto RAG `Document`s (every function returns `List[Document]`):

| Tool | Endpoint | Platform | What it does | Credits |
|---|---|---|---|---|
| `search` | `POST /api/v2/google` | Google | Real-time organic web results | 1 |
| `news` | `POST /api/v2/google/news` | Google | Recent articles on a topic | 1 |
| `reddit_search` | `POST /api/v1/reddit/search` | Reddit | Community discussion and sentiment | 1 |
| `youtube_search` | `POST /api/v1/youtube/search` | YouTube | Videos, channels, playlists | 2 |
| `youtube_video` | `POST /api/v1/youtube/video` | YouTube | Full details for one video | 1 |
| `youtube_transcript` | `POST /api/v1/youtube/transcript` | YouTube | Transcript or timed subtitles | 5 |
| `youtube_comments` | `POST /api/v1/youtube/comments` | YouTube | Top-level comments on a video | 1 |
| `amazon_search` | `POST /api/v1/amazon/search` | Amazon | Product listings | 1 |
| `extract` | `POST /api/v1/extract` | *(core, not a platform)* | Any URL as Markdown, plain text, or raw HTML | 1 / 1 / 2 by `mode` |

Platforms covered here: **Google (2), YouTube (4), Reddit (1), Amazon (1)**, plus `extract`, which belongs to no platform — it reads any URL. Not covered: Walmart, TikTok, TikTok Shop, Instagram, X, LinkedIn, eBay, Target, Home Depot, Zillow, Redfin, Booking, Airbnb, Tripadvisor, Yelp, Indeed, Glassdoor, the App Store, Google Play, SEC, Companies House, G2, Capterra, Google Ads, Meta Ad Library, Threads, Kuaishou, and the other 12 Google verticals (Maps, Shopping, Flights, Hotels, Trends, AI Mode, ...). That is deliberate — this is a document-retrieval tool spec, not an API client — and it stays that way.

### Reaching the rest of the API

For full coverage — all 195 endpoints across 31 platforms (**Google, YouTube, Amazon, Walmart, Reddit, TikTok, TikTok Shop, Instagram, X, LinkedIn, eBay, Target, Home Depot, Zillow, Redfin, Booking, Airbnb, Tripadvisor, Yelp, Indeed, Glassdoor, App Store, Google Play, SEC, Companies House, G2, Capterra, Google Ads, Meta Ad Library, Threads, Kuaishou**) — point a LlamaIndex agent at the hosted MCP server at `https://mcp.scavio.dev/mcp`, which exposes 191 tools with no install required:

```python
from llama_index.tools.mcp import BasicMCPClient, McpToolSpec

scavio_mcp = McpToolSpec(
    client=BasicMCPClient(
        "https://mcp.scavio.dev/mcp",
        headers={"x-api-key": "sk_live_..."},
    )
)
tools = await scavio_mcp.to_tool_list_async()
```

See the [MCP docs](https://scavio.dev/docs/mcp). Or call the `scavio` SDK directly — it is already a dependency here — and wrap whichever endpoints you need.

## Scavio vs Tavily vs SerpAPI

These rows compare the **APIs**, not this tool spec. Rows marked below the table are the ones `ScavioToolSpec` itself exposes; the rest are reachable via MCP or the `scavio` SDK.

| | Scavio | Tavily | SerpAPI |
|---|---|---|---|
| Google SERP | Yes | No (web search only) | Yes |
| Google News | Yes | No | Yes |
| Reddit | Yes | No | No |
| YouTube | Yes | No | Yes |
| Amazon / Walmart | Yes | No | Yes (add-on) |
| TikTok / TikTok Shop / Instagram | Yes | No | No |
| X / LinkedIn | Yes | No | No |
| Read any URL (extract) | Yes | Yes | No |
| Hosted MCP server | Yes | Yes | No |
| LlamaIndex tool | Yes | Yes | Yes |

In `ScavioToolSpec`: Google SERP, Google News, Reddit, YouTube, Amazon search, and URL extraction. Everything else in the Scavio column comes from MCP or the SDK.

## Migrating from Tavily

```diff
- from llama_index.tools.tavily_research import TavilyToolSpec
- tool_spec = TavilyToolSpec(api_key="tvly-...")
+ from llama_index.tools.scavio import ScavioToolSpec
+ tool_spec = ScavioToolSpec(api_key="sk_live_...")

  docs = tool_spec.search("your query")
```

## Credits

**This package is not flat 1 credit.** Six of the nine tools cost 1, `youtube_search` costs 2, `youtube_transcript` costs **5** — budget for it if you are bulk-ingesting transcripts into an index — and `extract` is priced by the tier you ask for, not per call.

| Tool | Credits |
|---|---|
| `search`, `news`, `reddit_search`, `youtube_video`, `youtube_comments`, `amazon_search` | 1 |
| `youtube_search` | 2 |
| `youtube_transcript` | 5 |
| `extract` | **1** on `mode="normal"`, **1** on `"advanced"`, **2** on `"ultra"` |
| `map_website` *(opt-in)* | **1** on `normal`; up to 2 on `advanced`, up to 10 on `ultra` |
| `crawl_website`, `batch_extract` *(opt-in)* | per successful page: **1** on `normal` or `advanced`, **2** on `ultra` |
| `crawl_status`, `batch_status` *(opt-in)* | free |

`extract` bills only on a successful read: a dead link, a bot wall or a timeout costs nothing. Start on `normal`; move to `advanced` when the page builds its content with JavaScript, and to `ultra` only for the hardest pages.

Elsewhere in the API (not exposed here): YouTube streams 3, Instagram 2-10, LinkedIn 1-10 with a job at 30, Kuaishou 1-40 per endpoint, G2 5, Capterra 2, everything else 1. New accounts get 50 one-time signup credits — no monthly refill. See [scavio.dev/docs](https://scavio.dev/docs) and [the extract docs](https://scavio.dev/docs/extract).

## About Scavio

[Scavio](https://scavio.dev) is a real-time search API built for AI agents — 195 endpoints across 31 platforms (Google, YouTube, Amazon, Walmart, Reddit, TikTok, Instagram, X, LinkedIn, eBay, Zillow, Booking, Indeed, the App Store, SEC filings and more), plus `extract` to read any URL, all returning clean JSON on one key. It is a cost-effective [Tavily alternative](https://scavio.dev/docs) and [SerpAPI alternative](https://scavio.dev/docs) with broader platform coverage. Learn more in the [LlamaIndex integration docs](https://scavio.dev/docs/llamaindex).

## Links

- Scavio: https://scavio.dev
- Docs: https://scavio.dev/docs/llamaindex
- Dashboard: https://dashboard.scavio.dev
