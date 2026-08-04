# llama-index-tools-scavio

[![PyPI version](https://img.shields.io/pypi/v/llama-index-tools-scavio.svg)](https://pypi.org/project/llama-index-tools-scavio/)
[![Python versions](https://img.shields.io/pypi/pyversions/llama-index-tools-scavio.svg)](https://pypi.org/project/llama-index-tools-scavio/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[Scavio](https://scavio.dev) real-time search tools for [LlamaIndex](https://www.llamaindex.ai/) — Google, Google News, Reddit, YouTube, and Amazon as clean `Document`s, with one API key. A drop-in [Tavily alternative](https://scavio.dev/docs) and [SerpAPI alternative](https://scavio.dev/docs) for LlamaIndex RAG pipelines and agents.

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

## Scope: 8 endpoints, by design

`ScavioToolSpec` is a **curated subset** of the Scavio API, not a wrapper around all of it. It exposes **8** of Scavio's 98 endpoints across **4** of its 10 platforms — the ones that map cleanly onto RAG `Document`s (every function returns `List[Document]`):

| Tool | Endpoint | Platform | What it does | Credits |
|---|---|---|---|---|
| `search` | `POST /api/v2/google` | Google | Real-time organic web results | 1 |
| `news` | `POST /api/v2/google/news` | Google | Recent articles on a topic | 1 |
| `reddit_search` | `POST /api/v1/reddit/search` | Reddit | Community discussion and sentiment | 1 |
| `youtube_search` | `POST /api/v1/youtube/search` | YouTube | Videos, channels, playlists | 2 |
| `youtube_video` | `POST /api/v1/youtube/video` | YouTube | Full details for one video | 1 |
| `youtube_transcript` | `POST /api/v1/youtube/transcript` | YouTube | Transcript or timed subtitles | 8 |
| `youtube_comments` | `POST /api/v1/youtube/comments` | YouTube | Top-level comments on a video | 1 |
| `amazon_search` | `POST /api/v1/amazon/search` | Amazon | Product listings | 1 |

Platforms covered here: **Google (2), YouTube (4), Reddit (1), Amazon (1).** Not covered: Walmart, TikTok, TikTok Shop, Instagram, X, LinkedIn, and the other 10 Google v2 verticals (Maps, Shopping, Flights, Hotels, Trends, AI Mode, ...). That is deliberate — this is a document-retrieval tool spec, not an API client — and it stays that way.

### Reaching the rest of the API

For full coverage — all 98 endpoints across **Google, YouTube, Amazon, Walmart, Reddit, TikTok, TikTok Shop, Instagram, X, and LinkedIn** — point a LlamaIndex agent at the hosted MCP server at `https://mcp.scavio.dev/mcp`, which exposes 100 tools (one per endpoint) with no install required:

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
| Hosted MCP server | Yes | Yes | No |
| LlamaIndex tool | Yes | Yes | Yes |

In `ScavioToolSpec`: Google SERP, Google News, Reddit, YouTube, Amazon search. Everything else in the Scavio column comes from MCP or the SDK.

## Migrating from Tavily

```diff
- from llama_index.tools.tavily_research import TavilyToolSpec
- tool_spec = TavilyToolSpec(api_key="tvly-...")
+ from llama_index.tools.scavio import ScavioToolSpec
+ tool_spec = ScavioToolSpec(api_key="sk_live_...")

  docs = tool_spec.search("your query")
```

## Credits

**This package is not flat 1 credit.** Six of the eight tools cost 1, but `youtube_search` costs 2 and `youtube_transcript` costs **8** — budget for it if you are bulk-ingesting transcripts into an index.

| Tool | Credits |
|---|---|
| `search`, `news`, `reddit_search`, `youtube_video`, `youtube_comments`, `amazon_search` | 1 |
| `youtube_search` | 2 |
| `youtube_transcript` | 8 |

Elsewhere in the API (not exposed here): YouTube streams 3, Instagram 2-10, LinkedIn 1-10 with a job at 30, everything else 1. New accounts get 50 one-time signup credits — no monthly refill. See [scavio.dev/docs](https://scavio.dev/docs).

## About Scavio

[Scavio](https://scavio.dev) is a real-time search API built for AI agents — a unified API over Google, YouTube, Amazon, Walmart, Reddit, TikTok, TikTok Shop, Instagram, X, and LinkedIn that returns clean JSON. It is a cost-effective [Tavily alternative](https://scavio.dev/docs) and [SerpAPI alternative](https://scavio.dev/docs) with broader platform coverage. Learn more in the [LlamaIndex integration docs](https://scavio.dev/docs/llamaindex).

## Links

- Scavio: https://scavio.dev
- Docs: https://scavio.dev/docs/llamaindex
- Dashboard: https://dashboard.scavio.dev
