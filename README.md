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

## Tools

`ScavioToolSpec` exposes a **curated subset** of the Scavio API — the endpoints that map cleanly onto RAG `Document`s (each function returns `List[Document]`):

| Tool | What it does | Credits |
|---|---|---|
| `search` | Google SERP — real-time organic web results | 1 |
| `news` | Google News — recent articles on a topic | 1 |
| `reddit_search` | Reddit posts — community discussion and sentiment | 1 |
| `youtube_search` | YouTube — videos, channels, playlists | 2 |
| `youtube_video` | YouTube — full details for one video | 1 |
| `youtube_transcript` | YouTube — transcript or timed subtitles | 8 |
| `youtube_comments` | YouTube — top-level comments on a video | 1 |
| `amazon_search` | Amazon — product listings | 1 |

This is deliberately a subset, not the whole API. For full coverage — all 99 endpoints across Google, YouTube, Amazon, Walmart, Reddit, TikTok, TikTok Shop, Instagram, X, and LinkedIn — point any LlamaIndex agent at the hosted MCP server at [`mcp.scavio.dev`](https://scavio.dev/docs/mcp), which exposes 99 tools with no install required.

## Scavio vs Tavily vs SerpAPI

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

## Migrating from Tavily

```diff
- from llama_index.tools.tavily_research import TavilyToolSpec
- tool_spec = TavilyToolSpec(api_key="tvly-...")
+ from llama_index.tools.scavio import ScavioToolSpec
+ tool_spec = ScavioToolSpec(api_key="sk_live_...")

  docs = tool_spec.search("your query")
```

## Credits

Most calls cost 1 credit. YouTube search costs 2 and YouTube transcripts cost 8. Elsewhere in the API, Instagram costs 2-10 and LinkedIn 1-30. See [scavio.dev/docs](https://scavio.dev/docs).

## About Scavio

[Scavio](https://scavio.dev) is a real-time search API built for AI agents — a unified API over Google, YouTube, Amazon, Walmart, Reddit, TikTok, TikTok Shop, Instagram, X, and LinkedIn that returns clean JSON. It is a cost-effective [Tavily alternative](https://scavio.dev/docs) and [SerpAPI alternative](https://scavio.dev/docs) with broader platform coverage. Learn more in the [LlamaIndex integration docs](https://scavio.dev/docs/llamaindex).

## Links

- Scavio: https://scavio.dev
- Docs: https://scavio.dev/docs/llamaindex
- Dashboard: https://dashboard.scavio.dev
