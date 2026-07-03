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

Get a Scavio API key from the [Scavio Dashboard](https://dashboard.scavio.dev) (new accounts get free credits, no credit card). Set `SCAVIO_API_KEY` or pass `api_key=` to the tool spec.

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

`ScavioToolSpec` exposes these functions (each returns `List[Document]`):

| Tool | What it does |
|---|---|
| `search` | Google SERP — real-time organic web results |
| `news` | Google News — recent articles on a topic |
| `reddit_search` | Reddit posts — community discussion and sentiment |
| `youtube_search` | YouTube — videos, channels, playlists |
| `amazon_search` | Amazon — product listings |

Need every endpoint (Walmart, TikTok, Instagram, Maps, Shopping, Trends, and more)? Point any LlamaIndex agent at the hosted MCP server at [`mcp.scavio.dev`](https://scavio.dev/docs/mcp) — no install required.

## Scavio vs Tavily vs SerpAPI

| | Scavio | Tavily | SerpAPI |
|---|---|---|---|
| Google SERP | Yes | No (web search only) | Yes |
| Google News | Yes | No | Yes |
| Reddit | Yes | No | No |
| YouTube | Yes | No | Yes |
| Amazon / Walmart | Yes | No | Yes (add-on) |
| TikTok / Instagram | Yes | No | No |
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

Most calls cost 1 credit; Reddit costs 2. See [scavio.dev/docs](https://scavio.dev/docs).

## About Scavio

[Scavio](https://scavio.dev) is a real-time search API built for AI agents — a unified API over Google, YouTube, Amazon, Walmart, Reddit, TikTok, and Instagram that returns clean JSON. It is a cost-effective [Tavily alternative](https://scavio.dev/docs) and [SerpAPI alternative](https://scavio.dev/docs) with broader platform coverage. Learn more in the [LlamaIndex integration docs](https://scavio.dev/docs/llamaindex).

## Links

- Scavio: https://scavio.dev
- Docs: https://scavio.dev/docs/llamaindex
- Dashboard: https://dashboard.scavio.dev
