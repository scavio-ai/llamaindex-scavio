"""Scavio + LlamaIndex cookbook / live smoke test.

Set SCAVIO_API_KEY (get one at https://scavio.dev) and run:

    python cookbook.py

Doubles as a live smoke test: it calls the real Scavio API (1 credit) and,
if OPENAI_API_KEY is set, wires the tools into a LlamaIndex agent.
"""

import os

from llama_index.tools.scavio import ScavioToolSpec


def main() -> None:
    api_key = os.getenv("SCAVIO_API_KEY")
    if not api_key:
        raise SystemExit("Set SCAVIO_API_KEY first (https://scavio.dev)")

    tool_spec = ScavioToolSpec(api_key=api_key)

    # 1. Direct tool call.
    docs = tool_spec.search("best real-time search API for AI agents", max_results=5)
    print(f"Google search returned {len(docs)} documents:")
    for doc in docs[:3]:
        print("  -", doc.text.split("\n")[0][:80], "|", doc.metadata.get("url"))

    # 2. Read the page behind the top result (1 credit on the default tier).
    if docs and docs[0].metadata.get("url"):
        page = tool_spec.extract(docs[0].metadata["url"])
        print(f"\nExtract returned {len(page[0].text)} characters of markdown:")
        print("  ", page[0].text[:200].replace("\n", " "))

    # 3. Optional: hand the tools to a LlamaIndex agent.
    if os.getenv("OPENAI_API_KEY"):
        from llama_index.core.agent.workflow import FunctionAgent
        from llama_index.llms.openai import OpenAI

        agent = FunctionAgent(
            tools=tool_spec.to_tool_list(),
            llm=OpenAI(model="gpt-5.5"),
            system_prompt="You are a research assistant. Use Scavio to find fresh info.",
        )
        import asyncio

        response = asyncio.run(
            agent.run("What are people on Reddit saying about Tavily alternatives?")
        )
        print("\nAgent response:\n", response)


if __name__ == "__main__":
    main()
