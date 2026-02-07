"""Smoke test: run one query per bucket to verify the full agent."""

from __future__ import annotations

import asyncio
import sys

sys.path.insert(0, "src")

from langchain_core.messages import HumanMessage  # noqa: E402

from graph.builder import build_graph  # noqa: E402
from tools.math_tools import calculator  # noqa: E402
from tools.mcp_bridge import get_mcp_client  # noqa: E402
from tools.rag_tool import search_knowledge_base  # noqa: E402

TEST_QUERIES = [
    ("Bucket 1 — real_time",    "What is the temperature in San Jose right now?"),
    ("Bucket 2 — comparative",  "Is it hotter in Phoenix or Miami right now?"),
    ("Bucket 4 — safety",       "Are there any active weather alerts in Oklahoma?"),
    ("Bucket 5 — explanation",  "Why is there a heat dome over the Midwest?"),
    ("Bucket 6 — math",         "What is the temperature in Boston in Kelvin?"),
]


async def main() -> None:
    client = get_mcp_client()
    mcp_tools = await client.get_tools()
    all_tools = mcp_tools + [calculator, search_knowledge_base]
    graph = build_graph(all_tools)

    for label, query in TEST_QUERIES:
        print(f"\n{'='*60}")
        print(f"{label}")
        print(f"Query: {query}")
        print(f"{'='*60}")

        result = await graph.ainvoke(
            {"messages": [HumanMessage(content=query)], "intent": ""}
        )

        print(f"Intent: {result['intent']}")
        print(f"\n{result['messages'][-1].content[:300]}")


if __name__ == "__main__":
    asyncio.run(main())
