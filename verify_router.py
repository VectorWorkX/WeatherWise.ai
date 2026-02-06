"""Verify the router correctly classifies a Bucket 5 (explanation) query."""

from __future__ import annotations

import asyncio
import sys

sys.path.insert(0, "src")

from langchain_core.messages import HumanMessage  # noqa: E402

from graph.builder import build_graph  # noqa: E402


async def main() -> None:
    graph = build_graph()

    test_query = "Why is it so hot in Texas right now?"
    print(f"Query: {test_query}\n")

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=test_query)], "intent": ""}
    )

    print(f"Intent: {result['intent']}")
    expected = "explanation"
    status = "PASS" if result["intent"] == expected else "FAIL"
    print(f"Expected: {expected} — {status}")


if __name__ == "__main__":
    asyncio.run(main())
