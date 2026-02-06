"""End-to-end verification: Router → Tool Invoker → Tools → Final Answer."""

from __future__ import annotations

import asyncio
import sys

sys.path.insert(0, "src")

from langchain_core.messages import HumanMessage  # noqa: E402

from graph.builder import build_graph  # noqa: E402
from tools.math_tools import calculator  # noqa: E402
from tools.mcp_bridge import get_mcp_client  # noqa: E402


async def main() -> None:
    # Gather all tools: MCP weather tools + local calculator
    client = get_mcp_client()
    mcp_tools = await client.get_tools()
    all_tools = mcp_tools + [calculator]
    print(f"Loaded {len(all_tools)} tools ({len(mcp_tools)} MCP + 1 calculator)\n")

    # Build graph
    graph = build_graph(all_tools)

    # Test query (Bucket 1: real_time)
    query = "What is the temperature in San Jose right now?"
    print(f"Query: {query}\n")

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=query)], "intent": ""}
    )

    print(f"Intent: {result['intent']}\n")
    print("--- Final Answer ---")
    print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
