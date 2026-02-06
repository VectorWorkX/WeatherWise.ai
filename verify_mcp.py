"""Verify that the Xweather MCP Server is reachable and list available tools."""

from __future__ import annotations

import asyncio
import sys

# Allow running from project root
sys.path.insert(0, "src")

from tools.mcp_bridge import get_mcp_client  # noqa: E402


async def main() -> None:
    print("Connecting to Xweather MCP Server...")
    try:
        client = get_mcp_client()
        tools = await client.get_tools()
        print(f"\nSuccess! Found {len(tools)} tool(s):\n")
        for tool in tools:
            print(f"  - {tool.name}: {tool.description[:80]}")
    except Exception as exc:
        print(f"\nFailed to connect: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
