"""Xweather MCP client — connects to the Xweather MCP server and loads tools."""

from __future__ import annotations

import asyncio
import os
import time

from langchain_mcp_adapters.client import MultiServerMCPClient

from weatherwise.logging import get_logger

logger = get_logger(__name__)

# Module-level tool cache to avoid re-negotiating MCP schemas on every fetch.
# Each negotiation round-trip adds ~2-3s latency and inflates input tokens.
_tool_cache: list | None = None
_tool_cache_lock = asyncio.Lock()
_tool_cache_time: float = 0.0
_TOOL_CACHE_TTL = 300.0  # 5 minutes


def get_xweather_mcp_config() -> dict:
    """Build the MultiServerMCPClient config for the Xweather MCP server.

    Requires XWEATHER_CLIENT_ID and XWEATHER_CLIENT_SECRET env vars.
    The combined key format is: <client_id>_<client_secret>
    """
    client_id = os.environ.get("XWEATHER_CLIENT_ID", "")
    client_secret = os.environ.get("XWEATHER_CLIENT_SECRET", "")

    if not client_id or not client_secret:
        raise ValueError(
            "XWEATHER_CLIENT_ID and XWEATHER_CLIENT_SECRET must be set in .env"
        )

    api_key = f"{client_id}_{client_secret}"

    return {
        "xweather": {
            "transport": "streamable_http",
            "url": f"https://mcp.api.xweather.com/mcp?api_key={api_key}",
        }
    }


async def get_xweather_tools() -> list:
    """Return cached MCP tools, fetching from the server only on first call or TTL expiry.

    Uses an asyncio lock to prevent thundering herd when multiple parallel
    fetchers start simultaneously (fan-out).
    """
    global _tool_cache, _tool_cache_time

    if _tool_cache is not None and (time.monotonic() - _tool_cache_time) < _TOOL_CACHE_TTL:
        logger.debug("xweather_tools_cache_hit")
        return _tool_cache

    async with _tool_cache_lock:
        # Double-check after acquiring lock (another coroutine may have refreshed)
        if _tool_cache is not None and (time.monotonic() - _tool_cache_time) < _TOOL_CACHE_TTL:
            return _tool_cache

        logger.info("xweather_tools_loading")
        config = get_xweather_mcp_config()
        client = MultiServerMCPClient(config)
        _tool_cache = await client.get_tools()
        _tool_cache_time = time.monotonic()
        logger.info("xweather_tools_cached", tool_count=len(_tool_cache))
        return _tool_cache


def invalidate_tool_cache() -> None:
    """Force a cache refresh on next call. Useful for testing."""
    global _tool_cache, _tool_cache_time
    _tool_cache = None
    _tool_cache_time = 0.0
