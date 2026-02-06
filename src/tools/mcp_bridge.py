"""Bridge between the Xweather MCP Server and LangChain tools.

Connects to the remote Xweather MCP endpoint via streamable HTTP transport
using MultiServerMCPClient. Auth is passed as a query parameter.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()


def _build_server_config() -> dict:
    """Build the Xweather MCP connection config with query-param auth."""
    base_url = os.getenv("XWEATHER_MCP_URL")
    if not base_url:
        raise RuntimeError(
            "XWEATHER_MCP_URL is not set in .env. "
            "Example: XWEATHER_MCP_URL=https://mcp.api.xweather.com/mcp"
        )

    client_id = os.environ["XWEATHER_CLIENT_ID"]
    client_secret = os.environ["XWEATHER_CLIENT_SECRET"]
    api_key = f"{client_id}_{client_secret}"

    url = f"{base_url}?api_key={api_key}"

    return {
        "transport": "streamable_http",
        "url": url,
    }


def get_mcp_client() -> MultiServerMCPClient:
    """Create a MultiServerMCPClient configured for the Xweather MCP server.

    Usage::

        client = get_mcp_client()
        tools = await client.get_tools()
    """
    server_config = _build_server_config()
    return MultiServerMCPClient({"xweather": server_config})
