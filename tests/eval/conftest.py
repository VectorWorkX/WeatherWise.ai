"""Shared pytest fixtures and skip markers for evaluation tests."""

from __future__ import annotations

import os

import pytest

requires_anthropic = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)

requires_google = pytest.mark.skipif(
    not os.environ.get("GOOGLE_API_KEY"),
    reason="GOOGLE_API_KEY not set",
)

requires_langsmith = pytest.mark.skipif(
    not os.environ.get("LANGCHAIN_API_KEY"),
    reason="LANGCHAIN_API_KEY not set",
)

requires_openai = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)

requires_mcp = pytest.mark.skipif(
    not os.environ.get("XWEATHER_CLIENT_ID"),
    reason="Xweather MCP credentials not set",
)
