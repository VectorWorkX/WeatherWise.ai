"""Retry wrapper with exponential backoff for MCP tool calls.

Wraps async tool coroutines with automatic retry logic to handle
transient network failures when calling the Xweather MCP server.
"""

from __future__ import annotations

import asyncio
import logging
from functools import wraps
from typing import Sequence

from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)

# Retry configuration
MAX_RETRIES = 3
BACKOFF_BASE = 1.0  # seconds: 1s, 2s, 4s


def with_retry(
    tools: Sequence[BaseTool],
    *,
    max_retries: int = MAX_RETRIES,
    backoff_base: float = BACKOFF_BASE,
) -> list[BaseTool]:
    """Wrap each tool's coroutine with retry + exponential backoff.

    Args:
        tools: MCP tools from ``client.get_tools()``.
        max_retries: Maximum number of retry attempts (default 3).
        backoff_base: Base delay in seconds, doubled each retry (default 1.0).

    Returns:
        The same tool list with patched coroutines.
    """
    for tool in tools:
        original_ainvoke = tool.ainvoke

        @wraps(original_ainvoke)
        async def _retrying_ainvoke(
            input_data,
            *args,
            _orig=original_ainvoke,
            _name=tool.name,
            **kwargs,
        ):
            last_exc = None
            for attempt in range(max_retries + 1):
                try:
                    return await _orig(input_data, *args, **kwargs)
                except Exception as exc:
                    last_exc = exc
                    if attempt < max_retries:
                        delay = backoff_base * (2 ** attempt)
                        logger.warning(
                            "Tool '%s' failed (attempt %d/%d): %s. "
                            "Retrying in %.1fs...",
                            _name,
                            attempt + 1,
                            max_retries + 1,
                            exc,
                            delay,
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            "Tool '%s' failed after %d attempts: %s",
                            _name,
                            max_retries + 1,
                            exc,
                        )
            raise last_exc  # type: ignore[misc]

        # Bypass Pydantic v2 field validation — ainvoke is a method, not a model field
        object.__setattr__(tool, "ainvoke", _retrying_ainvoke)

    return list(tools)
