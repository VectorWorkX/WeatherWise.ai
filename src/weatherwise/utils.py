"""Shared utilities for WeatherWise nodes."""

from __future__ import annotations

from langchain_core.messages import BaseMessage, trim_messages

_DEFAULT_HISTORY_TOKEN_BUDGET: int = 4_000


def trim_chat_history(
    messages: list[BaseMessage],
    *,
    max_tokens: int = _DEFAULT_HISTORY_TOKEN_BUDGET,
) -> list[BaseMessage]:
    """Trim chat history to fit within a token budget.

    Uses an approximate token counter (fast, suitable for hot path)
    and keeps the most recent messages, ensuring history starts on
    a human message for valid LLM input.

    System messages are excluded because calling nodes prepend
    their system prompt separately.
    """
    if not messages:
        return []

    return trim_messages(
        messages,
        max_tokens=max_tokens,
        token_counter="approximate",
        strategy="last",
        start_on="human",
        include_system=False,
    )
