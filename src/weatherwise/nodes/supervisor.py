"""Supervisor node — classifies query as clear or ambiguous using a fast model."""

from __future__ import annotations

import os
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, SystemMessage

from weatherwise.logging import get_logger
from weatherwise.retry import llm_retry
from weatherwise.state import WeatherState

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are a query classifier for a weather assistant.
Your ONLY job is to decide if the user's query is clear or ambiguous.

A query is AMBIGUOUS if:
- A city name could refer to multiple places (e.g., "Paris" could be Paris, France or Paris, Texas)
- The query is too vague to act on (e.g., "weather" with no location)
- Critical information is missing

A query is CLEAR if:
- The location(s) are unambiguous or well-known (e.g., "New York", "Tokyo", "London")
- The intent is understandable even if complex

Respond with EXACTLY one word: CLEAR or AMBIGUOUS
If AMBIGUOUS, add a second line with a short clarifying question."""


async def supervisor(state: WeatherState) -> dict[str, Any]:
    """Classify the user query as clear or ambiguous.

    Uses Claude Haiku for speed/cost. If ambiguous, returns an AIMessage
    asking for clarification and sets routing to "ambiguous".
    On failure, defaults to "clear" to avoid blocking the pipeline.
    """
    logger.info("supervisor_start")

    try:
        llm = ChatAnthropic(
            model="claude-haiku-4-5-20251001",
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
            max_tokens=100,
            temperature=0,
            timeout=30.0,
        )

        @llm_retry()
        async def _classify():
            return await llm.ainvoke(
                [SystemMessage(content=SYSTEM_PROMPT)] + list(state["messages"])
            )

        response = await _classify()
        text = response.content.strip()

        if text.upper().startswith("AMBIGUOUS"):
            clarification = text.split("\n", 1)[1].strip() if "\n" in text else (
                "Could you clarify your request? I need more specific details."
            )
            logger.info("supervisor_result", routing="ambiguous")
            return {
                "messages": [AIMessage(content=clarification)],
                "routing": "ambiguous",
            }

        logger.info("supervisor_result", routing="clear")
        return {"routing": "clear"}

    except Exception as e:
        logger.error("supervisor_failed", error=str(e))
        # Default to clear — let the pipeline proceed rather than silently fail
        return {"routing": "clear"}
