"""Supervisor node — classifies intent and routes the query.

Three-way classification: weather queries proceed to the planner,
greetings and off-topic queries get a direct LLM response.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, SystemMessage
from pydantic import BaseModel, Field

from weatherwise.logging import get_logger
from weatherwise.retry import llm_retry
from weatherwise.state import WeatherState

logger = get_logger(__name__)

SYSTEM_PROMPT = """\
You are an intent classifier for a weather assistant called WeatherWise.
Analyze the user's message and assign it to exactly ONE bucket.

## Buckets

1. **weather** — Any query related to weather, climate, forecasts, historical weather, air quality, alerts, storms, outdoor activity planning, or calculations involving weather data.
   Examples: "What's the temp in Tokyo?", "5-day forecast for Denver", "Is it safe to go hiking tomorrow?", "Compare air quality in Delhi and Beijing", "What was the temperature on Jan 16, 2023?"

2. **greeting** — Greetings, introductions, or questions about what WeatherWise can do.
   Examples: "Hello", "Hi there!", "What can you do?"

3. **off_topic** — Queries clearly unrelated to weather, climate, or outdoor conditions.
   Examples: "What is the GDP of France?", "Write me a poem about cats."

## Rules
- If a query could plausibly relate to weather, climate, outdoor conditions or similar, choose "weather" — not "off_topic".
- For "greeting" and "off_topic" buckets, you MUST populate the `response` field:
  - **greeting**: Introduce yourself as WeatherWise, a professional AI weather intelligence system. Briefly describe your capabilities (current conditions, forecasts, historical data, air quality, severe weather alerts, tropical storms, calculations, and research). Be friendly and invite the user to ask a weather question.
  - **off_topic**: Politely explain that you specialize in weather and redirect the user.
- For "weather", leave the `response` field empty (null).
"""

# Buckets that don't need the planner/executor pipeline
_DIRECT_RESPONSE_BUCKETS = {"greeting", "off_topic"}


class RouteDecision(BaseModel):
    """Structured output from the supervisor LLM."""

    bucket: Literal["weather", "greeting", "off_topic"] = Field(
        description="The intent bucket for this query."
    )
    reasoning: str = Field(description="Brief explanation of why this bucket was chosen.")
    response: str | None = Field(
        default=None,
        description="For greeting/off_topic buckets only: the natural-language reply.",
    )


# Cached LLM instance
_supervisor_llm: ChatAnthropic | None = None


def _get_supervisor_llm() -> ChatAnthropic:
    global _supervisor_llm
    if _supervisor_llm is None:
        _supervisor_llm = ChatAnthropic(
            model="claude-haiku-4-5-20251001",
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
            max_tokens=512,
            temperature=0,
            timeout=30.0,
        )
    return _supervisor_llm


async def supervisor(state: WeatherState) -> dict[str, Any]:
    """Classify the user's latest message into an intent bucket.

    For weather intents, sets routing to "clear" so the planner runs.
    For greeting/off_topic, generates a direct response and sets routing
    to "ambiguous" (which routes to END, skipping the pipeline).
    On failure, defaults to "clear" / "real_time" to avoid blocking.
    """
    logger.info("supervisor_start")

    try:
        now = datetime.now(timezone.utc)
        local_now = datetime.now()
        date_ctx = (
            f"Current date and time: {local_now.strftime('%A, %B %d, %Y at %I:%M %p')} local, "
            f"{now.strftime('%Y-%m-%dT%H:%M:%SZ')} UTC."
        )

        llm = _get_supervisor_llm().with_structured_output(RouteDecision)

        @llm_retry()
        async def _classify():
            return await llm.ainvoke(
                [SystemMessage(content=[
                    {"type": "text", "text": f"{date_ctx}\n\n{SYSTEM_PROMPT}",
                     "cache_control": {"type": "ephemeral"}},
                ])]
                + state["messages"][-20:]  # recent history for follow-up context
            )

        decision: RouteDecision = await _classify()
        intent = decision.bucket

        logger.info("supervisor_result", intent=intent, reasoning=decision.reasoning)

        # Greeting / off-topic: respond directly, skip pipeline
        if intent in _DIRECT_RESPONSE_BUCKETS:
            return {
                "messages": [AIMessage(content=decision.response or "")],
                "routing": "ambiguous",  # routes to END
                "intent": intent,
            }

        # Weather intent: proceed to planner
        return {
            "routing": "clear",
            "intent": intent,
        }

    except Exception as e:
        logger.error("supervisor_failed", error=str(e))
        return {"routing": "clear", "intent": "weather"}
