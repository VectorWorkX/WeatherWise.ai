"""Supervisor node — classifies intent and routes the query.

Classifies into 8 intent buckets. Weather intents proceed to the planner.
Greetings and off-topic queries get a direct LLM response without invoking
the planner/executor pipeline.
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

1. **real_time** — Direct weather lookups for a single location.
   Example: "What is the temp in San Jose?"

2. **comparative** — Comparing weather between two or more locations.
   Example: "Is it hotter in Phoenix or Miami right now?"

3. **lifestyle** — Advisory questions about activities or planning that depend on weather.
   Examples: "Is today good for a wedding in Napa?", "Can I take my kid to the park in San Jose?"

4. **safety** — Severe weather alerts, warnings, or emergency conditions.
   Example: "Are there any active tornado warnings in my county?"

5. **explanation** — "Why" questions requiring atmospheric science or deeper reasoning.
   Example: "Why is there a heat dome over the Midwest?"

6. **math** — Queries that require fetching weather data AND performing calculations.
   Example: "Add the temperature of Boston and New York and give the answer in Kelvin."

7. **greeting** — Greetings, introductions, or questions about what WeatherWise can do.
   Examples: "Hello", "Hi there!", "What can you do?"

8. **off_topic** — Queries clearly unrelated to weather, climate, or outdoor conditions.
   Examples: "What is the GDP of France?", "Write me a poem about cats."

## Rules
- Choose the MOST specific bucket that fits.
- If a query combines fetching data AND math operations, choose "math".
- If a query asks "why" about a weather phenomenon, choose "explanation".
- If a query mentions alerts, warnings, or emergencies, choose "safety".
- If a query could plausibly relate to outdoor conditions or activity planning, choose "lifestyle" — not "off_topic".
- For "greeting" and "off_topic" buckets, you MUST populate the `response` field:
  - **greeting**: Introduce yourself as WeatherWise, a professional AI weather intelligence system. Briefly describe your 6 capabilities. Be friendly and invite the user to ask a weather question.
  - **off_topic**: Politely explain that you specialize in weather and redirect the user.
- For all other buckets, leave the `response` field empty (null).
"""

# Buckets that don't need the planner/executor pipeline
_DIRECT_RESPONSE_BUCKETS = {"greeting", "off_topic"}


class RouteDecision(BaseModel):
    """Structured output from the supervisor LLM."""

    bucket: Literal[
        "real_time", "comparative", "lifestyle", "safety", "explanation", "math",
        "greeting", "off_topic",
    ] = Field(description="The intent bucket for this query.")
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
                + state["messages"][-1:]  # only last message for speed
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
        return {"routing": "clear", "intent": "real_time"}
