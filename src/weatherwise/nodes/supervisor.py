"""Supervisor node — classifies intent, detects ambiguity, and routes the query.

Three-way classification: weather queries proceed to the planner,
greetings and off-topic queries get a direct LLM response.
If a weather query is too ambiguous to produce useful results,
the supervisor asks a clarifying question before proceeding.
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
from weatherwise.utils import trim_chat_history

logger = get_logger(__name__)

SYSTEM_PROMPT = """\
You are an intent classifier and query validator for a weather assistant called WeatherWise.
Analyze the user's message, classify it, and check if it's specific enough to proceed.

## Buckets

1. **weather** — Any query related to weather, climate, forecasts, historical weather, air quality, alerts, storms, outdoor activity planning, or calculations involving weather data.

2. **greeting** — Greetings, introductions, or questions about what WeatherWise can do.

3. **off_topic** — Queries clearly unrelated to weather, climate, or outdoor conditions.

## Rules
- If a query could plausibly relate to weather, climate, outdoor conditions or similar, choose "weather" — not "off_topic".
- Consider the conversation history when classifying. A follow-up message that refers to a previous weather discussion is still a weather query, even if it doesn't explicitly mention weather.
- For "greeting" and "off_topic" buckets, you MUST populate the `response` field:
  - **greeting**: Introduce yourself as WeatherWise, a professional AI weather intelligence system. Briefly describe your capabilities. Be friendly and invite the user to ask a weather question.
  - **off_topic**: Politely explain that you specialize in weather and redirect the user.
- For "weather" queries, check if the query is specific enough to fetch useful data. \
Set needs_clarification to true ONLY when the query is genuinely impossible to act on — \
for example, a location name that maps to multiple places with no way to determine which \
one from context (e.g., "Springfield" with no state), or a query with no location at all \
when one is clearly needed. Do NOT ask for clarification when the user mentions a region, \
area, or geographic feature (e.g., "the Great Lakes", "the Gulf Coast", "Pacific Northwest", \
"Michigan shoreline") — the planner can select representative cities for regional queries. \
When in doubt, proceed rather than asking — the system handles broad queries well.
"""

_DIRECT_RESPONSE_BUCKETS = {"greeting", "off_topic"}


class RouteDecision(BaseModel):
    """Structured output from the supervisor LLM."""

    bucket: Literal["weather", "greeting", "off_topic"] = Field(
        description="The intent bucket for this query."
    )
    reasoning: str = Field(description="Brief explanation of why this bucket was chosen.")
    needs_clarification: bool = Field(
        default=False,
        description="True only if the weather query is too ambiguous to proceed without user input.",
    )
    response: str | None = Field(
        default=None,
        description="For greeting/off_topic: the reply. For weather with needs_clarification: the clarifying question.",
    )


# Cached LLM instance
_supervisor_llm: ChatAnthropic | None = None


def _get_supervisor_llm() -> ChatAnthropic:
    global _supervisor_llm
    if _supervisor_llm is None:
        _supervisor_llm = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
            max_tokens=1024,
            temperature=0,
            timeout=30.0,
        )
    return _supervisor_llm


async def supervisor(state: WeatherState) -> dict[str, Any]:
    """Classify the user's latest message and validate query clarity.

    For weather intents: proceeds to planner (routing="clear").
    For ambiguous weather intents: asks clarifying question (routing="ambiguous").
    For greeting/off_topic: generates direct response (routing="ambiguous").
    On failure: defaults to "clear" to avoid blocking.
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
                + trim_chat_history(state["messages"])
            )

        decision: RouteDecision = await _classify()
        intent = decision.bucket

        logger.info("supervisor_result", intent=intent,
                     reasoning=decision.reasoning,
                     needs_clarification=decision.needs_clarification)

        # Greeting / off-topic: respond directly, skip pipeline
        if intent in _DIRECT_RESPONSE_BUCKETS:
            return {
                "messages": [AIMessage(content=decision.response or "")],
                "routing": "ambiguous",
                "intent": intent,
            }

        # Weather but needs clarification: ask and skip pipeline
        if decision.needs_clarification and decision.response:
            return {
                "messages": [AIMessage(content=decision.response)],
                "routing": "ambiguous",
                "intent": intent,
            }

        # Weather intent, clear enough: proceed to planner
        return {
            "routing": "clear",
            "intent": intent,
        }

    except Exception as e:
        logger.error("supervisor_failed", error=str(e))
        return {"routing": "clear", "intent": "weather"}
