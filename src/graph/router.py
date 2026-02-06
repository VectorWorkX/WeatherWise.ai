"""Intent router — classifies user queries into one of 6 buckets.

Uses structured output so the LLM returns a validated RouteDecision
instead of free-form text.
"""

from __future__ import annotations

import os
from typing import Literal

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field

from graph.state import AgentState

load_dotenv()

ROUTER_SYSTEM_PROMPT = """\
You are an intent classifier for a weather assistant called WeatherWise.
Analyze the user's latest message and assign it to exactly ONE bucket.

## Buckets

1. **real_time** — Direct weather lookups for a single location.
   Example: "What is the temp in San Jose?"

2. **comparative** — Comparing weather between two or more locations.
   Example: "Is it hotter in Phoenix or Miami right now?"

3. **lifestyle** — Advisory questions about activities, comfort, or planning.
   Example: "Is today good for a wedding in Napa?"

4. **safety** — Severe weather alerts, warnings, or emergency conditions.
   Example: "Are there any active tornado warnings in my county?"

5. **explanation** — "Why" questions requiring atmospheric science or deeper reasoning.
   Example: "Why is there a heat dome over the Midwest?"

6. **math** — Queries that require fetching weather data AND performing calculations.
   Example: "Add the temperature of Boston and New York and give the answer in Kelvin."

## Rules
- Choose the MOST specific bucket that fits.
- If a query combines fetching data AND math operations, choose "math".
- If a query asks "why" about a weather phenomenon, choose "explanation".
- If a query mentions alerts, warnings, or emergencies, choose "safety".
"""


class RouteDecision(BaseModel):
    """Structured output from the router LLM."""

    bucket: Literal[
        "real_time", "comparative", "lifestyle", "safety", "explanation", "math"
    ] = Field(description="The intent bucket for this query.")
    reasoning: str = Field(description="Brief explanation of why this bucket was chosen.")


def _get_router_llm() -> ChatAnthropic:
    """Create a lightweight LLM instance for routing."""
    return ChatAnthropic(
        model="claude-sonnet-4-20250514",
        api_key=os.environ["ANTHROPIC_API_KEY"],
        max_tokens=256,
        temperature=0,
    )


def analyze_intent(state: AgentState) -> dict:
    """Router node: classify the user's latest message into a bucket.

    Returns a state update with the ``intent`` field set.
    """
    llm = _get_router_llm().with_structured_output(RouteDecision)
    decision: RouteDecision = llm.invoke(
        [SystemMessage(content=ROUTER_SYSTEM_PROMPT)] + state["messages"]
    )
    return {"intent": decision.bucket}
