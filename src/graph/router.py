"""Intent router — classifies user queries into one of 6 buckets.

DEPRECATED: Superseded by ``graph.nodes.supervisor`` (Phase 8).
Kept for reference only — not imported by the active graph.

Uses structured output so the LLM returns a validated RouteDecision
instead of free-form text.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage
from graph.state import AgentState, RouteDecision

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



_router_llm: ChatAnthropic | None = None


def _get_router_llm() -> ChatAnthropic:
    """Return a cached lightweight LLM instance for routing."""
    global _router_llm
    if _router_llm is None:
        _router_llm = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            api_key=os.environ["ANTHROPIC_API_KEY"],
            max_tokens=256,
            temperature=0,
        )
    return _router_llm


def analyze_intent(state: AgentState) -> dict:
    """Router node: classify the user's latest message into a bucket.

    Returns a state update with the ``intent`` field set.
    """
    now = datetime.now(timezone.utc)
    local_now = datetime.now()
    date_ctx = (
        f"Current date and time: {local_now.strftime('%A, %B %d, %Y at %I:%M %p')} local, "
        f"{now.strftime('%Y-%m-%dT%H:%M:%SZ')} UTC."
    )
    prompt = f"{date_ctx}\n\n{ROUTER_SYSTEM_PROMPT}"

    llm = _get_router_llm().with_structured_output(RouteDecision)
    system_msg = SystemMessage(
        content=prompt,
        additional_kwargs={"cache_control": {"type": "ephemeral"}},
    )
    decision: RouteDecision = llm.invoke([system_msg] + state["messages"])
    return {"intent": decision.bucket}
