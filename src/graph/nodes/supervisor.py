"""Supervisor node — classifies intent and routes to the correct specialist agent.

Also handles non-weather intents (greetings, off-topic) by generating a
natural-language response via the LLM — no hardcoded keyword lists or
static refusal messages.

Uses Haiku for fast, cheap classification and only sends the last message
(not full history) to minimise token usage.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.graph import END

from graph.prompts import SUPERVISOR_SYSTEM_PROMPT
from graph.state import AgentState, RouteDecision

load_dotenv()

# Maps weather intent buckets → specialist agent node names
AGENT_MAP: dict[str, str] = {
    "real_time": "forecast_agent",
    "comparative": "forecast_agent",
    "lifestyle": "forecast_agent",
    "safety": "forecast_agent",
    "explanation": "research_agent",
    "math": "analyst_agent",
}

# Buckets that don't need a specialist agent — the supervisor handles them
_DIRECT_RESPONSE_BUCKETS = {"greeting", "off_topic"}

_supervisor_llm: ChatAnthropic | None = None


def _get_supervisor_llm() -> ChatAnthropic:
    """Return a cached Haiku instance for fast intent classification."""
    global _supervisor_llm
    if _supervisor_llm is None:
        _supervisor_llm = ChatAnthropic(
            model="claude-haiku-4-5-20251001",
            api_key=os.environ["ANTHROPIC_API_KEY"],
            max_tokens=512,
            temperature=0,
        )
    return _supervisor_llm


async def supervisor_node(state: AgentState) -> dict:
    """Classify the user's latest message and set routing fields.

    For weather-related buckets, sets routing state so ``route_to_agent``
    dispatches to the correct specialist agent.

    For greeting / off_topic buckets, the LLM generates a response directly
    (via the ``response`` field in ``RouteDecision``), which is injected as
    an ``AIMessage`` so the graph can end without invoking an agent.
    """
    now = datetime.now(timezone.utc)
    local_now = datetime.now()
    date_ctx = (
        f"Current date and time: {local_now.strftime('%A, %B %d, %Y at %I:%M %p')} local, "
        f"{now.strftime('%Y-%m-%dT%H:%M:%SZ')} UTC."
    )
    prompt = f"{date_ctx}\n\n{SUPERVISOR_SYSTEM_PROMPT}"

    llm = _get_supervisor_llm().with_structured_output(RouteDecision)
    system_msg = SystemMessage(
        content=prompt,
        additional_kwargs={"cache_control": {"type": "ephemeral"}},
    )

    # Only send the last message for classification
    last_message = state["messages"][-1:]
    decision: RouteDecision = await llm.ainvoke([system_msg] + last_message)

    # Greeting / off-topic: the LLM already generated the reply
    if decision.bucket in _DIRECT_RESPONSE_BUCKETS:
        return {
            "intent": decision.bucket,
            "route_decision": decision,
            "messages": [AIMessage(content=decision.response or "")],
        }

    agent = AGENT_MAP.get(decision.bucket, "forecast_agent")

    return {
        "intent": decision.bucket,
        "route_decision": decision,
        "agent_name": agent,
    }


def route_to_agent(state: AgentState) -> str:
    """Conditional edge: route to the correct specialist agent.

    Mapping:
        real_time / comparative / lifestyle / safety → forecast_agent
        explanation → research_agent
        math → analyst_agent
        greeting / off_topic → END (response already in state)
    """
    intent = state.get("intent", "")
    if intent in _DIRECT_RESPONSE_BUCKETS:
        return END
    return state.get("agent_name", "forecast_agent")
