"""Planner node — decomposes a user query into independent sub-tasks."""

from __future__ import annotations

import os
from typing import Any, Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field

from weatherwise.logging import get_logger
from weatherwise.retry import llm_retry
from weatherwise.state import WeatherState

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are a task planner for a weather assistant.
Given a user query, decompose it into a list of independent sub-tasks.

Each task has these fields:
- "op": one of "fetch", "research", "calculate"
- "location": city name (for "fetch" tasks, empty string otherwise)
- "query": search query (for "research") or math description (for "calculate"), empty string for "fetch"

Rules:
- "fetch" = get current weather data for a specific city
- "research" = search the knowledge base for background info (port closures, historical events, etc.)
- "calculate" = perform a math operation (anomalies, comparisons, etc.) — these DEPEND on fetched data and will run after fetch tasks complete

Examples:

Query: "What's the weather in Tokyo?"
Tasks: [{"op": "fetch", "location": "Tokyo", "query": ""}]

Query: "Compare weather in New York and London"
Tasks: [{"op": "fetch", "location": "New York", "query": ""}, {"op": "fetch", "location": "London", "query": ""}]

Query: "Is Santos port at risk of closure due to wind? How does current wind compare to historical average?"
Tasks: [{"op": "fetch", "location": "Santos", "query": ""}, {"op": "research", "query": "Santos port wind closure thresholds", "location": ""}, {"op": "calculate", "query": "wind anomaly Santos", "location": "Santos"}]"""


class PlanTask(BaseModel):
    """A single sub-task in the execution plan."""

    op: Literal["fetch", "research", "calculate"]
    location: str = ""
    query: str = ""


class PlanOutput(BaseModel):
    """Structured output from the planner LLM."""

    tasks: list[PlanTask] = Field(min_length=1)


async def planner(state: WeatherState) -> dict[str, Any]:
    """Decompose the user query into a plan of independent sub-tasks."""
    logger.info("planner_start")

    try:
        llm = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
            max_tokens=500,
            temperature=0,
            timeout=60.0,
        )
        structured_llm = llm.with_structured_output(PlanOutput)

        @llm_retry()
        async def _plan():
            return await structured_llm.ainvoke(
                [SystemMessage(content=[
                    {"type": "text", "text": SYSTEM_PROMPT,
                     "cache_control": {"type": "ephemeral"}},
                ])] + list(state["messages"])
            )

        result = await _plan()
        plan = [task.model_dump() for task in result.tasks]
        logger.info("planner_result", task_count=len(plan), plan=plan)

    except Exception as e:
        logger.error("planner_failed", error=str(e))
        # Fallback: extract location from user message, attempt single fetch
        user_msg = ""
        for msg in reversed(state.get("messages", [])):
            if hasattr(msg, "type") and msg.type == "human":
                user_msg = msg.content
                break
        plan = [{"op": "fetch", "location": user_msg or "unknown", "query": ""}]

    return {"plan": plan, "completed_indices": []}
