"""Planner node — decomposes a user query into independent sub-tasks."""

from __future__ import annotations

import os
from datetime import datetime
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

IMPORTANT: Today's date is {today_date}. You MUST convert ALL relative dates in the user's query to absolute dates in the "query" field. For example:
- "last 3 days" → "daily weather summary from {three_days_ago} to {today_date}"
- "yesterday" → "weather for {yesterday}"
- "tomorrow" → "forecast for {tomorrow}"
- "this week" → "from {week_start} to {week_end}"
- "last time it rained" → "daily weather summaries with precipitation for the past 30 days from {thirty_days_ago} to {today_date}"

Each task has these fields:
- "op": one of "fetch", "research", "calculate"
- "location": city or region name (empty string if not location-specific)
- "query": describes WHAT data to get — the specific intent behind the request. ALWAYS include absolute dates when the query involves time. Leave empty only for simple current weather lookups.

Rules:
- "fetch" = get weather-related data for a location. The "query" field tells the fetcher WHAT to look for (forecast, historical data, air quality, alerts, tropical storms, hourly conditions, etc.). The fetcher has many tools and will pick the right one based on your query.
- "research" = search the knowledge base for background info (port closures, historical events, climate science, etc.)
- "calculate" = perform a math operation (anomalies, comparisons, unit conversions, etc.) — these DEPEND on fetched data and will run after fetch tasks complete

Sample Examples:

Query: "What's the weather in Tokyo?"
Tasks: [{{"op": "fetch", "location": "Tokyo", "query": ""}}]

Query: "What's the 5-day forecast for London?"
Tasks: [{{"op": "fetch", "location": "London", "query": "5-day weather forecast starting {today_date}"}}]

Query: "What was the temperature in San Jose on January 16, 2023?"
Tasks: [{{"op": "fetch", "location": "San Jose", "query": "daily weather summary for 2023-01-16"}}]

Query: "Did it rain in the last 3 days?"
Tasks: [{{"op": "fetch", "location": "San Jose", "query": "daily weather summary from {three_days_ago} to {today_date} with precipitation data"}}]

Query: "When was the last time it rained?"
Tasks: [{{"op": "fetch", "location": "San Jose", "query": "daily weather summaries with precipitation from {thirty_days_ago} to {today_date} to find last rain event"}}]

Query: "Will it rain in Seattle tomorrow?"
Tasks: [{{"op": "fetch", "location": "Seattle", "query": "forecast for {tomorrow} with precipitation timing and rain probability"}}]

Query: "Compare weather in New York and London"
Tasks: [{{"op": "fetch", "location": "New York", "query": ""}}, {{"op": "fetch", "location": "London", "query": ""}}]

Query: "What's the air quality in Delhi and how will it change this week?"
Tasks: [{{"op": "fetch", "location": "Delhi", "query": "current air quality"}}, {{"op": "fetch", "location": "Delhi", "query": "air quality forecast from {today_date} to {week_end}"}}]

Query: "Are there any active tropical storms near Florida?"
Tasks: [{{"op": "fetch", "location": "Florida", "query": "active tropical storms nearby"}}]

Query: "Can I play tennis at 4pm in San Jose today?"
Tasks: [{{"op": "fetch", "location": "San Jose", "query": "hourly weather conditions for {today_date} including rain probability"}}]

Query: "Are there any tornado warnings in Oklahoma?"
Tasks: [{{"op": "fetch", "location": "Oklahoma", "query": "active weather alerts and warnings"}}]

Query: "Is Santos port at risk of closure due to wind? How does current wind compare to historical average?"
Tasks: [{{"op": "fetch", "location": "Santos", "query": ""}}, {{"op": "fetch", "location": "Santos", "query": "historical daily weather summaries for wind data"}}, {{"op": "research", "query": "Santos port wind closure thresholds", "location": ""}}, {{"op": "calculate", "query": "wind anomaly: compare current wind to historical average for Santos", "location": "Santos"}}]

Query: "Add the temperatures of Boston and New York and convert to Kelvin"
Tasks: [{{"op": "fetch", "location": "Boston", "query": ""}}, {{"op": "fetch", "location": "New York", "query": ""}}, {{"op": "calculate", "query": "add temperatures of Boston and New York, convert sum to Kelvin", "location": ""}}]"""


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
            model="claude-haiku-4-5-20251001",
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
            max_tokens=500,
            temperature=0,
            timeout=30.0,
        )
        structured_llm = llm.with_structured_output(PlanOutput)

        # Resolve date placeholders so the LLM has absolute dates
        from datetime import timedelta
        now = datetime.now()
        today = now.strftime("%Y-%m-%d")
        date_values = {
            "today_date": today,
            "yesterday": (now - timedelta(days=1)).strftime("%Y-%m-%d"),
            "tomorrow": (now + timedelta(days=1)).strftime("%Y-%m-%d"),
            "three_days_ago": (now - timedelta(days=3)).strftime("%Y-%m-%d"),
            "thirty_days_ago": (now - timedelta(days=30)).strftime("%Y-%m-%d"),
            "week_start": now.strftime("%Y-%m-%d"),
            "week_end": (now + timedelta(days=7)).strftime("%Y-%m-%d"),
        }
        resolved_prompt = SYSTEM_PROMPT.format(**date_values)

        @llm_retry()
        async def _plan():
            return await structured_llm.ainvoke(
                [SystemMessage(content=[
                    {"type": "text", "text": resolved_prompt,
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
        plan = [{"op": "fetch", "location": user_msg or "unknown", "query": user_msg}]

    # Reset all per-query state fields (empty list/dict triggers reset in reducers)
    return {
        "plan": plan,
        "completed_indices": [],
        "fetched_weather": {},
        "retrieved_knowledge": [],
        "calculations": [],
    }
