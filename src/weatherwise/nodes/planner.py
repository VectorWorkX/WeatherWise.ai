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
Given a user query, decompose it into sub-tasks with explicit dependencies.

IMPORTANT: Today's date is {today_date}. You MUST convert ALL relative dates in the user's query to absolute dates in the "query" field. For example:
- "last 3 days" → "daily weather summary from {three_days_ago} to {today_date}"
- "yesterday" → "weather for {yesterday}"
- "tomorrow" → "forecast for {tomorrow}"
- "this week" → "from {week_start} to {week_end}"
- "last time it rained" → "daily weather summaries with precipitation for the past 30 days from {thirty_days_ago} to {today_date}"

Each task has these fields:
- "op": one of "fetch", "research", "calculate", "visualize"
- "location": city or region name (empty string if not location-specific)
- "query": describes WHAT data to get — the specific intent behind the request. ALWAYS include absolute dates when the query involves time. Leave empty only for simple current weather lookups.
- "depends_on": list of task indices (0-based) that must complete before this task runs. Use [] for tasks with no dependencies. Tasks with the same dependencies will run in parallel.

Rules:
- "fetch" = get weather-related data for a location. The "query" field tells the fetcher WHAT to look for (forecast, historical data, air quality, alerts, tropical storms, hourly conditions, etc.). The fetcher has many tools and will pick the right one based on your query.
- "research" = search the knowledge base for background info (port closures, historical events, climate science, etc.)
- "calculate" = perform a math operation (anomalies, comparisons, unit conversions, statistics, etc.)
- "visualize" = generate a visual chart or graph. Only include when the user explicitly asks to see, plot, graph, chart, or visualize something. Always pair with fetch tasks via depends_on.
- Use "depends_on" to express which tasks must finish first. The executor uses this to determine parallel execution waves automatically — tasks with satisfied dependencies run in parallel.
- A "calculate" task that needs fetched data should list the relevant fetch task indices in "depends_on".
- A "calculate" task that needs results from another calculation should list that calculation's index in "depends_on". It will receive those prior results.

Examples:

Query: "What's the weather in Tokyo?"
Tasks: [{{"op": "fetch", "location": "Tokyo", "query": "", "depends_on": []}}]

Query: "5-day forecast for London"
Tasks: [{{"op": "fetch", "location": "London", "query": "5-day weather forecast starting {today_date}", "depends_on": []}}]

Query: "Compare weather in New York and London"
Tasks: [{{"op": "fetch", "location": "New York", "query": "", "depends_on": []}}, {{"op": "fetch", "location": "London", "query": "", "depends_on": []}}]

Query: "Is Santos port at risk of closure due to wind? How does current wind compare to historical average?"
Tasks: [{{"op": "fetch", "location": "Santos", "query": "", "depends_on": []}}, {{"op": "fetch", "location": "Santos", "query": "historical daily weather summaries for wind data from {thirty_days_ago} to {today_date}", "depends_on": []}}, {{"op": "research", "query": "Santos port wind closure thresholds", "location": "", "depends_on": []}}, {{"op": "calculate", "query": "wind anomaly: compare current wind to historical average for Santos", "location": "Santos", "depends_on": [0, 1]}}]

Query: "Add the temperatures of Boston and New York and convert to Kelvin"
Tasks: [{{"op": "fetch", "location": "Boston", "query": "", "depends_on": []}}, {{"op": "fetch", "location": "New York", "query": "", "depends_on": []}}, {{"op": "calculate", "query": "add temperatures of Boston and New York, convert sum to Kelvin", "location": "", "depends_on": [0, 1]}}]

Query: "Compare temperature deviation from mean for Delhi vs San Jose over the last month"
Tasks: [{{"op": "fetch", "location": "Delhi", "query": "daily weather summary from {thirty_days_ago} to {today_date} with temperature data", "depends_on": []}}, {{"op": "fetch", "location": "San Jose", "query": "daily weather summary from {thirty_days_ago} to {today_date} with temperature data", "depends_on": []}}, {{"op": "calculate", "location": "Delhi", "query": "calculate mean temperature and standard deviation for Delhi", "depends_on": [0]}}, {{"op": "calculate", "location": "San Jose", "query": "calculate mean temperature and standard deviation for San Jose", "depends_on": [1]}}, {{"op": "calculate", "location": "", "query": "compare temperature deviations: which city (Delhi or San Jose) has greater deviation from its mean", "depends_on": [2, 3]}}]

Query: "Generate a visual report of lightning strikes across the 5 largest Minneapolis suburbs"
Tasks: [{{"op": "fetch", "location": "Brooklyn Park", "query": "historical lightning strikes for the past month", "depends_on": []}}, {{"op": "fetch", "location": "Plymouth", "query": "historical lightning strikes for the past month", "depends_on": []}}, {{"op": "fetch", "location": "Maple Grove", "query": "historical lightning strikes for the past month", "depends_on": []}}, {{"op": "fetch", "location": "Woodbury", "query": "historical lightning strikes for the past month", "depends_on": []}}, {{"op": "fetch", "location": "Blaine", "query": "historical lightning strikes for the past month", "depends_on": []}}, {{"op": "visualize", "location": "", "query": "visual report of cloud-to-ground lightning strike counts across 5 Minneapolis suburbs with comparison charts and insights", "depends_on": [0, 1, 2, 3, 4]}}]

Query: "Compare Delhi and San Jose temperatures and show me a chart"
Tasks: [{{"op": "fetch", "location": "Delhi", "query": "", "depends_on": []}}, {{"op": "fetch", "location": "San Jose", "query": "", "depends_on": []}}, {{"op": "visualize", "location": "", "query": "visual comparison of current weather conditions between Delhi and San Jose with temperature and key metrics", "depends_on": [0, 1]}}]"""


class PlanTask(BaseModel):
    """A single sub-task in the execution plan."""

    op: Literal["fetch", "research", "calculate", "visualize"]
    location: str = ""
    query: str = ""
    depends_on: list[int] = Field(
        default_factory=list,
        description="Indices (0-based) of tasks that must complete before this one runs.",
    )
    description: str = Field(
        default="",
        description="A natural first-person sentence explaining what this step does and why, for display to the user",
    )


class PlanOutput(BaseModel):
    """Structured output from the planner LLM."""

    summary: str = Field(
        description="One natural sentence describing your overall approach to answering this query",
    )
    tasks: list[PlanTask] = Field(min_length=1)


# Cached LLM instance — avoids re-creating HTTP clients on every call
_planner_llm: ChatAnthropic | None = None


def _get_planner_llm() -> ChatAnthropic:
    global _planner_llm
    if _planner_llm is None:
        _planner_llm = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
            max_tokens=1024,
            temperature=0,
            timeout=45.0,
        )
    return _planner_llm


async def planner(state: WeatherState) -> dict[str, Any]:
    """Decompose the user query into a plan of independent sub-tasks."""
    logger.info("planner_start")

    try:
        structured_llm = _get_planner_llm().with_structured_output(PlanOutput)

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
            "week_start": (now - timedelta(days=now.weekday())).strftime("%Y-%m-%d"),
            "week_end": (now - timedelta(days=now.weekday()) + timedelta(days=6)).strftime("%Y-%m-%d"),
        }
        resolved_prompt = SYSTEM_PROMPT.format(**date_values)

        @llm_retry()
        async def _plan():
            return await structured_llm.ainvoke(
                [SystemMessage(content=[
                    {"type": "text", "text": resolved_prompt,
                     "cache_control": {"type": "ephemeral"}},
                ])] + list(state["messages"][-12:])
            )

        result = await _plan()
        plan = [task.model_dump() for task in result.tasks]
        summary = result.summary
        logger.info("planner_result", task_count=len(plan), summary=summary, plan=plan)

    except Exception as e:
        logger.error("planner_failed", error=str(e))
        # Fallback: extract location from user message, attempt single fetch
        user_msg = ""
        for msg in reversed(state.get("messages", [])):
            if hasattr(msg, "type") and msg.type == "human":
                user_msg = msg.content
                break
        plan = [{"op": "fetch", "location": user_msg or "unknown", "query": user_msg}]
        summary = ""

    # Reset all per-query state fields (empty list/dict triggers reset in reducers)
    return {
        "plan": plan,
        "summary": summary,
        "completed_indices": [],
        "fetched_weather": {},
        "retrieved_knowledge": [],
        "calculations": [],
        "visual_report": {},
    }
