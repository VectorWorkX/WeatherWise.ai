"""Fetcher agent node — retrieves weather data via Xweather MCP server."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.agents import create_agent

from weatherwise.logging import get_logger
from weatherwise.retry import llm_retry
from weatherwise.state import FetchPayload
from weatherwise.tools.xweather_mcp import get_xweather_tools

logger = get_logger(__name__)

FETCHER_SYSTEM_PROMPT = """You are a weather data fetcher. Today's date is {today_date}.

Your job is to use the available tools to get the requested weather information.

Rules:
- Read the descriptions of all available tools and pick the most appropriate one(s) for the request
- You may call multiple tools if the request requires different types of data
- When dates are mentioned, use them exactly. When no dates are given and the query is about current conditions, use today's date ({today_date})
- If a tool call fails, try an alternative tool that could provide similar information
- Return the raw data you receive — do not interpret or summarize it"""


@llm_retry(max_attempts=3)
async def _invoke_fetch_agent(tools: list, prompt: str) -> dict:
    """Run the ReAct agent with retries on transient failures (429, timeouts)."""
    llm = ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
        max_tokens=1024,
        temperature=0,
        timeout=45.0,
    )
    today = datetime.now().strftime("%Y-%m-%d")
    system_prompt = FETCHER_SYSTEM_PROMPT.format(today_date=today)
    agent = create_agent(llm, tools)
    return await agent.ainvoke(
        {"messages": [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt),
        ]}
    )


async def fetcher_agent(state: FetchPayload) -> dict[str, Any]:
    """Fetch weather data for the location specified in the task payload.

    Receives a FetchPayload (via Send), not the full WeatherState.
    Uses the Xweather MCP server via a ReAct agent to fetch real data.
    Returns partial state updates that merge into WeatherState via reducers.
    On failure, returns an error message as the raw_response (graceful degradation).
    """
    location = state["location"]
    query = state.get("query", "")
    plan_index = state["plan_index"]

    logger.info("fetcher_start", location=location, plan_index=plan_index)

    try:
        tools = await get_xweather_tools()

        if query:
            prompt = f"{query} for {location}" if location else query
        else:
            prompt = f"Get weather information for {location}"

        result = await _invoke_fetch_agent(tools, prompt)

        # Extract the final AI message content
        raw_response = "No data returned from Xweather"
        for msg in reversed(result.get("messages", [])):
            if hasattr(msg, "type") and msg.type == "ai" and msg.content:
                raw_response = msg.content
                break

        logger.info("fetcher_success", location=location)

    except Exception as e:
        logger.error("fetcher_failed", location=location, error=str(e))
        raw_response = (
            f"Weather data unavailable for {location}: {type(e).__name__}: {e}"
        )

    # Use plan_index in key to prevent data loss when multiple fetches
    # target the same location (e.g., current + historical for Santos)
    key = f"{location.lower().strip()}_{plan_index}"
    return {
        "fetched_weather": {key: {
            "location": location,
            "query": query,
            "raw_response": raw_response,
        }},
        "completed_indices": [plan_index],
    }
