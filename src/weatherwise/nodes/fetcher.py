"""Fetcher agent node — retrieves weather data via Xweather MCP server."""

from __future__ import annotations

import os
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.agents import create_agent

from weatherwise.logging import get_logger
from weatherwise.retry import llm_retry
from weatherwise.state import FetchPayload
from weatherwise.tools.xweather_mcp import get_xweather_tools

logger = get_logger(__name__)

FETCHER_SYSTEM_PROMPT = """You are a weather data fetcher. Your job is to use the available Xweather tools to get the requested weather information for a specific location.

Rules:
- Call the most appropriate tool for the request
- For basic current weather, use xweather_get_current_weather
- For forecasts, use xweather_get_forecast_weather
- For air quality, use xweather_get_current_airquality
- For alerts, use xweather_get_current_active_weather_alerts
- Return the raw data you receive — do not interpret or summarize it
- If a tool call fails, report the error clearly"""


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
    agent = create_agent(llm, tools)
    return await agent.ainvoke(
        {"messages": [
            SystemMessage(content=FETCHER_SYSTEM_PROMPT),
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
            prompt = f"Get current weather conditions for {location}"

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

    return {
        "fetched_weather": {location.lower().strip(): {
            "location": location,
            "raw_response": raw_response,
        }},
        "completed_indices": [plan_index],
    }
