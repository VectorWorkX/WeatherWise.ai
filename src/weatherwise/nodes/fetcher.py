"""Fetcher agent node — retrieves weather data via Xweather MCP server.

Uses a manual tool-calling loop (same pattern as calculator_agent) instead of
create_agent. This removes the framework overhead of compiling an internal
StateGraph per invocation and gives direct control over iteration limits.
The LLM decides autonomously whether to retry with a different tool on error.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from weatherwise.logging import get_logger
from weatherwise.retry import llm_retry
from weatherwise.state import FetchPayload
from weatherwise.tools.xweather_mcp import get_xweather_tools

logger = get_logger(__name__)

# Limit concurrent Haiku API calls to avoid blowing the 50K input tokens/min rate limit.
# With 4 parallel fetchers, unlimited concurrency sends ~32K tokens instantly.
# Semaphore(3) allows up to 3 concurrent fetchers. With prompt caching active
# (identical system prompts across parallel fetchers), cached tokens reduce
# effective token usage per call, giving headroom above the original estimate.
_FETCH_SEMAPHORE = asyncio.Semaphore(3)

FETCHER_SYSTEM_PROMPT = """You are a weather data fetcher. Today's date is {today_date}.

Your job is to use the available tools to get the requested weather information.

Rules:
- Read the descriptions of all available tools and pick the single most appropriate one for the request
- Call only ONE tool per request. Most tools return comprehensive data — do not call a second tool for data the first tool already provides
- Only call a second tool if the first tool's result is clearly missing critical data the user needs
- When dates are mentioned, use them exactly. When no dates are given and the query is about current conditions, use today's date ({today_date})
- If a tool call fails, try an alternative tool that could provide similar information
- Return the raw data you receive — do not interpret or summarize it"""

# Cached LLM instance — avoids re-creating HTTP clients on every parallel fetch
_fetcher_llm: ChatAnthropic | None = None


def _get_fetcher_llm() -> ChatAnthropic:
    global _fetcher_llm
    if _fetcher_llm is None:
        _fetcher_llm = ChatAnthropic(
            model="claude-haiku-4-5-20251001",
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
            max_tokens=1024,
            temperature=0,
            timeout=45.0,
            max_retries=0,  # Disable SDK auto-retry; tenacity @llm_retry handles retries
        )
    return _fetcher_llm


@llm_retry(max_attempts=3)
async def _invoke_fetch_loop(tools: list, prompt: str) -> str:
    """Manual tool-calling loop — LLM selects tools and decides when to stop.

    Same ReAct pattern as create_agent but without the internal StateGraph
    overhead. The LLM autonomously decides whether to call another tool or
    return a final text response based on the tool results it receives.

    Max 3 iterations (tool selection + up to 2 follow-ups for error recovery
    or supplementary data).
    """
    llm = _get_fetcher_llm()
    today = datetime.now().strftime("%Y-%m-%d")
    system_prompt = FETCHER_SYSTEM_PROMPT.format(today_date=today)
    tool_map = {t.name: t for t in tools}

    llm_with_tools = llm.bind_tools(tools)

    messages: list = [
        SystemMessage(content=[
            {"type": "text", "text": system_prompt,
             "cache_control": {"type": "ephemeral"}},
        ]),
        HumanMessage(content=prompt),
    ]

    max_iterations = 3
    for iteration in range(max_iterations):
        response = await llm_with_tools.ainvoke(messages)
        messages.append(response)

        if not response.tool_calls:
            # LLM produced a final text answer — done
            break

        # Execute tool calls
        for tool_call in response.tool_calls:
            tool_fn = tool_map.get(tool_call["name"])
            if tool_fn is None:
                tool_result = f"Unknown tool: {tool_call['name']}"
            else:
                try:
                    tool_result = await tool_fn.ainvoke(tool_call["args"])
                except Exception as e:
                    tool_result = f"Tool execution error: {type(e).__name__}: {e}"

            messages.append(ToolMessage(
                content=str(tool_result),
                tool_call_id=tool_call["id"],
            ))

    # Extract final text response
    content = response.content if isinstance(response.content, str) else str(response.content)
    return content


async def fetcher_agent(state: FetchPayload) -> dict[str, Any]:
    """Fetch weather data for the location specified in the task payload.

    Receives a FetchPayload (via Send), not the full WeatherState.
    Uses a manual tool-calling loop to fetch real data via Xweather MCP.
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

        async with _FETCH_SEMAPHORE:
            raw_response = await _invoke_fetch_loop(tools, prompt)

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
