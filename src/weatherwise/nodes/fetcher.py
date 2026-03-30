"""Fetcher agent node — retrieves weather data via Xweather MCP server.

Uses a manual tool-calling loop (same pattern as calculator_agent) instead of
create_agent. This removes the framework overhead of compiling an internal
StateGraph per invocation and gives direct control over iteration limits.
The LLM decides autonomously whether to retry with a different tool on error.
"""

from __future__ import annotations

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

FETCHER_SYSTEM_PROMPT = """You are a weather data fetcher. Today's date is {today_date}.

Your job is to use the available tools to get the requested weather information.

Rules:
- Carefully read the DESCRIPTIONS of all available tools, not just their names. The description tells you what data the tool actually provides and its limitations. Pick the tool whose description best matches the request.
- Call only ONE tool per request. Most tools return comprehensive data — do not call a second tool for data the first tool already provides
- Only call a second tool if the first tool's result is clearly missing critical data the user needs
- When dates are mentioned, use them exactly. When no dates are given and the query is about current conditions, use today's date ({today_date})
- If a tool call fails or returns zero records, try a different tool that could provide similar information
- Return the raw data you receive — do not interpret or summarize it
- Do NOT use a 'fields' parameter to filter the response. Return all available fields — \
downstream calculations may need fields not explicitly mentioned in the request."""

# Cached LLM instance — avoids re-creating HTTP clients on every parallel fetch
_fetcher_llm: ChatAnthropic | None = None


def _get_fetcher_llm() -> ChatAnthropic:
    global _fetcher_llm
    if _fetcher_llm is None:
        _fetcher_llm = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
            max_tokens=1024,
            temperature=0,
            timeout=45.0,
            max_retries=0,  # Disable SDK auto-retry; tenacity @llm_retry handles retries
        )
    return _fetcher_llm


def _extract_text(content: Any) -> str:
    """Extract clean text from content that may be a string or list of content blocks.

    The Anthropic API and MCP tools can return content as:
    - A plain string
    - A list of content blocks: [{"type": "text", "text": "..."}]

    str() on a list produces '[{"type": "text", "text": "..."}]' which is
    a Python repr — not parseable as JSON and not readable as text.
    This function extracts the actual text content in all cases.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return str(content)


@llm_retry(max_attempts=3)
async def _invoke_fetch_loop(tools: list, prompt: str) -> tuple[str, str]:
    """Manual tool-calling loop — LLM selects tools and decides when to stop.

    Returns (text_response, raw_tool_data):
    - text_response: LLM's text interpretation of the data (for synthesizer)
    - raw_tool_data: clean MCP tool response, typically structured JSON (for calculator/visualizer)

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

    last_tool_data = ""

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

            # Extract clean text from the tool result — avoids the
            # [{'type': 'text', 'text': '...'}] wrapper that str() produces
            clean_result = _extract_text(tool_result)
            last_tool_data = clean_result

            messages.append(ToolMessage(
                content=clean_result,
                tool_call_id=tool_call["id"],
            ))

    # Extract final text response — clean content blocks properly
    text_response = _extract_text(response.content)
    return text_response, last_tool_data


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

        raw_response, raw_data = await _invoke_fetch_loop(tools, prompt)

        logger.info("fetcher_success", location=location)

    except Exception as e:
        logger.error("fetcher_failed", location=location, error=str(e))
        raw_response = (
            f"Weather data unavailable for {location}: {type(e).__name__}: {e}"
        )
        raw_data = ""

    # Use plan_index in key to prevent data loss when multiple fetches
    # target the same location (e.g., current + historical for Santos)
    key = f"{location.lower().strip()}_{plan_index}"
    return {
        "fetched_weather": {key: {
            "location": location,
            "query": query,
            "raw_response": raw_response,
            "raw_data": raw_data,
        }},
        "completed_indices": [plan_index],
    }
