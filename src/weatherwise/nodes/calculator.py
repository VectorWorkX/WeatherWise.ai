"""Calculator agent node — performs math operations using fetched weather data.

The LLM decides which math tools to call based on the data and query.
Tools: compute_expression (arithmetic) and calculate_anomaly (historical comparison).
"""

from __future__ import annotations

import os
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from weatherwise.logging import get_logger
from weatherwise.state import CalcPayload
from weatherwise.tools.math_ops import compute_expression, calculate_anomaly

logger = get_logger(__name__)

CALC_SYSTEM_PROMPT = """You are a weather data calculator. You have been given fetched weather data and a calculation request.

You have access to tools:
- compute_expression: for arithmetic (e.g., "28.5 - 25.0", "(72 - 32) * 5/9")
- calculate_anomaly: for comparing a current value against historical averages

Your job:
1. Extract the relevant numeric values from the provided weather data
2. Use the appropriate tool to perform the calculation
3. Return a clear result with the numbers shown

Always use the tools — do NOT do mental math."""

_calc_tools = [compute_expression, calculate_anomaly]

_calc_llm: ChatAnthropic | None = None


def _get_calc_llm() -> ChatAnthropic:
    global _calc_llm
    if _calc_llm is None:
        _calc_llm = ChatAnthropic(
            model="claude-haiku-4-5-20251001",
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
            max_tokens=512,
            temperature=0,
            timeout=30.0,
        )
    return _calc_llm


async def calculator_agent(state: CalcPayload) -> dict[str, Any]:
    """Perform a calculation using fetched weather data from state.

    Receives a CalcPayload (via Send) which includes fetched_weather
    collected in phase 1. Uses tool-calling so the LLM decides how to compute.
    """
    query = state["query"]
    location = state["location"]
    plan_index = state["plan_index"]
    fetched_weather = state.get("fetched_weather", {})

    logger.info("calculator_start", query=query, location=location, plan_index=plan_index)

    try:
        llm = _get_calc_llm().bind_tools(_calc_tools)

        data_summary = ""
        for loc, data in fetched_weather.items():
            data_summary += f"\n{loc}: {data}"

        messages = [
            SystemMessage(content=CALC_SYSTEM_PROMPT),
            HumanMessage(
                content=f"Calculation request: {query}\n"
                        f"Location: {location}\n"
                        f"Available weather data:{data_summary}"
            ),
        ]

        # ReAct loop: let the LLM call tools until it produces a final answer
        max_iterations = 5
        for _ in range(max_iterations):
            response = await llm.ainvoke(messages)
            messages.append(response)

            if not response.tool_calls:
                # LLM produced a final text answer
                break

            # Execute tool calls and feed results back
            from langchain_core.messages import ToolMessage
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]

                if tool_name == "compute_expression":
                    result = compute_expression.invoke(tool_args)
                elif tool_name == "calculate_anomaly":
                    result = calculate_anomaly.invoke(tool_args)
                else:
                    result = f"Unknown tool: {tool_name}"

                messages.append(ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call["id"],
                ))

        # Extract the final text response
        result = response.content if isinstance(response.content, str) else str(response.content)
        logger.info("calculator_success", plan_index=plan_index)

    except Exception as e:
        logger.error("calculator_failed", error=str(e))
        result = f"Calculation failed for '{query}': {type(e).__name__}: {e}"

    return {
        "calculations": [result],
        "completed_indices": [plan_index],
    }
