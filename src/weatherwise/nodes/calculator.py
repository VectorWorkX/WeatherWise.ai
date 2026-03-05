"""Calculator agent node — performs math operations using fetched weather data."""

from __future__ import annotations

import os
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from weatherwise.logging import get_logger
from weatherwise.retry import llm_retry
from weatherwise.state import CalcPayload
from weatherwise.tools.math_ops import compute_expression

logger = get_logger(__name__)

CALC_SYSTEM_PROMPT = """You are a weather data calculator. You have been given fetched weather data and a calculation request.

Your job:
1. Extract the relevant numeric values from the provided weather data
2. Perform the requested calculation
3. Return a clear, concise result with the numbers shown

For anomaly calculations: compare current value against historical average, report the difference.
For comparisons: extract values for both locations, compute and report the difference.

Always show your work: current value, baseline/comparison value, and the result."""


async def calculator_agent(state: CalcPayload) -> dict[str, Any]:
    """Perform a calculation using fetched weather data from state.

    Receives a CalcPayload (via Send) which includes fetched_weather
    collected in phase 1. This runs in phase 2, after all fetchers complete.
    """
    query = state["query"]
    location = state["location"]
    plan_index = state["plan_index"]
    fetched_weather = state.get("fetched_weather", {})

    logger.info("calculator_start", query=query, location=location, plan_index=plan_index)

    try:
        # Pure arithmetic — no LLM needed
        if all(c in "0123456789.+-*/() " for c in query):
            result = compute_expression(query)
        else:
            llm = ChatAnthropic(
                model="claude-haiku-4-5-20251001",
                api_key=os.environ.get("ANTHROPIC_API_KEY"),
                max_tokens=512,
                temperature=0,
                timeout=30.0,
            )

            data_summary = ""
            for loc, data in fetched_weather.items():
                data_summary += f"\n{loc}: {data}"

            @llm_retry()
            async def _calculate():
                return await llm.ainvoke([
                    SystemMessage(content=CALC_SYSTEM_PROMPT),
                    HumanMessage(
                        content=f"Calculation request: {query}\n"
                                f"Location: {location}\n"
                                f"Available weather data:{data_summary}"
                    ),
                ])

            response = await _calculate()
            result = response.content

        logger.info("calculator_success", plan_index=plan_index)

    except Exception as e:
        logger.error("calculator_failed", error=str(e))
        result = f"Calculation failed for '{query}': {type(e).__name__}: {e}"

    return {
        "calculations": [result],
        "completed_indices": [plan_index],
    }
