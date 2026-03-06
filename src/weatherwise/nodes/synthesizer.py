"""Synthesizer node — weaves all gathered data into the final response."""

from __future__ import annotations

import os
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

from weatherwise.logging import get_logger
from weatherwise.retry import llm_retry
from weatherwise.state import WeatherState

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are WeatherWise.ai, an expert weather analyst.
You have been given weather data, research context, and calculation results.
Your job is to synthesize everything into a clear, insightful response.

Guidelines:
- Lead with the direct answer to the user's question
- Use markdown: **bold** for key values and locations, headers for sections
- Include specific numbers and data points with units
- If research context is available, weave in the "why" behind the data
- If calculations were performed, explain what they mean
- Be concise but thorough — explain the science when relevant
- If data is missing or a tool returned no results, say so honestly"""

_synthesizer_llm: ChatAnthropic | None = None


def _get_synthesizer_llm() -> ChatAnthropic:
    global _synthesizer_llm
    if _synthesizer_llm is None:
        _synthesizer_llm = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
            max_tokens=1500,
            temperature=0.3,
            timeout=60.0,
        )
    return _synthesizer_llm


async def synthesizer(state: WeatherState) -> dict[str, Any]:
    """Read the full state and compose the final analyst-grade response."""
    logger.info("synthesizer_start")

    # Build a data summary for the LLM
    sections: list[str] = []

    weather = state.get("fetched_weather", {})
    if weather:
        lines = []
        for loc in sorted(weather.keys()):
            lines.append(f"  {loc}: {weather[loc]}")
        sections.append("WEATHER DATA:\n" + "\n".join(lines))

    knowledge = state.get("retrieved_knowledge", [])
    if knowledge:
        sections.append("RESEARCH CONTEXT:\n" + "\n".join(f"  - {k}" for k in knowledge))

    calcs = state.get("calculations", [])
    if calcs:
        sections.append("CALCULATIONS:\n" + "\n".join(f"  - {c}" for c in calcs))

    data_block = "\n\n".join(sections) if sections else "No data was gathered."

    # Get the original user query from messages
    user_query = ""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            user_query = msg.content
            break

    try:
        llm = _get_synthesizer_llm()

        @llm_retry()
        async def _synthesize():
            return await llm.ainvoke(
                [
                    SystemMessage(content=[
                        {"type": "text", "text": SYSTEM_PROMPT,
                         "cache_control": {"type": "ephemeral"}},
                    ]),
                    HumanMessage(content=[
                        {"type": "text",
                         "text": f"User query: {user_query}\n\nGathered data:\n{data_block}",
                         "cache_control": {"type": "ephemeral"}},
                    ]),
                ],
                config={"tags": ["synthesizer"]},
            )

        response = await _synthesize()
        content = response.content
        logger.info("synthesizer_success")

    except Exception as e:
        logger.error("synthesizer_failed", error=str(e))
        content = (
            f"I gathered the following data but encountered an error during synthesis:\n\n"
            f"{data_block}\n\n(Synthesis error: {type(e).__name__}: {e})"
        )

    return {"messages": [AIMessage(content=content)]}
