"""Worker node — calls the LLM with tools bound, guided by the router's intent."""

from __future__ import annotations

import os
from typing import Sequence

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool

from graph.state import AgentState

load_dotenv()

INTENT_PROMPTS: dict[str, str] = {
    "real_time": (
        "You are a weather assistant. Use the weather tools to fetch current "
        "conditions for the requested location. Report temperature, conditions, "
        "and wind. Do not hallucinate data."
    ),
    "comparative": (
        "You are a weather assistant. The user wants to compare weather between "
        "multiple locations. Fetch data for each location, then clearly state "
        "which is higher/lower and by how much."
    ),
    "lifestyle": (
        "You are an advisory weather assistant. Fetch relevant weather data "
        "(temperature, precipitation, wind) and provide a helpful recommendation "
        "about the user's planned activity. Highlight any risks."
    ),
    "safety": (
        "You are an emergency weather assistant. Check for active weather alerts "
        "and warnings. Use a clear, urgent tone. Prioritize user safety."
    ),
    "explanation": (
        "You are a weather science expert. The user is asking 'why' about a "
        "weather phenomenon. Provide a scientific explanation. "
        "(RAG knowledge base will be added in a future phase.)"
    ),
    "math": (
        "You are a weather data analyst. Fetch the required weather data, then "
        "use the calculator tool to perform the requested math operations. "
        "Show your work: state the fetched values, the expression, and the result."
    ),
}


def make_tool_invoker(tools: Sequence[BaseTool]):
    """Create a tool_invoker node closed over the given tools.

    Returns a function suitable for ``graph.add_node()``.
    """
    llm = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        api_key=os.environ["ANTHROPIC_API_KEY"],
        temperature=0,
    ).bind_tools(tools)

    def tool_invoker(state: AgentState) -> dict:
        """Call the LLM with an intent-specific system prompt and bound tools."""
        intent = state.get("intent", "real_time")
        system_content = INTENT_PROMPTS.get(intent, INTENT_PROMPTS["real_time"])

        response = llm.invoke(
            [SystemMessage(content=system_content)] + state["messages"]
        )
        return {"messages": [response]}

    return tool_invoker
