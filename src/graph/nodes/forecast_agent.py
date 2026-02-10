"""Forecast specialist agent — handles Buckets 1-4.

Handles real_time, comparative, lifestyle, and safety queries.
Tools: Xweather MCP only (no calculator, no RAG).
Internal ReAct loop: forecast_worker ⇄ ToolNode(mcp_tools).
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Sequence

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from graph.prompts import FORECAST_PROMPTS
from graph.state import AgentState

load_dotenv()

_forecast_llm: ChatAnthropic | None = None


def _get_forecast_llm() -> ChatAnthropic:
    """Return a cached Sonnet instance for forecast tasks."""
    global _forecast_llm
    if _forecast_llm is None:
        _forecast_llm = ChatAnthropic(
            model="claude-sonnet-4-5-20250929",
            api_key=os.environ["ANTHROPIC_API_KEY"],
            temperature=0,
            max_tokens=2048,
        )
    return _forecast_llm


def _date_context() -> str:
    now = datetime.now(timezone.utc)
    local_now = datetime.now()
    return (
        f"Current date and time: {local_now.strftime('%A, %B %d, %Y at %I:%M %p')} local, "
        f"{now.strftime('%Y-%m-%dT%H:%M:%SZ')} UTC."
    )


def build_forecast_subgraph(mcp_tools: Sequence[BaseTool]):
    """Build and compile the forecast agent subgraph.

    Returns a compiled graph suitable for ``parent_graph.add_node()``.
    """
    tools = list(mcp_tools)
    llm = _get_forecast_llm().bind_tools(tools)

    async def forecast_worker(state: AgentState) -> dict:
        intent = state.get("intent", "real_time")
        system_content = (
            f"{_date_context()}\n\n"
            f"{FORECAST_PROMPTS.get(intent, FORECAST_PROMPTS['real_time'])}"
        )
        system_msg = SystemMessage(
            content=system_content,
            additional_kwargs={"cache_control": {"type": "ephemeral"}},
        )
        response = await llm.ainvoke([system_msg] + state["messages"])
        return {"messages": [response]}

    def should_continue(state: AgentState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("forecast_worker", forecast_worker)
    graph.add_node("tools", ToolNode(tools, handle_tool_errors=True))

    graph.set_entry_point("forecast_worker")
    graph.add_conditional_edges("forecast_worker", should_continue, {
        "tools": "tools",
        END: END,
    })
    graph.add_edge("tools", "forecast_worker")

    return graph.compile()
