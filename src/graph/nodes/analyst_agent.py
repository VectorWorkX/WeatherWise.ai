"""Analyst specialist agent — handles Bucket 6 (math).

Tools: Calculator + Xweather MCP (no RAG).
Internal ReAct loop: analyst_worker ⇄ ToolNode(mcp_tools + [calculator_tool]).
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

from graph.prompts import MATH_PROMPT
from graph.state import AgentState

load_dotenv()

_analyst_llm: ChatAnthropic | None = None


def _get_analyst_llm() -> ChatAnthropic:
    """Return a cached Sonnet instance for analyst tasks."""
    global _analyst_llm
    if _analyst_llm is None:
        _analyst_llm = ChatAnthropic(
            model="claude-sonnet-4-5-20250929",
            api_key=os.environ["ANTHROPIC_API_KEY"],
            temperature=0,
            max_tokens=2048,
        )
    return _analyst_llm


def _date_context() -> str:
    now = datetime.now(timezone.utc)
    local_now = datetime.now()
    return (
        f"Current date and time: {local_now.strftime('%A, %B %d, %Y at %I:%M %p')} local, "
        f"{now.strftime('%Y-%m-%dT%H:%M:%SZ')} UTC."
    )


def build_analyst_subgraph(
    mcp_tools: Sequence[BaseTool],
    calculator_tool: BaseTool,
):
    """Build and compile the analyst agent subgraph.

    Returns a compiled graph suitable for ``parent_graph.add_node()``.
    """
    tools = list(mcp_tools) + [calculator_tool]
    llm = _get_analyst_llm().bind_tools(tools)

    async def analyst_worker(state: AgentState) -> dict:
        system_content = f"{_date_context()}\n\n{MATH_PROMPT}"
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
    graph.add_node("analyst_worker", analyst_worker)
    graph.add_node("tools", ToolNode(tools, handle_tool_errors=True))

    graph.set_entry_point("analyst_worker")
    graph.add_conditional_edges("analyst_worker", should_continue, {
        "tools": "tools",
        END: END,
    })
    graph.add_edge("tools", "analyst_worker")

    return graph.compile()
