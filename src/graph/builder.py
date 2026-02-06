"""Graph construction — wires router, worker, and tools into a compiled LangGraph."""

from __future__ import annotations

from typing import Sequence

from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from graph.nodes.worker import make_tool_invoker
from graph.router import analyze_intent
from graph.state import AgentState


def _route_by_intent(state: AgentState) -> str:
    """After analyze_intent: skip to END for 'explanation' (RAG not yet wired),
    otherwise proceed to the tool_invoker."""
    if state.get("intent") == "explanation":
        return END
    return "tool_invoker"


def _should_continue(state: AgentState) -> str:
    """After tool_invoker: if the LLM made tool calls, execute them;
    otherwise it's the final answer — go to END."""
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return END


def build_graph(tools: Sequence[BaseTool]):
    """Build and compile the WeatherWise agent graph.

    Structure:
        START → analyze_intent ─┬─ (explanation) → END
                                └─ (other)       → tool_invoker ─┬─ (tool_calls) → tools → tool_invoker
                                                                  └─ (no calls)   → END
    """
    graph = StateGraph(AgentState)

    # Nodes
    graph.add_node("analyze_intent", analyze_intent)
    graph.add_node("tool_invoker", make_tool_invoker(tools))
    graph.add_node("tools", ToolNode(tools, handle_tool_errors=True))

    # Edges
    graph.set_entry_point("analyze_intent")
    graph.add_conditional_edges("analyze_intent", _route_by_intent)
    graph.add_conditional_edges("tool_invoker", _should_continue)
    graph.add_edge("tools", "tool_invoker")

    return graph.compile()
