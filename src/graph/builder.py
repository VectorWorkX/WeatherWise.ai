"""Graph construction — wires supervisor, specialist agents, and synthesizer.

Structure:
    START → supervisor → ┬─ forecast_agent → safety_review ─┐
                          ├─ research_agent ─────────────────├→ synthesizer → END
                          ├─ analyst_agent  ─────────────────┘
                          ├─ greeting  → END
                          └─ off_topic → END

Each agent is a compiled subgraph with its own internal ReAct tool loop.
The safety_review node uses interrupt() to pause for human approval on severe alerts.
The synthesizer formats the final response using Haiku.
The supervisor handles greetings and off-topic queries directly (LLM-generated response).
"""

from __future__ import annotations

from typing import Sequence

from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph

from graph.nodes.analyst_agent import build_analyst_subgraph
from graph.nodes.forecast_agent import build_forecast_subgraph
from graph.nodes.research_agent import build_research_subgraph
from graph.nodes.safety_review import safety_review_node
from graph.nodes.supervisor import route_to_agent, supervisor_node
from graph.nodes.synthesizer import synthesizer_node
from graph.state import AgentState


def build_graph(
    mcp_tools: Sequence[BaseTool],
    rag_tool: BaseTool,
    calculator_tool: BaseTool,
    *,
    checkpointer=None,
):
    """Build and compile the multi-agent WeatherWise graph.

    Args:
        mcp_tools: Xweather MCP tools for live weather data.
        rag_tool: The ``search_knowledge_base`` RAG retrieval tool.
        calculator_tool: The ``calculator`` math tool.
        checkpointer: Optional LangGraph checkpointer for memory persistence.
    """
    graph = StateGraph(AgentState)

    # --- Nodes ---
    graph.add_node("supervisor", supervisor_node)

    # Specialist agents (compiled subgraphs with internal tool loops)
    graph.add_node("forecast_agent", build_forecast_subgraph(list(mcp_tools)))
    graph.add_node("research_agent", build_research_subgraph(list(mcp_tools), rag_tool))
    graph.add_node("analyst_agent", build_analyst_subgraph(list(mcp_tools), calculator_tool))

    # Safety review (interrupt for severe alerts)
    graph.add_node("safety_review", safety_review_node)

    graph.add_node("synthesizer", synthesizer_node)

    # --- Edges ---
    graph.set_entry_point("supervisor")

    graph.add_conditional_edges("supervisor", route_to_agent, {
        "forecast_agent": "forecast_agent",
        "research_agent": "research_agent",
        "analyst_agent": "analyst_agent",
        END: END,
    })

    # forecast_agent → safety_review → synthesizer (may interrupt on severe alerts)
    graph.add_edge("forecast_agent", "safety_review")
    graph.add_edge("safety_review", "synthesizer")

    # research + analyst → synthesizer directly
    graph.add_edge("research_agent", "synthesizer")
    graph.add_edge("analyst_agent", "synthesizer")

    graph.add_edge("synthesizer", END)

    return graph.compile(checkpointer=checkpointer)
