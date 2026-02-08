"""Chainlit UI for the WeatherWise.ai LangGraph agent.

Launch with::

    chainlit run app.py
"""

from __future__ import annotations

import os
import sys
import uuid
from typing import Any

sys.path.insert(0, "src")

import chainlit as cl  # noqa: E402
from langchain_core.messages import (  # noqa: E402
    AIMessage,
    HumanMessage,
    ToolMessage,
)

from graph.builder import build_graph  # noqa: E402
from tools.math_tools import calculator  # noqa: E402
from tools.mcp_bridge import get_mcp_client  # noqa: E402
from tools.rag_tool import search_knowledge_base  # noqa: E402
from utils.tracing import get_run_config, is_tracing_enabled, validate_tracing_config  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INTENT_LABELS: dict[str, str] = {
    "real_time": "Real-Time Forecast",
    "comparative": "Location Comparison",
    "lifestyle": "Lifestyle Advisory",
    "safety": "Safety & Alerts",
    "explanation": "Atmospheric Science",
    "math": "Weather Math",
}

LANGSMITH_BASE_URL = "https://smith.langchain.com"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _truncate(text: str, max_len: int = 1500) -> str:
    """Truncate long text for display."""
    if len(text) > max_len:
        return text[:max_len] + "\n\n*... truncated ...*"
    return text


def _format_tool_args(args: dict[str, Any]) -> str:
    """Format tool call arguments for step display."""
    parts = []
    for k, v in args.items():
        val = str(v)
        if len(val) > 300:
            val = val[:300] + "..."
        parts.append(f"**{k}:** {val}")
    return "\n".join(parts) if parts else str(args)


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------


@cl.on_chat_start
async def on_chat_start():
    """Initialize MCP tools, build graph, and store in session."""
    init_msg = cl.Message(content="Connecting to weather services...")
    await init_msg.send()

    try:
        # Validate LangSmith
        trace_info = validate_tracing_config()
        tracing_on = trace_info["tracing_enabled"] == "True"

        # Load MCP weather tools
        client = get_mcp_client()
        mcp_tools = await client.get_tools()

        # Combine all tools
        all_tools = mcp_tools + [calculator, search_knowledge_base]

        # Build graph
        graph = build_graph(all_tools)

        # Store in session
        cl.user_session.set("graph", graph)
        cl.user_session.set("message_history", [])

        # Ready message
        tracing_note = " | [LangSmith](https://smith.langchain.com) tracing active" if tracing_on else ""
        init_msg.content = (
            f"**WeatherWise.ai** ready — "
            f"{len(mcp_tools)} weather + {len(all_tools) - len(mcp_tools)} utility tools loaded"
            f"{tracing_note}"
        )
        await init_msg.update()

    except Exception as e:
        init_msg.content = f"**Initialization failed:** {e}"
        await init_msg.update()


# ---------------------------------------------------------------------------
# Message handling
# ---------------------------------------------------------------------------


@cl.on_message
async def on_message(message: cl.Message):
    """Process user messages through the LangGraph agent."""
    graph = cl.user_session.get("graph")
    message_history: list = cl.user_session.get("message_history")

    if graph is None:
        await cl.Message(content="Agent not initialized. Please refresh the page.").send()
        return

    # Append user message to history
    human_msg = HumanMessage(content=message.content)
    message_history.append(human_msg)

    graph_input = {
        "messages": list(message_history),
        "intent": "",
    }

    # LangSmith config
    run_id = uuid.uuid4()
    config = get_run_config(
        run_name="chainlit_chat",
        tags=["chainlit"],
        metadata={"session_id": cl.context.session.id},
    )
    config["run_id"] = run_id

    # Show thinking indicator
    thinking_msg = cl.Message(content="*Analyzing your query...*")
    await thinking_msg.send()

    try:
        # Invoke the full graph
        result = await graph.ainvoke(graph_input, config=config)

        # Remove thinking message
        await thinking_msg.remove()

        # Extract intent
        intent = result.get("intent", "")
        intent_label = INTENT_LABELS.get(intent, intent)

        # Show intent classification as a step
        async with cl.Step(name="Query Classification", type="run") as step:
            step.output = f"**{intent_label}** (`{intent}`)"

        # Show tool calls as collapsible steps
        all_messages = result["messages"]
        for msg in all_messages:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    async with cl.Step(name=tc["name"], type="tool") as step:
                        step.input = _format_tool_args(tc.get("args", {}))
                        # Find the matching ToolMessage response
                        tc_id = tc.get("id", "")
                        for resp in all_messages:
                            if isinstance(resp, ToolMessage) and resp.tool_call_id == tc_id:
                                step.output = _truncate(str(resp.content))
                                break

        # Get the final AI response (last message)
        final_response = all_messages[-1].content if all_messages else "No response generated."

        # Send the fully formatted response
        await cl.Message(content=final_response).send()

        # Update conversation history
        message_history.append(AIMessage(content=final_response))

        # LangSmith trace link
        if is_tracing_enabled():
            await cl.Message(
                content=f"*[View trace in LangSmith]({LANGSMITH_BASE_URL})* | Run: `{str(run_id)[:8]}`",
                author="system",
            ).send()

    except Exception as e:
        await thinking_msg.remove()
        await cl.Message(content=f"**Error:** {e}").send()
