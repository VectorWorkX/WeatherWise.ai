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
from langgraph.checkpoint.memory import MemorySaver  # noqa: E402
from tools.math_tools import calculator  # noqa: E402
from tools.mcp_bridge import get_mcp_client  # noqa: E402
from tools.rag_tool import search_knowledge_base  # noqa: E402
from tools.retry_wrapper import with_retry  # noqa: E402
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
    "greeting": "Greeting",
    "off_topic": "Off-Topic",
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

        # Load MCP weather tools with retry wrapper
        client = get_mcp_client()
        mcp_tools = with_retry(await client.get_tools())

        # Build graph with MemorySaver checkpointing
        checkpointer = MemorySaver()
        graph = build_graph(
            mcp_tools, search_knowledge_base, calculator, checkpointer=checkpointer,
        )

        # Store in session
        cl.user_session.set("graph", graph)

        # Ready message
        tracing_note = " | [LangSmith](https://smith.langchain.com) tracing active" if tracing_on else ""
        init_msg.content = (
            f"**WeatherWise.ai** ready — "
            f"{len(mcp_tools)} weather + 2 utility tools loaded"
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

    if graph is None:
        await cl.Message(content="Agent not initialized. Please refresh the page.").send()
        return

    # MemorySaver manages conversation history via thread_id
    graph_input = {"messages": [HumanMessage(content=message.content)]}

    # LangSmith config
    session_id = cl.context.session.id
    run_id = uuid.uuid4()
    config = get_run_config(
        run_name="chainlit_chat",
        tags=["chainlit"],
        metadata={"session_id": session_id},
    )
    config["run_id"] = run_id
    config["configurable"] = {"thread_id": session_id}

    # Show thinking indicator
    thinking_msg = cl.Message(content="*Analyzing your query...*")
    await thinking_msg.send()

    try:
        full_response = ""
        intent_sent = False
        tool_steps: dict[str, cl.Step] = {}

        response_msg = cl.Message(content="")

        async for event in graph.astream_events(
            graph_input, config=config, version="v2"
        ):
            kind = event["event"]

            # --- Intent classification ---
            if (
                kind == "on_chain_end"
                and event.get("name") == "supervisor"
                and not intent_sent
            ):
                output = event.get("data", {}).get("output", {})
                intent = output.get("intent", "") if isinstance(output, dict) else ""
                if intent:
                    await thinking_msg.remove()
                    intent_label = INTENT_LABELS.get(intent, intent)
                    async with cl.Step(name="Query Classification", type="run") as step:
                        step.output = f"**{intent_label}** (`{intent}`)"
                    intent_sent = True

            # --- Tool call started ---
            elif kind == "on_tool_start":
                tool_name = event.get("name", "unknown")
                tool_input = event.get("data", {}).get("input", {})
                ts = cl.Step(name=tool_name, type="tool")
                ts.input = _format_tool_args(tool_input) if isinstance(tool_input, dict) else str(tool_input)
                await ts.send()
                tool_steps[str(event.get("run_id", ""))] = ts

            # --- Tool call finished ---
            elif kind == "on_tool_end":
                run_id_str = str(event.get("run_id", ""))
                ts = tool_steps.pop(run_id_str, None)
                if ts:
                    tool_output = event.get("data", {}).get("output", "")
                    ts.output = _truncate(str(tool_output))
                    await ts.update()

            # --- Synthesizer token streaming ---
            elif kind == "on_chat_model_stream":
                tags = event.get("tags", [])
                is_synthesizer = any("synthesizer" in t for t in tags)
                if is_synthesizer:
                    chunk = event.get("data", {}).get("chunk")
                    if chunk and hasattr(chunk, "content") and chunk.content:
                        if not (hasattr(chunk, "tool_calls") and chunk.tool_calls):
                            text = chunk.content if isinstance(chunk.content, str) else "".join(
                                block if isinstance(block, str) else block.get("text", "")
                                for block in chunk.content
                            )
                            if text:
                                full_response += text
                                await response_msg.stream_token(text)

        # Finalize the streamed response
        if not intent_sent:
            await thinking_msg.remove()

        response_msg.content = full_response or "No response generated."
        await response_msg.send()

        # LangSmith trace link
        if is_tracing_enabled():
            await cl.Message(
                content=f"*[View trace in LangSmith]({LANGSMITH_BASE_URL})* | Run: `{str(run_id)[:8]}`",
                author="system",
            ).send()

    except Exception as e:
        await thinking_msg.remove()
        await cl.Message(content=f"**Error:** {e}").send()
