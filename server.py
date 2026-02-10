"""FastAPI WebSocket server for the WeatherWise.ai premium web UI.

Launch with::

    uvicorn server:app --reload --port 8000
"""

from __future__ import annotations

import os
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

sys.path.insert(0, "src")

from fastapi import FastAPI, WebSocket, WebSocketDisconnect  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402

from graph.builder import build_graph  # noqa: E402
from tools.math_tools import calculator  # noqa: E402
from tools.mcp_bridge import get_mcp_client  # noqa: E402
from tools.rag_tool import search_knowledge_base  # noqa: E402
from utils.tracing import get_run_config, is_tracing_enabled  # noqa: E402

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


# ---------------------------------------------------------------------------
# Lifespan — initialise MCP tools + graph once at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load MCP weather tools, build the LangGraph agent, and store on app state."""
    client = get_mcp_client()
    mcp_tools = await client.get_tools()
    all_tools = mcp_tools + [calculator, search_knowledge_base]

    app.state.graph = build_graph(all_tools)
    app.state.tool_count = len(all_tools)
    app.state.mcp_tool_count = len(mcp_tools)

    yield  # app is running


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="WeatherWise.ai", version="0.2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health():
    """Health check — confirms agent is initialized."""
    return {
        "status": "ok",
        "tools": app.state.tool_count,
        "mcp_tools": app.state.mcp_tool_count,
        "tracing": is_tracing_enabled(),
    }


@app.get("/api/capabilities")
async def capabilities():
    """Return the 6 capability buckets for the landing page."""
    return {
        "capabilities": [
            {
                "id": "real_time",
                "title": "Real-Time Forecast",
                "description": "Current conditions for any location worldwide",
                "example": "What is the wind speed and gust spread in San Francisco right now?",
            },
            {
                "id": "comparative",
                "title": "Location Comparison",
                "description": "Compare weather between two or more cities",
                "example": "How does the current winter storm severity in Buffalo compare to Boston's accumulation?",
            },
            {
                "id": "lifestyle",
                "title": "Lifestyle Advisory",
                "description": "Activity-based weather recommendations",
                "example": "I'm planning a ski trip to Colorado in February. Based on current snowpack and wind forecasts, which resort town should I target?",
            },
            {
                "id": "safety",
                "title": "Safety & Alerts",
                "description": "Severe weather warnings and safety guidance",
                "example": "My logistics fleet is traveling from Dallas to Denver between Feb 7th and 9th. Are there any severe weather advisories or icing risks along this route?",
            },
            {
                "id": "explanation",
                "title": "Atmospheric Science",
                "description": "Deep scientific explanations with RAG knowledge",
                "example": "Why did the 2023 Canadian wildfire smoke travel so far south into the US, and what atmospheric blocking patterns caused it?",
            },
            {
                "id": "math",
                "title": "Weather Math",
                "description": "Calculations and unit conversions on live data",
                "example": "Calculate the difference between Chicago's current wind chill and Miami's heat index in degrees Celsius.",
            },
        ]
    }


# ---------------------------------------------------------------------------
# In-memory session store
# ---------------------------------------------------------------------------

sessions: dict[str, list] = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _truncate(text: str, max_len: int = 1500) -> str:
    if len(text) > max_len:
        return text[:max_len] + " ... (truncated)"
    return text


def _safe_str(obj: Any) -> str:
    """Safely convert an object to string for JSON serialization."""
    try:
        return str(obj)
    except Exception:
        return "<unserializable>"


def _safe_json(obj: Any) -> Any:
    """Recursively convert an object to a JSON-safe structure."""
    if obj is None or isinstance(obj, (bool, int, float)):
        return obj
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        return {str(k): _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_json(v) for v in obj]
    return str(obj)


# ---------------------------------------------------------------------------
# WebSocket chat endpoint
# ---------------------------------------------------------------------------

@app.websocket("/ws/chat")
async def chat_ws(ws: WebSocket):
    """Bidirectional streaming chat over WebSocket.

    Client sends::

        {"type": "user_message", "content": "What's the temp in SF?"}

    Server sends a sequence of::

        {"type": "intent",       "intent": "...", "label": "..."}
        {"type": "tool_call",    "name": "...", "args": {...}, "call_id": "..."}
        {"type": "tool_result",  "name": "...", "content": "...", "call_id": "..."}
        {"type": "token",        "content": "..."}        # per token
        {"type": "done",         "full_content": "..."}
        {"type": "error",        "message": "..."}
    """
    await ws.accept()
    session_id = str(uuid.uuid4())
    sessions[session_id] = []

    await ws.send_json({"type": "connected", "session_id": session_id})

    try:
        while True:
            data = await ws.receive_json()

            if data.get("type") != "user_message":
                continue

            content = data.get("content", "").strip()
            if not content:
                continue

            history = sessions[session_id]
            human_msg = HumanMessage(content=content)
            history.append(human_msg)

            graph_input = {"messages": list(history), "intent": ""}

            run_id = uuid.uuid4()
            config = get_run_config(
                run_name="webui_chat",
                tags=["webui"],
                metadata={"session_id": session_id},
            )
            config["run_id"] = run_id

            full_response = ""
            intent_sent = False

            try:
                async for event in app.state.graph.astream_events(
                    graph_input, config=config, version="v2"
                ):
                    kind = event["event"]

                    # --- Intent classification ---
                    if (
                        kind == "on_chain_end"
                        and event.get("name") == "analyze_intent"
                        and not intent_sent
                    ):
                        output = event.get("data", {}).get("output", {})
                        intent = output.get("intent", "") if isinstance(output, dict) else ""
                        if intent:
                            await ws.send_json({
                                "type": "intent",
                                "intent": intent,
                                "label": INTENT_LABELS.get(intent, intent),
                            })
                            intent_sent = True

                    # --- Tool call started ---
                    elif kind == "on_tool_start":
                        tool_input = event.get("data", {}).get("input", {})
                        await ws.send_json({
                            "type": "tool_call",
                            "name": event.get("name", "unknown"),
                            "args": _safe_json(tool_input),
                            "call_id": str(event.get("run_id", "")),
                        })

                    # --- Tool call finished ---
                    elif kind == "on_tool_end":
                        tool_output = event.get("data", {}).get("output", "")
                        await ws.send_json({
                            "type": "tool_result",
                            "name": event.get("name", "unknown"),
                            "content": _truncate(_safe_str(tool_output)),
                            "call_id": str(event.get("run_id", "")),
                        })

                    # --- LLM token streaming ---
                    elif kind == "on_chat_model_stream":
                        chunk = event.get("data", {}).get("chunk")
                        if chunk and hasattr(chunk, "content") and chunk.content:
                            # Only stream tokens from the final answer,
                            # not from intermediate tool-calling turns
                            if not (hasattr(chunk, "tool_calls") and chunk.tool_calls):
                                # chunk.content can be str or list[str|dict]
                                text = chunk.content if isinstance(chunk.content, str) else "".join(
                                    block if isinstance(block, str) else block.get("text", "")
                                    for block in chunk.content
                                )
                                if text:
                                    full_response += text
                                    await ws.send_json({
                                        "type": "token",
                                        "content": text,
                                    })

                # Stream complete
                await ws.send_json({
                    "type": "done",
                    "full_content": full_response,
                    "run_id": str(run_id)[:8],
                })

                history.append(AIMessage(content=full_response))

            except Exception as e:
                await ws.send_json({"type": "error", "message": str(e)})

    except WebSocketDisconnect:
        sessions.pop(session_id, None)


# ---------------------------------------------------------------------------
# Serve frontend build (production)
# ---------------------------------------------------------------------------

_frontend_dist = Path(__file__).parent / "frontend" / "dist"
if _frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(_frontend_dist), html=True), name="frontend")
