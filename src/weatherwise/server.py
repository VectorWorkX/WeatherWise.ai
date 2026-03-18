"""FastAPI + WebSocket server for WeatherWise.ai."""

from __future__ import annotations

import json
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, HumanMessage

from weatherwise.graph import build_graph
from weatherwise.logging import setup_logging, get_logger
from weatherwise.tools.xweather_mcp import get_xweather_tools
from weatherwise.tracing import setup_tracing

load_dotenv()
setup_tracing()

setup_logging()
logger = get_logger(__name__)

_graph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create the graph and pre-warm external caches.

    For Redis, AsyncRedisSaver must be created inside an async context
    (its __init__ calls asyncio.get_running_loop()). The lifespan context
    manager is the correct place for this in FastAPI.

    Pre-warms the Xweather MCP tool cache so the first user query doesn't
    pay the ~2-3s schema negotiation cost. The cache lives in the xweather_mcp
    module with a 5-minute TTL — subsequent fetcher calls hit it directly.
    """
    global _graph
    mode = os.environ.get("CHECKPOINTER_MODE", "memory")

    if mode == "redis":
        from langgraph.checkpoint.redis.aio import AsyncRedisSaver

        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
        logger.info("redis_checkpointer_init", redis_url=redis_url)
        async with AsyncRedisSaver.from_conn_string(
            redis_url,
            ttl={"default_ttl": 1440, "refresh_on_read": True},
        ) as saver:
            _graph = build_graph(checkpointer_instance=saver)
            logger.info("graph_ready", checkpointer="redis")
            await _prewarm_caches()
            yield
    else:
        _graph = build_graph(checkpointer=mode)
        logger.info("graph_ready", checkpointer=mode)
        await _prewarm_caches()
        yield


async def _prewarm_caches() -> None:
    """Pre-warm external tool caches at startup.

    Called during lifespan before the server accepts requests.
    Failures are logged but do not block startup — the fetcher's
    own retry logic will handle MCP negotiation on first use.
    """
    try:
        tools = await get_xweather_tools()
        logger.info("mcp_cache_prewarmed", tool_count=len(tools))
    except Exception as e:
        logger.warning("mcp_cache_prewarm_failed", error=str(e))


app = FastAPI(title="WeatherWise.ai", version="0.2.0", lifespan=lifespan)

_cors_origins = [
    "https://weather-wise.ai",
    "https://www.weather-wise.ai",
    "https://weatherwise-2i5llicxea-uc.a.run.app",
    "https://weatherwise-228097686588.us-central1.run.app",
]
if os.environ.get("CORS_ORIGIN"):
    _cors_origins.append(os.environ["CORS_ORIGIN"])
if os.environ.get("ENVIRONMENT") != "production":
    _cors_origins.extend(["http://localhost:5173", "http://localhost:5174"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/query")
async def query(payload: dict[str, Any]) -> dict[str, Any]:
    """Synchronous (non-streaming) query endpoint."""
    user_query = payload.get("query", "")
    thread_id = payload.get("thread_id", str(uuid.uuid4()))

    if not user_query:
        return {"error": "Missing 'query' field"}

    logger.info("api_query", query=user_query, thread_id=thread_id)
    config = {"configurable": {"thread_id": thread_id}}

    result = await _graph.ainvoke(
        {"messages": [HumanMessage(content=user_query)]},
        config=config,
    )

    response = ""
    for msg in reversed(result.get("messages", [])):
        if isinstance(msg, AIMessage) and msg.content:
            response = msg.content if isinstance(msg.content, str) else str(msg.content)
            break

    return {"response": response, "thread_id": thread_id}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _truncate(text: str, max_len: int = 1500) -> str:
    if len(text) > max_len:
        return text[:max_len] + " ... (truncated)"
    return text


def _safe_str(obj: Any) -> str:
    try:
        return str(obj)
    except Exception:
        return "<unserializable>"


def _extract_suggestions(text: str) -> tuple[str, list[str]]:
    """Extract >>>-prefixed follow-up suggestions from the end of a response.

    Returns (cleaned_text, suggestions_list).
    """
    lines = text.rstrip().split("\n")
    suggestions: list[str] = []
    # Collect >>> lines from the end
    while lines and lines[-1].strip().startswith(">>>"):
        suggestion = lines.pop().strip().removeprefix(">>>").strip()
        if suggestion:
            suggestions.insert(0, suggestion)
    cleaned = "\n".join(lines).rstrip()
    return cleaned, suggestions


def _safe_json(obj: Any) -> Any:
    """Recursively convert an object to a JSON-safe structure."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_json(v) for v in obj]
    return str(obj)


def _extract_text(content: Any) -> str:
    """Extract text from a chunk content that may be str or list of blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block if isinstance(block, str) else block.get("text", "")
            for block in content
        )
    return ""


# ---------------------------------------------------------------------------
# WebSocket chat endpoint
# ---------------------------------------------------------------------------


@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """Bidirectional streaming chat over WebSocket.

    Client sends::
        {"type": "user_message", "content": "What's the temp in SF?"}

    Server sends::
        {"type": "tool_call",   "name": "...", "args": {...}, "call_id": "..."}
        {"type": "tool_result", "name": "...", "content": "...", "call_id": "..."}
        {"type": "token",       "content": "..."}
        {"type": "done",        "full_content": "..."}
        {"type": "error",       "message": "..."}
    """
    await websocket.accept()
    session_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id}}

    logger.info("ws_connected", thread_id=session_id)
    await websocket.send_json({"type": "connected", "session_id": session_id})

    try:
        while True:
            data = await websocket.receive_json()

            if data.get("type") != "user_message":
                continue

            content = data.get("content", "").strip()
            if not content:
                continue

            logger.info("ws_query", query=content, thread_id=session_id)

            run_id = uuid.uuid4()
            run_config = {**config, "run_id": run_id}
            full_response = ""
            intent_sent = False

            # Intent label mapping for the frontend
            intent_labels = {
                "weather": "Weather Analysis",
                "greeting": "Greeting",
                "off_topic": "Off-Topic",
            }

            try:
                # Token batching: accumulate tokens and flush in chunks
                # to reduce WebSocket messages and React re-renders.
                # Instead of 500+ individual sends, this produces ~30-50.
                token_buffer = ""
                _TOKEN_FLUSH_SIZE = 30  # flush every ~30 characters

                async for event in _graph.astream_events(
                    {"messages": [HumanMessage(content=content)]},
                    config=run_config,
                    version="v2",
                ):
                    kind = event.get("event", "")

                    # --- Intent classification (from supervisor) ---
                    if (
                        kind == "on_chain_end"
                        and event.get("name") == "supervisor"
                        and not intent_sent
                    ):
                        output = event.get("data", {}).get("output", {})
                        intent = output.get("intent", "") if isinstance(output, dict) else ""
                        if intent:
                            await websocket.send_json({
                                "type": "intent",
                                "intent": intent,
                                "label": intent_labels.get(intent, intent),
                                "reasoning": output.get("reasoning", "") if isinstance(output, dict) else "",
                            })
                            intent_sent = True

                    # --- Planner started ---
                    elif kind == "on_chain_start" and event.get("name") == "planner":
                        await websocket.send_json({
                            "type": "status",
                            "step": "planning",
                        })

                    # --- Planner finished — send plan summary and task descriptions ---
                    elif kind == "on_chain_end" and event.get("name") == "planner":
                        output = event.get("data", {}).get("output", {})
                        if isinstance(output, dict):
                            plan_tasks = output.get("plan", [])
                            await websocket.send_json({
                                "type": "status",
                                "step": "plan_ready",
                                "summary": output.get("summary", ""),
                                "tasks": [
                                    {
                                        "op": t.get("op", ""),
                                        "query": t.get("query", ""),
                                        "location": t.get("location", ""),
                                        "description": t.get("description", ""),
                                    }
                                    for t in plan_tasks
                                ],
                            })

                    # --- Synthesizer started ---
                    elif kind == "on_chain_start" and event.get("name") == "synthesizer":
                        await websocket.send_json({
                            "type": "status",
                            "step": "synthesizing",
                        })

                    # --- Tool call started ---
                    elif kind == "on_tool_start":
                        tool_input = event.get("data", {}).get("input", {})
                        await websocket.send_json({
                            "type": "tool_call",
                            "name": event.get("name", "unknown"),
                            "args": _safe_json(tool_input),
                            "call_id": str(event.get("run_id", "")),
                        })

                    # --- Tool call finished ---
                    elif kind == "on_tool_end":
                        tool_output = event.get("data", {}).get("output", "")
                        await websocket.send_json({
                            "type": "tool_result",
                            "name": event.get("name", "unknown"),
                            "content": _truncate(_safe_str(tool_output)),
                            "call_id": str(event.get("run_id", "")),
                        })

                    # --- LLM token streaming (synthesizer only) ---
                    elif kind == "on_chat_model_stream":
                        tags = event.get("tags", [])
                        is_synthesizer = any("synthesizer" in t for t in tags)
                        if is_synthesizer:
                            chunk = event.get("data", {}).get("chunk")
                            if chunk and hasattr(chunk, "content") and chunk.content:
                                # Skip tool-call chunks
                                if hasattr(chunk, "tool_calls") and chunk.tool_calls:
                                    continue
                                text = _extract_text(chunk.content)
                                if text:
                                    token_buffer += text
                                    full_response += text
                                    if len(token_buffer) >= _TOKEN_FLUSH_SIZE:
                                        await websocket.send_json({
                                            "type": "token",
                                            "content": token_buffer,
                                        })
                                        token_buffer = ""

                # Flush remaining buffered tokens
                if token_buffer:
                    await websocket.send_json({
                        "type": "token",
                        "content": token_buffer,
                    })
                    token_buffer = ""

                # Fallback: if streaming captured no tokens, extract from state
                if not full_response:
                    graph_state = await _graph.aget_state(run_config)
                    if graph_state.values:
                        msgs = graph_state.values.get("messages", [])
                        for msg in reversed(msgs):
                            if isinstance(msg, AIMessage) and msg.content:
                                full_response = _extract_text(msg.content)
                                break

                # Extract follow-up suggestions from the response
                cleaned_response, suggestions = _extract_suggestions(full_response)

                await websocket.send_json({
                    "type": "done",
                    "full_content": cleaned_response,
                })

                # Send follow-up suggestions if any were generated
                if suggestions:
                    await websocket.send_json({
                        "type": "suggestions",
                        "suggestions": suggestions,
                    })

                # Send visual report if one was generated
                try:
                    graph_state = await _graph.aget_state(run_config)
                    if graph_state.values:
                        viz = graph_state.values.get("visual_report", {})
                        if viz and (viz.get("sections") or viz.get("error")):
                            await websocket.send_json({
                                "type": "visual_report",
                                "data": _safe_json(viz),
                            })
                except Exception as ve:
                    logger.warning("visual_report_send_failed", error=str(ve))

            except Exception as e:
                logger.error("ws_error", error=str(e), thread_id=session_id)
                await websocket.send_json({"type": "error", "message": str(e)})

    except WebSocketDisconnect:
        logger.info("ws_disconnected", thread_id=session_id)


# ---------------------------------------------------------------------------
# Serve frontend build (production) — only when SERVE_FRONTEND=1
# ---------------------------------------------------------------------------

if os.environ.get("SERVE_FRONTEND") == "1":
    _frontend_dist = Path(os.environ.get("FRONTEND_DIST_DIR", "frontend/dist")).resolve()
    if not _frontend_dist.exists():
        # Fallback: relative to source file (dev mode)
        _frontend_dist = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"
    if _frontend_dist.exists():
        app.mount("/", StaticFiles(directory=str(_frontend_dist), html=True), name="frontend")
