"""FastAPI + WebSocket server for WeatherWise.ai."""

from __future__ import annotations

import os
import uuid
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from langchain_core.messages import HumanMessage

from weatherwise.graph import build_graph
from weatherwise.logging import setup_logging, get_logger
from weatherwise.tracing import setup_tracing

load_dotenv()
setup_tracing()

setup_logging()
logger = get_logger(__name__)

app = FastAPI(title="WeatherWise.ai", version="0.1.0")

_graph = build_graph(
    checkpointer=os.environ.get("CHECKPOINTER_MODE", "memory"),
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/query")
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
        if hasattr(msg, "type") and msg.type == "ai":
            response = msg.content
            break

    return {"response": response, "thread_id": thread_id}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """WebSocket endpoint for streaming responses."""
    await websocket.accept()
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    logger.info("ws_connected", thread_id=thread_id)

    try:
        while True:
            user_query = await websocket.receive_text()
            logger.info("ws_query", query=user_query, thread_id=thread_id)

            await websocket.send_json({"type": "start", "thread_id": thread_id})

            async for event in _graph.astream_events(
                {"messages": [HumanMessage(content=user_query)]},
                config=config,
                version="v2",
            ):
                kind = event["event"]
                if kind == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    if hasattr(chunk, "content") and chunk.content:
                        await websocket.send_json({
                            "type": "token",
                            "content": chunk.content,
                        })

            await websocket.send_json({"type": "end"})

    except WebSocketDisconnect:
        logger.info("ws_disconnected", thread_id=thread_id)
