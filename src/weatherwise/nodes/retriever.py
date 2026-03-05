"""Retriever agent node — searches the knowledge base for relevant context."""

from __future__ import annotations

from typing import Any

from weatherwise.logging import get_logger
from weatherwise.state import FetchPayload
from weatherwise.tools.vector_search import search_knowledge_base

logger = get_logger(__name__)


async def retriever_agent(state: FetchPayload) -> dict[str, Any]:
    """Search the knowledge base for the query in the task payload.

    Receives a FetchPayload (via Send), not the full WeatherState.
    Returns partial state updates that merge into WeatherState via reducers.
    """
    query = state["query"]
    plan_index = state["plan_index"]

    logger.info("retriever_start", query=query, plan_index=plan_index)

    try:
        results = search_knowledge_base(query)
        logger.info("retriever_success", result_count=len(results))
    except Exception as e:
        logger.error("retriever_failed", error=str(e))
        results = [f"Knowledge base search failed: {type(e).__name__}: {e}"]

    return {
        "retrieved_knowledge": results,
        "completed_indices": [plan_index],
    }
