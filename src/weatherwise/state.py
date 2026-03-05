"""WeatherWise state definitions with parallel-safe reducers."""

from __future__ import annotations

import operator
from typing import Any, Annotated, Literal

from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


# ---------------------------------------------------------------------------
# Custom reducer: merges dicts from parallel branches instead of overwriting
# ---------------------------------------------------------------------------
def merge_dicts(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Merge two dicts. Right values win on key conflicts."""
    return {**left, **right}


# ---------------------------------------------------------------------------
# Main graph state — shared across all nodes
# ---------------------------------------------------------------------------
class WeatherState(TypedDict):
    """Unified state for the WeatherWise multi-agent graph.

    Every field updated by parallel workers uses an Annotated reducer
    to prevent race conditions during fan-in.
    """

    # Chat history — add_messages handles dedup by message ID
    messages: Annotated[list[BaseMessage], add_messages]

    # The plan: list of task dicts produced by the planner.
    # Set once by the planner (overwrite semantics — no reducer).
    plan: list[dict[str, Any]]

    # Tracks which plan indices have been executed.
    # Workers append their index on completion; executor checks coverage.
    completed_indices: Annotated[list[int], operator.add]

    # Weather API results keyed by location name.
    # Parallel fetchers write different keys; merge_dicts combines them.
    fetched_weather: Annotated[dict[str, Any], merge_dicts]

    # RAG retrieval results — parallel retrievers append context strings.
    retrieved_knowledge: Annotated[list[str], operator.add]

    # Math/calculation results — parallel calculators append result strings.
    calculations: Annotated[list[str], operator.add]

    # Routing decision set by the supervisor.
    # "clear" = proceed to planner, "ambiguous" = ask user for clarification.
    routing: Literal["clear", "ambiguous"]

    # Intent classification from the supervisor (e.g., "real_time", "math", "greeting").
    intent: str


# ---------------------------------------------------------------------------
# Worker payloads — what each Send-spawned worker receives
# ---------------------------------------------------------------------------
class FetchPayload(TypedDict):
    """Payload for fetcher and retriever workers."""

    plan_index: int
    op: str  # "fetch" or "research"
    location: str
    query: str


class CalcPayload(TypedDict):
    """Payload for calculator workers — includes fetched data for computation."""

    plan_index: int
    op: str  # "calculate"
    location: str
    query: str
    fetched_weather: dict[str, Any]  # all weather data collected so far
