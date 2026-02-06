"""Graph state definition for the WeatherWise agent."""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Shared state passed between all nodes in the graph.

    Attributes:
        messages: Conversation history, managed by LangGraph's add_messages reducer.
        intent: The bucket classification assigned by the router
                (e.g. "real_time", "comparative", "explanation").
    """

    messages: Annotated[list[AnyMessage], add_messages]
    intent: str
