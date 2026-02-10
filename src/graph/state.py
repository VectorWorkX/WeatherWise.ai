"""Graph state definition and shared models for the WeatherWise agent."""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class RouteDecision(BaseModel):
    """Structured output from the supervisor LLM."""

    bucket: Literal[
        "real_time", "comparative", "lifestyle", "safety", "explanation", "math",
        "greeting", "off_topic",
    ] = Field(description="The intent bucket for this query.")
    reasoning: str = Field(description="Brief explanation of why this bucket was chosen.")
    response: str | None = Field(
        default=None,
        description="For greeting/off_topic buckets only: the natural-language reply to send.",
    )


class AgentState(TypedDict):
    """Shared state passed between all nodes in the graph.

    Attributes:
        messages: Conversation history, managed by LangGraph's add_messages reducer.
        intent: The bucket classification assigned by the supervisor
                (e.g. "real_time", "comparative", "explanation", "rejected").
        route_decision: Full routing decision with reasoning from the supervisor.
        agent_name: Which specialist agent handles the query
                    (forecast_agent | research_agent | analyst_agent).
        requires_human_approval: Whether the query needs human approval (safety).
        human_approved: Whether the user approved a safety-critical response.
        rag_documents: Retrieved RAG documents for Corrective RAG grading.
        rag_grade: RAG relevance grade ("relevant" | "irrelevant" | "ambiguous").
        structured_response: Structured output from the specialist agent.
    """

    messages: Annotated[list[AnyMessage], add_messages]
    intent: str
    route_decision: RouteDecision | None
    agent_name: str
    requires_human_approval: bool
    human_approved: bool
    rag_documents: list[str]
    rag_grade: str
    structured_response: dict[str, Any] | None
