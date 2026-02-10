"""Research specialist agent — handles Bucket 5 (explanation).

Tools: RAG (search_knowledge_base) + Xweather MCP (no calculator).
Internal ReAct loop: research_worker ⇄ ToolNode(mcp_tools + [rag_tool]).

Corrective RAG: after tool execution, a grading node checks if the RAG results
are relevant. If irrelevant, it rewrites the query and retries once.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Literal, Sequence

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from graph.prompts import EXPLANATION_PROMPT, QUERY_REWRITER_PROMPT, RAG_GRADER_PROMPT
from graph.state import AgentState

load_dotenv()

_research_llm: ChatAnthropic | None = None
_grader_llm: ChatAnthropic | None = None


class RelevanceGrade(BaseModel):
    """Structured output for RAG document relevance grading."""

    grade: Literal["relevant", "irrelevant", "ambiguous"] = Field(
        description="Whether the retrieved documents are relevant to the query."
    )
    reasoning: str = Field(
        description="Brief explanation of the grading decision."
    )


def _get_research_llm() -> ChatAnthropic:
    """Return a cached Sonnet instance for research tasks."""
    global _research_llm
    if _research_llm is None:
        _research_llm = ChatAnthropic(
            model="claude-sonnet-4-5-20250929",
            api_key=os.environ["ANTHROPIC_API_KEY"],
            temperature=0,
            max_tokens=2048,
        )
    return _research_llm


def _get_grader_llm() -> ChatAnthropic:
    """Return a cached Haiku instance for fast RAG grading."""
    global _grader_llm
    if _grader_llm is None:
        _grader_llm = ChatAnthropic(
            model="claude-haiku-4-5-20251001",
            api_key=os.environ["ANTHROPIC_API_KEY"],
            temperature=0,
            max_tokens=256,
        )
    return _grader_llm


def _date_context() -> str:
    now = datetime.now(timezone.utc)
    local_now = datetime.now()
    return (
        f"Current date and time: {local_now.strftime('%A, %B %d, %Y at %I:%M %p')} local, "
        f"{now.strftime('%Y-%m-%dT%H:%M:%SZ')} UTC."
    )


def build_research_subgraph(
    mcp_tools: Sequence[BaseTool],
    rag_tool: BaseTool,
):
    """Build and compile the research agent subgraph with Corrective RAG.

    Flow::

        research_worker → tools → grade_documents ─┬─ relevant  → research_worker (continue)
                                                    ├─ ambiguous → research_worker (continue)
                                                    └─ irrelevant → rewrite_query → research_worker (retry)

    Returns a compiled graph suitable for ``parent_graph.add_node()``.
    """
    tools = list(mcp_tools) + [rag_tool]
    llm = _get_research_llm().bind_tools(tools)

    async def research_worker(state: AgentState) -> dict:
        system_content = f"{_date_context()}\n\n{EXPLANATION_PROMPT}"
        system_msg = SystemMessage(
            content=system_content,
            additional_kwargs={"cache_control": {"type": "ephemeral"}},
        )
        response = await llm.ainvoke([system_msg] + state["messages"])
        return {"messages": [response]}

    def should_continue(state: AgentState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return END

    async def grade_documents(state: AgentState) -> dict:
        """Grade whether the RAG retrieval returned relevant documents."""
        messages = state["messages"]

        # Find the most recent ToolMessage from search_knowledge_base
        rag_content = ""
        user_query = ""
        for msg in reversed(messages):
            if isinstance(msg, ToolMessage) and not rag_content:
                if "knowledge base" in msg.content.lower() or "relevant documents" in msg.content.lower():
                    rag_content = msg.content
            if isinstance(msg, HumanMessage) and not user_query:
                user_query = msg.content if isinstance(msg.content, str) else ""

        # If no RAG content found (e.g., only MCP tools used), skip grading
        if not rag_content or not user_query:
            return {"rag_grade": "relevant"}

        # Skip grading if RAG returned "no relevant documents" or "empty"
        if "no relevant documents" in rag_content.lower() or "currently empty" in rag_content.lower():
            return {"rag_grade": "irrelevant", "rag_documents": [rag_content]}

        grader = _get_grader_llm().with_structured_output(RelevanceGrade)
        result = await grader.ainvoke([
            SystemMessage(content=RAG_GRADER_PROMPT),
            HumanMessage(content=(
                f"User question: {user_query}\n\n"
                f"Retrieved documents:\n{rag_content[:2000]}"
            )),
        ])

        return {
            "rag_grade": result.grade,
            "rag_documents": [rag_content[:500]],
        }

    def route_after_grading(state: AgentState) -> str:
        grade = state.get("rag_grade", "relevant")
        if grade == "irrelevant":
            return "rewrite_query"
        # For relevant or ambiguous, let the agent continue its ReAct loop
        return "research_worker"

    async def rewrite_query(state: AgentState) -> dict:
        """Rewrite the user query for better RAG retrieval."""
        messages = state["messages"]

        # Find original user query
        user_query = ""
        for msg in messages:
            if isinstance(msg, HumanMessage):
                user_query = msg.content if isinstance(msg.content, str) else ""
                break

        rewriter = _get_grader_llm()
        rewritten = await rewriter.ainvoke([
            SystemMessage(content=QUERY_REWRITER_PROMPT),
            HumanMessage(content=f"Original query: {user_query}"),
        ])

        # Inject the rewritten query as a new HumanMessage so the agent retries RAG
        rewritten_text = rewritten.content if isinstance(rewritten.content, str) else user_query
        return {
            "messages": [HumanMessage(content=(
                f"{user_query}\n\n"
                f"[System: Previous RAG search was not relevant. "
                f"Try searching with: \"{rewritten_text}\"]"
            ))],
            "rag_grade": "retrying",
        }

    graph = StateGraph(AgentState)
    graph.add_node("research_worker", research_worker)
    graph.add_node("tools", ToolNode(tools, handle_tool_errors=True))
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("rewrite_query", rewrite_query)

    graph.set_entry_point("research_worker")
    graph.add_conditional_edges("research_worker", should_continue, {
        "tools": "tools",
        END: END,
    })
    graph.add_edge("tools", "grade_documents")
    graph.add_conditional_edges("grade_documents", route_after_grading, {
        "research_worker": "research_worker",
        "rewrite_query": "rewrite_query",
    })
    graph.add_edge("rewrite_query", "research_worker")

    return graph.compile()
