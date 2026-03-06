"""WeatherWise graph builder — wires the StateGraph with two-phase Send fan-out."""

from __future__ import annotations

import os
from typing import Literal

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from weatherwise.state import WeatherState
from weatherwise.nodes.supervisor import supervisor
from weatherwise.nodes.planner import planner
from weatherwise.nodes.executor import executor, route_from_executor
from weatherwise.nodes.fetcher import fetcher_agent
from weatherwise.nodes.retriever import retriever_agent
from weatherwise.nodes.calculator import calculator_agent
from weatherwise.nodes.synthesizer import synthesizer

# Maximum graph recursion steps (prevents infinite loops)
RECURSION_LIMIT = 25


def route_from_supervisor(state: WeatherState) -> str:
    """Route based on supervisor's classification."""
    if state.get("routing") == "ambiguous":
        return END  # Graph pauses; user sees the clarifying AIMessage
    return "planner"


def _get_checkpointer(mode: Literal["memory", "sqlite", "none"] = "memory"):
    """Create the appropriate checkpointer based on mode."""
    if mode == "none":
        return None
    if mode == "sqlite":
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        db_path = os.environ.get("CHECKPOINT_DB_PATH", "weatherwise_checkpoints.db")
        return AsyncSqliteSaver.from_conn_string(db_path)
    return MemorySaver()


def build_graph(
    checkpointer: Literal["memory", "sqlite", "none"] | bool = True,
    checkpointer_instance=None,
) -> StateGraph:
    """Construct and compile the WeatherWise multi-agent graph.

    Two-phase graph flow:
        START -> supervisor -> [ambiguous: END, clear: planner]
        planner -> executor

        Phase 1 (parallel):
            executor -> Send(fetcher_agent, retriever_agent)
            fetcher_agent  -> executor  (loop back)
            retriever_agent -> executor (loop back)

        Phase 2 (after phase 1 completes):
            executor -> Send(calculator_agent) with fetched data
            calculator_agent -> executor (loop back)

        Completion:
            executor -> synthesizer (when all tasks done)
            synthesizer -> END

    Args:
        checkpointer: Backend selection for memory/sqlite/none (dev/test).
        checkpointer_instance: Pre-created async checkpointer (e.g., AsyncRedisSaver).
            When provided, takes precedence over the ``checkpointer`` string.

    Returns:
        Compiled graph ready for `.ainvoke()` or `.astream()`.
    """
    graph = StateGraph(WeatherState)

    # -- Register nodes --
    graph.add_node("supervisor", supervisor)
    graph.add_node("planner", planner)
    graph.add_node("executor", executor)
    graph.add_node("fetcher_agent", fetcher_agent)
    graph.add_node("retriever_agent", retriever_agent)
    graph.add_node("calculator_agent", calculator_agent)
    graph.add_node("synthesizer", synthesizer)

    # -- Wire edges --

    # 1. START -> supervisor
    graph.add_edge(START, "supervisor")

    # 2. supervisor -> planner OR END (ambiguous)
    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {END: END, "planner": "planner"},
    )

    # 3. planner -> executor
    graph.add_edge("planner", "executor")

    # 4. executor -> fan-out via Send OR synthesizer
    graph.add_conditional_edges(
        "executor",
        route_from_executor,
        ["fetcher_agent", "retriever_agent", "calculator_agent", "synthesizer"],
    )

    # 5. Fan-in: all workers loop back to executor (not synthesizer)
    graph.add_edge("fetcher_agent", "executor")
    graph.add_edge("retriever_agent", "executor")
    graph.add_edge("calculator_agent", "executor")

    # 6. synthesizer -> END
    graph.add_edge("synthesizer", END)

    # -- Compile --
    if checkpointer_instance is not None:
        cp = checkpointer_instance
    else:
        # Backwards compat: bool True -> "memory", False -> "none"
        if checkpointer is True:
            checkpointer = "memory"
        elif checkpointer is False:
            checkpointer = "none"
        cp = _get_checkpointer(checkpointer)
    compiled = graph.compile(checkpointer=cp)
    compiled.recursion_limit = RECURSION_LIMIT
    return compiled
