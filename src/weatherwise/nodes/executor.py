"""Executor node — two-phase fan-out via Send.

Phase 1: Fan out all fetch + research tasks in parallel.
Phase 2: Once phase 1 is complete, fan out calculate tasks
         (which receive the fetched data in their payload).
When all phases are done, route to synthesizer.
"""

from __future__ import annotations

from typing import Any, Union

from langgraph.types import Send

from weatherwise.logging import get_logger
from weatherwise.state import WeatherState

logger = get_logger(__name__)

# Safety limit: if executor is called more times than this, force synthesize
MAX_EXECUTOR_ITERATIONS = 10
_VALID_OPS = {"fetch", "research", "calculate"}


async def executor(state: WeatherState) -> dict[str, Any]:
    """Pass-through node — routing logic lives in route_from_executor."""
    return {}


def route_from_executor(
    state: WeatherState,
) -> Union[list[Send], str]:
    """Two-phase routing: fetch/research first, then calculate, then synthesize.

    Phase 1: Send all "fetch" and "research" tasks.
    Phase 2: Once those complete, Send all "calculate" tasks with fetched data.
    Done:    Route to synthesizer.

    Includes safeguards:
    - Skips tasks with unrecognized ops (prevents infinite loop)
    - Caps executor iterations to MAX_EXECUTOR_ITERATIONS
    """
    plan = state.get("plan", [])
    completed = set(state.get("completed_indices", []))

    # Safety: if we've iterated too many times, force synthesis
    completed_count = len(completed)
    if completed_count >= MAX_EXECUTOR_ITERATIONS:
        logger.warning("executor_max_iterations", completed=completed_count)
        return "synthesizer"

    # Categorize remaining tasks
    phase1_sends: list[Send] = []  # fetch + research
    phase2_sends: list[Send] = []  # calculate (depends on fetched data)
    skipped: list[int] = []

    for i, task in enumerate(plan):
        if i in completed:
            continue

        op = task.get("op", "")
        if op not in _VALID_OPS:
            logger.warning("executor_skip_unknown_op", index=i, op=op)
            skipped.append(i)
            continue

        base_payload = {
            "plan_index": i,
            "op": op,
            "location": task.get("location", ""),
            "query": task.get("query", ""),
        }

        if op == "fetch":
            phase1_sends.append(Send("fetcher_agent", base_payload))
        elif op == "research":
            phase1_sends.append(Send("retriever_agent", base_payload))
        elif op == "calculate":
            calc_payload = {
                **base_payload,
                "fetched_weather": state.get("fetched_weather", {}),
            }
            phase2_sends.append(Send("calculator_agent", calc_payload))

    logger.info(
        "executor_routing",
        phase1=len(phase1_sends),
        phase2=len(phase2_sends),
        completed=completed_count,
        skipped=len(skipped),
    )

    # Phase 1: still have fetch/research tasks to run
    if phase1_sends:
        return phase1_sends

    # Phase 2: fetch/research done, now run calculators
    if phase2_sends:
        return phase2_sends

    # All done — synthesize
    return "synthesizer"
