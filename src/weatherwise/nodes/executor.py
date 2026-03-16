"""Executor node — dependency-aware fan-out via Send.

Instead of hardcoding phases by op type (fetch→calculate→done), the executor
reads each task's "depends_on" list and dispatches tasks whose dependencies
are all satisfied. This allows the planner to express arbitrary DAGs:

    fetch A ──┐
    fetch B ──┼──→ calc mean A, calc mean B ──→ calc compare ──→ synthesizer
    research ─┘

Tasks with no dependencies (or depends_on=[]) are dispatched immediately.
Tasks whose depends_on indices are all in completed_indices become ready.
The executor loops (workers → executor → next wave) until all tasks are done.
"""

from __future__ import annotations

from typing import Any, Union

from langgraph.types import Send

from weatherwise.logging import get_logger
from weatherwise.state import WeatherState

logger = get_logger(__name__)

# Safety limit: if executor is called more times than this, force synthesize
MAX_EXECUTOR_ITERATIONS = 15
_VALID_OPS = {"fetch", "research", "calculate", "visualize"}


async def executor(state: WeatherState) -> dict[str, Any]:
    """Pass-through node — routing logic lives in route_from_executor."""
    return {}


def _sanitize_depends_on(
    depends_on: list[int] | None,
    task_index: int,
    plan_length: int,
    op: str,
    plan: list[dict],
) -> list[int]:
    """Validate and sanitize a task's depends_on list.

    Strips invalid entries:
    - Self-references (task depending on itself)
    - Out-of-bounds indices (>= plan length or negative)
    - Non-integer values (defensive, should be caught by Pydantic)

    Falls back to implicit dependencies for calculate/visualize tasks without depends_on.
    """
    if depends_on is None:
        if op in ("calculate", "visualize"):
            # Safety fallback: calculate/visualize tasks without explicit depends_on
            # implicitly depend on all preceding tasks.
            return [
                j for j, t in enumerate(plan)
                if j < task_index and t.get("op") in ("fetch", "research", "calculate")
            ]
        return []

    sanitized = []
    for dep in depends_on:
        if not isinstance(dep, int):
            logger.warning("executor_invalid_dep_type", index=task_index, dep=dep)
            continue
        if dep == task_index:
            logger.warning("executor_self_reference", index=task_index)
            continue
        if dep < 0 or dep >= plan_length:
            logger.warning("executor_dep_out_of_bounds", index=task_index, dep=dep, plan_length=plan_length)
            continue
        sanitized.append(dep)

    return sanitized


def route_from_executor(
    state: WeatherState,
) -> Union[list[Send], str]:
    """Dependency-aware routing: dispatch tasks whose dependencies are satisfied.

    For each incomplete task:
    1. Sanitize its depends_on list (strip self-refs, out-of-bounds, etc.)
    2. Check if all valid dependencies are in completed_indices OR skipped
    3. If yes, the task is ready to dispatch
    4. If no, the task waits for the next executor invocation

    Deadlock detection: if no tasks are ready but tasks are waiting (circular
    deps or dependency on a permanently-blocked task), force-dispatch the
    blocked tasks with a warning rather than silently skipping them.
    """
    plan = state.get("plan", [])
    completed = set(state.get("completed_indices", []))

    # Safety: if we've iterated too many times, force synthesis
    completed_count = len(completed)
    if completed_count >= MAX_EXECUTOR_ITERATIONS:
        logger.warning("executor_max_iterations", completed=completed_count)
        return "synthesizer"

    # First pass: identify skipped tasks (invalid ops) so dependents can proceed
    skipped: set[int] = set()
    for i, task in enumerate(plan):
        if i in completed:
            continue
        op = task.get("op", "")
        if op not in _VALID_OPS:
            logger.warning("executor_skip_unknown_op", index=i, op=op)
            skipped.add(i)

    # For dependency resolution, treat skipped tasks as "completed"
    # so their dependents aren't permanently blocked
    resolved = completed | skipped

    ready_sends: list[Send] = []
    waiting_tasks: list[tuple[int, dict]] = []

    for i, task in enumerate(plan):
        if i in completed or i in skipped:
            continue

        op = task.get("op", "")

        # Sanitize depends_on: strip self-refs, out-of-bounds, negatives
        depends_on = _sanitize_depends_on(
            task.get("depends_on"),
            task_index=i,
            plan_length=len(plan),
            op=op,
            plan=plan,
        )

        # Check if all dependencies are satisfied
        if not all(dep in resolved for dep in depends_on):
            waiting_tasks.append((i, task))
            continue

        # Task is ready — build payload and dispatch
        base_payload = {
            "plan_index": i,
            "op": op,
            "location": task.get("location", ""),
            "query": task.get("query", ""),
        }

        if op == "fetch":
            ready_sends.append(Send("fetcher_agent", base_payload))
        elif op == "research":
            ready_sends.append(Send("retriever_agent", base_payload))
        elif op == "calculate":
            calc_payload = {
                **base_payload,
                "fetched_weather": state.get("fetched_weather", {}),
                "prior_calculations": list(state.get("calculations", [])),
            }
            ready_sends.append(Send("calculator_agent", calc_payload))
        elif op == "visualize":
            viz_payload = {
                "plan_index": i,
                "op": op,
                "query": task.get("query", ""),
                "fetched_weather": state.get("fetched_weather", {}),
                "calculations": list(state.get("calculations", [])),
            }
            ready_sends.append(Send("visualizer_agent", viz_payload))

    logger.info(
        "executor_routing",
        ready=len(ready_sends),
        waiting=len(waiting_tasks),
        completed=completed_count,
        skipped=len(skipped),
    )

    if ready_sends:
        return ready_sends

    # Deadlock detection: tasks are waiting but nothing is ready.
    # This happens with circular deps, or deps on tasks that can never complete.
    # Force-dispatch blocked tasks rather than silently dropping them.
    if waiting_tasks:
        logger.warning(
            "executor_deadlock_detected",
            blocked_indices=[i for i, _ in waiting_tasks],
            completed=sorted(completed),
        )
        forced_sends: list[Send] = []
        for i, task in waiting_tasks:
            op = task.get("op", "")
            base_payload = {
                "plan_index": i,
                "op": op,
                "location": task.get("location", ""),
                "query": task.get("query", ""),
            }
            if op == "fetch":
                forced_sends.append(Send("fetcher_agent", base_payload))
            elif op == "research":
                forced_sends.append(Send("retriever_agent", base_payload))
            elif op == "calculate":
                calc_payload = {
                    **base_payload,
                    "fetched_weather": state.get("fetched_weather", {}),
                    "prior_calculations": list(state.get("calculations", [])),
                }
                forced_sends.append(Send("calculator_agent", calc_payload))

        return forced_sends

    # All done — synthesize
    return "synthesizer"
