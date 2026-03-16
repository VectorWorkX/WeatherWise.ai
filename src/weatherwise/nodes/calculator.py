"""Calculator agent node — performs computations using fetched weather data.

Uses two tools:
- compute_expression: simple arithmetic (e.g., "23.11 - 20.59")
- run_calculation: sandboxed Python execution for complex operations
  (table parsing, statistics, correlations, multi-step analysis)

The LLM decides which tool to use based on the complexity of the request.
"""

from __future__ import annotations

import os
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from weatherwise.logging import get_logger
from weatherwise.state import CalcPayload
from weatherwise.tools.math_ops import compute_expression, run_calculation

logger = get_logger(__name__)

CALC_SYSTEM_PROMPT = """You are a weather data calculator with access to a Python sandbox.

You have two tools:
1. **compute_expression** — for simple arithmetic on known values (e.g., "23.11 - 20.59", "(72 - 32) * 5/9")
2. **run_calculation** — for anything complex: parsing tables, computing statistics over many values, correlations, multi-step analysis

CRITICAL RULES:
- When data contains a table with many rows, ALWAYS use run_calculation. Write Python code that parses the table to extract values programmatically. NEVER try to manually read and list values from a table — you WILL miss rows.
- Pass the raw weather data as the `data` parameter to run_calculation, then parse it in your code.
- Your code MUST assign the final answer to a variable called `result`.
- Available in the sandbox: `math`, `statistics` modules, and standard builtins (abs, round, min, max, sum, len, sorted, list, dict, float, int, str, etc.)
- For simple arithmetic with 2-3 known numbers, use compute_expression instead (faster).

Example for table parsing:
```python
lines = data.strip().split('\\n')
temps = []
for line in lines:
    parts = [p.strip() for p in line.split('|') if p.strip()]
    if len(parts) >= 4:
        try:
            temps.append(float(parts[3]))  # extract specific column
        except ValueError:
            pass
result = {'count': len(temps), 'mean': statistics.mean(temps), 'std_dev': statistics.stdev(temps)}
```"""

_calc_tools = [compute_expression, run_calculation]
_calc_tool_map = {t.name: t for t in _calc_tools}

_calc_llm: ChatAnthropic | None = None


def _get_calc_llm() -> ChatAnthropic:
    global _calc_llm
    if _calc_llm is None:
        _calc_llm = ChatAnthropic(
            model="claude-haiku-4-5-20251001",
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
            max_tokens=4096,
            temperature=0,
            timeout=60.0,
        )
    return _calc_llm


async def calculator_agent(state: CalcPayload) -> dict[str, Any]:
    """Perform a calculation using fetched weather data from state.

    Receives a CalcPayload (via Send) which includes fetched_weather
    collected in earlier phases. Uses tool-calling so the LLM decides
    whether to use simple arithmetic or sandboxed Python execution.
    """
    query = state["query"]
    location = state["location"]
    plan_index = state["plan_index"]
    fetched_weather = state.get("fetched_weather", {})
    prior_calculations = state.get("prior_calculations", [])

    logger.info("calculator_start", query=query, location=location, plan_index=plan_index)

    try:
        llm = _get_calc_llm().bind_tools(_calc_tools)

        data_summary = ""
        for loc, data in fetched_weather.items():
            entry = data if isinstance(data, str) else data.get("raw_response", str(data))
            data_summary += f"\n--- {loc} ---\n{entry}"

        # Include prior calculation results if this task depends on earlier calcs
        prior_context = ""
        if prior_calculations:
            prior_context = "\n\nPrior calculation results:\n" + "\n".join(
                f"  - {calc}" for calc in prior_calculations
            )

        messages: list = [
            SystemMessage(content=[
                {"type": "text", "text": CALC_SYSTEM_PROMPT,
                 "cache_control": {"type": "ephemeral"}},
            ]),
            HumanMessage(
                content=f"Calculation request: {query}\n"
                        f"Location: {location}\n"
                        f"Available weather data:{data_summary}{prior_context}"
            ),
        ]

        # ReAct loop: let the LLM call tools until it produces a final answer
        max_iterations = 5
        for _ in range(max_iterations):
            response = await llm.ainvoke(messages)
            messages.append(response)

            if not response.tool_calls:
                break

            for tool_call in response.tool_calls:
                tool_fn = _calc_tool_map.get(tool_call["name"])
                tool_result = (
                    tool_fn.invoke(tool_call["args"])
                    if tool_fn
                    else f"Unknown tool: {tool_call['name']}"
                )
                messages.append(ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_call["id"],
                ))

        result = response.content if isinstance(response.content, str) else str(response.content)
        logger.info("calculator_success", plan_index=plan_index)

    except Exception as e:
        logger.error("calculator_failed", error=str(e))
        result = f"Calculation failed for '{query}': {type(e).__name__}: {e}"

    return {
        "calculations": [result],
        "completed_indices": [plan_index],
    }
