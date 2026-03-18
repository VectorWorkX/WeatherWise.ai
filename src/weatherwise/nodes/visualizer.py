"""Visualizer agent node — generates visual stories from weather data.

The visualizer thinks like a data journalist:
1. Analyzes the data to find meaningful patterns
2. Extracts data programmatically using run_calculation
3. Generates the complete report (including ECharts options) as Python code
4. The Python interpreter validates everything at execution time
5. The ReAct loop retries automatically if code fails

All chart generation happens via code execution — not structured output.
This is fundamentally more reliable because LLMs are better at writing code
than generating complex nested JSON, and the interpreter catches errors
immediately so the LLM can fix them on the next iteration.
"""

from __future__ import annotations

import json
import os
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from weatherwise.logging import get_logger
from weatherwise.state import VisualizePayload
from weatherwise.tools.math_ops import run_calculation

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Visualizer LLM
# ---------------------------------------------------------------------------

VISUALIZER_SYSTEM_PROMPT = """You are a weather data visualization analyst. Your job is to create \
visual reports that tell a complete story — not just plot numbers.

THINK BEFORE YOU BUILD:
1. What patterns exist in this data? Look for any meaningful structure — trends, \
comparisons, distributions, anomalies, correlations, or anything else worth showing.
2. What would a viewer learn from seeing this visually that they can't get from text?
3. What data SUPPORTS the insight? Always show the underlying data, not just the summary. \
Think: if someone asks for a single statistic, what surrounding data would make that \
statistic meaningful in context? Show both or more if necessary.
4. How many chart sections does this story need? Use your judgment — sometimes one chart \
is enough, sometimes the data has multiple dimensions that each deserve their own view.
5. What context makes each chart self-explanatory? Consider reference lines, annotations, \
axis labels with units, and any other elements that help the viewer understand \
the chart without reading surrounding text.

YOUR WORKFLOW (two run_calculation calls):

**Call 1 — Data extraction:** Parse the raw weather data to extract numeric values. \
The raw data is available as the `data` variable. Write code that parses it and assigns \
the parsed values to `result` as a dict.

**Call 2 — Report generation:** Construct the COMPLETE visual report as a dict and output \
it with `result = json.dumps(report)`. Each call has its own isolated environment — \
variables from call 1 do NOT exist in call 2. However, the `data` variable in call 2 \
contains both the original weather data AND your extracted results from call 1 \
(after a "--- EXTRACTED DATA ---" marker). Parse what you need from `data`. \
The report dict must have this structure:
{
    "title": "string",
    "subtitle": "string (optional)",
    "metrics": [{"label": "string", "value": "string", "unit": "string", "detail": "string (optional)"}],
    "sections": [{
        "heading": "string",
        "narrative": "string",
        "priority": 1 or 2,  // 1 = full width, 2 = side by side
        "echarts_option": { ... complete ECharts option object ... }
    }],
    "insights": ["string", "string", ...]
}

ECHARTS GUIDELINES:
- Generate COMPLETE ECharts option objects ready for rendering
- Include tooltip, legend, axes with labels and units, and series with data
- The frontend applies a dark theme — do not set background colors, \
but set text colors to light values (#ccc, #999) for readability
- When comparing values with different units or scales, use separate charts or dual axes
- If the user requests a specific chart type, use that type

CRITICAL: Your charts must be based entirely on the actual fetched data provided to you. \
If the data contains zero records or no values for a metric, do NOT create charts for it — \
report the gap instead. Never fill in charts with general reference information or \
training knowledge.

IMPORTANT: In call 2, you have access to `json` module. Always output the report \
with `result = json.dumps(report)` so it can be parsed."""

_viz_llm: ChatAnthropic | None = None


def _get_viz_llm() -> ChatAnthropic:
    global _viz_llm
    if _viz_llm is None:
        _viz_llm = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
            max_tokens=16000,
            temperature=0.2,
            timeout=120.0,
        )
    return _viz_llm


def _try_parse_dict(content: str) -> dict[str, Any] | None:
    """Try to parse a string as a dict — JSON first, then Python repr fallback.

    The LLM should output json.dumps(report), but may output the raw dict
    (Python repr with single quotes). Handle both cases.
    """
    # Try JSON first (the expected format)
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: Python dict repr (single quotes, True/False/None)
    import ast
    try:
        parsed = ast.literal_eval(content)
        if isinstance(parsed, dict):
            return parsed
    except (ValueError, SyntaxError):
        pass

    return None


def _extract_report_from_messages(messages: list) -> dict[str, Any] | None:
    """Scan tool results for a valid report dict.

    Walks backwards through messages looking for a ToolMessage whose content
    parses to a dict with 'title' and 'sections' keys — the report output.
    Handles both JSON and Python repr formats.
    """
    for msg in reversed(messages):
        if not isinstance(msg, ToolMessage):
            continue
        content = msg.content
        if not isinstance(content, str):
            continue
        parsed = _try_parse_dict(content)
        if (
            parsed
            and "title" in parsed
            and "sections" in parsed
            and isinstance(parsed["sections"], list)
        ):
            return parsed
    return None


def _validate_and_fill_report(raw: dict[str, Any]) -> dict[str, Any]:
    """Validate report structure and fill missing optional fields with defaults.

    Instead of rejecting the report on missing fields, we fill sensible defaults.
    This ensures a partial report still renders rather than failing entirely.
    """
    report: dict[str, Any] = {
        "title": raw.get("title", "Visual Report"),
        "subtitle": raw.get("subtitle", ""),
        "metrics": [],
        "sections": [],
        "insights": [],
    }

    # Metrics
    for m in raw.get("metrics", []):
        if isinstance(m, dict) and "label" in m and "value" in m:
            report["metrics"].append({
                "label": m["label"],
                "value": str(m["value"]),
                "unit": m.get("unit", ""),
                "detail": m.get("detail", ""),
            })

    # Sections
    for s in raw.get("sections", []):
        if isinstance(s, dict) and "echarts_option" in s:
            report["sections"].append({
                "heading": s.get("heading", "Chart"),
                "narrative": s.get("narrative", ""),
                "priority": s.get("priority", 1),
                "echarts_option": s["echarts_option"],
            })

    # Insights
    for insight in raw.get("insights", []):
        if isinstance(insight, str) and insight.strip():
            report["insights"].append(insight)

    return report


# ---------------------------------------------------------------------------
# Visualizer node
# ---------------------------------------------------------------------------

async def visualizer_agent(state: VisualizePayload) -> dict[str, Any]:
    """Generate a visual story from fetched weather data.

    Receives a VisualizePayload (via Send) with fetched_weather and calculations.
    Uses a ReAct loop where the LLM generates the entire report — including
    ECharts options — as Python code via run_calculation. No structured output.
    """
    query = state.get("query", "")
    plan_index = state["plan_index"]
    fetched_weather = state.get("fetched_weather", {})
    calculations = state.get("calculations", [])

    logger.info("visualizer_start", query=query, plan_index=plan_index)

    try:
        # Build data context for the LLM — prefer raw_data (structured JSON)
        # over raw_response (markdown text) for reliable parsing
        data_summary = ""
        for loc, weather_data in fetched_weather.items():
            if isinstance(weather_data, dict):
                entry = weather_data.get("raw_data") or weather_data.get("raw_response", str(weather_data))
            else:
                entry = str(weather_data)
            data_summary += f"\n--- {loc} ---\n{entry}"

        calc_context = ""
        if calculations:
            calc_context = "\n\nCalculation results:\n" + "\n".join(
                f"  - {c}" for c in calculations
            )

        # ReAct loop — data extraction AND report generation, all via code
        llm = _get_viz_llm()
        _tools = [run_calculation]
        llm_with_tools = llm.bind_tools(_tools)
        tool_map = {t.name: t for t in _tools}

        messages: list = [
            SystemMessage(content=[
                {"type": "text", "text": VISUALIZER_SYSTEM_PROMPT,
                 "cache_control": {"type": "ephemeral"}},
            ]),
            HumanMessage(
                content=f"Create a visual report for this query: {query}\n\n"
                        f"Available weather data:{data_summary}{calc_context}\n\n"
                        f"Step 1: Use run_calculation to extract and parse the data.\n"
                        f"Step 2: Use run_calculation to construct the complete report dict "
                        f"(with ECharts options) and output it with result = json.dumps(report).\n\n"
                        f"IMPORTANT: Each run_calculation call has its own isolated environment. "
                        f"The `data` variable always contains the raw weather data plus any "
                        f"previously extracted results. You can parse either."
            ),
        ]

        # Track accumulated results so subsequent calls can access extracted data.
        # Each run_calculation has its own isolated sandbox — variables from call 1
        # don't exist in call 2. We solve this by appending previous results to the
        # data variable, so the LLM can parse them in later calls.
        accumulated_data = data_summary

        max_iterations = 6
        for _ in range(max_iterations):
            response = await llm_with_tools.ainvoke(messages)
            messages.append(response)

            if not response.tool_calls:
                break

            for tool_call in response.tool_calls:
                tool_fn = tool_map.get(tool_call["name"])
                if tool_fn:
                    args = dict(tool_call["args"])
                    if tool_call["name"] == "run_calculation" and not args.get("data"):
                        args["data"] = accumulated_data
                    tool_result = tool_fn.invoke(args)
                    # Append successful results so the next call can access them.
                    # All error prefixes from run_calculation in math_ops.py:
                    #   "Code validation error:", "Code execution error:",
                    #   "Code execution timed out", "Error: code must assign"
                    result_str = str(tool_result)
                    _error_prefixes = ("Code validation error", "Code execution error",
                                       "Code execution timed out", "Error: code must assign")
                    if not result_str.startswith(_error_prefixes):
                        accumulated_data = (
                            data_summary
                            + f"\n\n--- EXTRACTED DATA (from previous step) ---\n{result_str}"
                        )
                else:
                    tool_result = f"Unknown tool: {tool_call['name']}"
                messages.append(ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_call["id"],
                ))

        # Extract the report from tool results
        raw_report = _extract_report_from_messages(messages)

        if raw_report:
            visual_report = _validate_and_fill_report(raw_report)
            logger.info("visualizer_success", plan_index=plan_index,
                         section_count=len(visual_report.get("sections", [])))
        else:
            logger.warning("visualizer_no_report_found", plan_index=plan_index)
            visual_report = {
                "error": "Visualization could not be generated — no valid report was produced.",
                "sections": [],
            }

    except Exception as e:
        logger.error("visualizer_failed", error=str(e), plan_index=plan_index)
        visual_report = {
            "error": f"Visualization failed: {type(e).__name__}: {e}",
            "sections": [],
        }

    return {
        "visual_report": visual_report,
        "completed_indices": [plan_index],
    }
