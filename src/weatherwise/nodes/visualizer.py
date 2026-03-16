"""Visualizer agent node — generates visual stories from weather data.

The visualizer thinks like a data journalist:
1. Analyzes the data to find meaningful patterns
2. Decides what visual story to tell and how many charts it needs
3. Extracts data programmatically using run_calculation
4. Generates complete ECharts option JSON directly — no templates, no merging
5. Produces metrics, insights, and narratives that go beyond the question

The LLM writes the final ECharts configuration itself. The frontend renders
it directly with theme="dark". No intermediate format, no template search.
"""

from __future__ import annotations

import os
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field

from weatherwise.logging import get_logger
from weatherwise.retry import llm_retry
from weatherwise.state import VisualizePayload
from weatherwise.tools.math_ops import run_calculation

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Structured output models
# ---------------------------------------------------------------------------

class MetricCard(BaseModel):
    label: str
    value: str
    unit: str = ""
    detail: str = ""


class ReportSection(BaseModel):
    heading: str
    narrative: str
    priority: int = Field(default=1, description="1 = full width, 2 = half width (side by side)")
    echarts_option: dict[str, Any] = Field(
        description="Complete ECharts option JSON object ready for rendering"
    )


class VisualReport(BaseModel):
    title: str
    subtitle: str = ""
    metrics: list[MetricCard] = Field(default_factory=list)
    sections: list[ReportSection] = Field(min_length=1)
    insights: list[str] = Field(min_length=1)


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
statistic meaningful in context? Show both.
4. How many chart sections does this story need? Use your judgment — sometimes one chart \
is enough, sometimes the data has multiple dimensions that each deserve their own view.
5. What context makes each chart self-explanatory? Consider reference lines, annotations, \
axis labels with units, and any other elements that help the viewer understand \
the chart without reading surrounding text.

DATA EXTRACTION:
Use run_calculation to extract numeric values from the weather data programmatically. \
Never manually list values from tables — write code that parses the data completely.

GENERATING CHARTS:
For each chart section, generate a COMPLETE ECharts option JSON object in the \
echarts_option field. This must be valid JSON that ECharts can render directly. \
Include title, tooltip, appropriate axes with labels and units, and series with data. \
The frontend applies a dark theme automatically — do not set background colors, \
but do set text colors to light values (#ccc, #999) for readability.

Make charts visually rich and professional. When comparing values with different \
units or scales (for example, temperature and rainfall), use separate charts or \
dual axes to keep both readable.

If the user explicitly requests a specific chart type, use that type.

METRICS: Highlight key numbers as metric cards with units.
INSIGHTS: Provide meaningful analytical observations about what the data reveals.
PRIORITY: priority=1 for the main chart (full width), priority=2 for supporting \
charts (side-by-side)."""

_viz_llm: ChatAnthropic | None = None


def _get_viz_llm() -> ChatAnthropic:
    global _viz_llm
    if _viz_llm is None:
        _viz_llm = ChatAnthropic(
            model="claude-haiku-4-5-20251001",
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
            max_tokens=8000,
            temperature=0.2,
            timeout=90.0,
        )
    return _viz_llm


# ---------------------------------------------------------------------------
# Visualizer node
# ---------------------------------------------------------------------------

async def visualizer_agent(state: VisualizePayload) -> dict[str, Any]:
    """Generate a visual story from fetched weather data.

    Receives a VisualizePayload (via Send) with fetched_weather and calculations.
    Uses LLM to analyze data, extract values via run_calculation, and generate
    complete ECharts option JSON directly — no templates, no merging.
    """
    query = state.get("query", "")
    plan_index = state["plan_index"]
    fetched_weather = state.get("fetched_weather", {})
    calculations = state.get("calculations", [])

    logger.info("visualizer_start", query=query, plan_index=plan_index)

    try:
        # Build data context for the LLM
        data_summary = ""
        for loc, data in fetched_weather.items():
            entry = data if isinstance(data, str) else data.get("raw_response", str(data))
            data_summary += f"\n--- {loc} ---\n{entry}"

        calc_context = ""
        if calculations:
            calc_context = "\n\nCalculation results:\n" + "\n".join(
                f"  - {c}" for c in calculations
            )

        # ReAct loop with run_calculation for data extraction
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
                        f"Analyze the data, use run_calculation to extract values "
                        f"programmatically, then generate the VisualReport with "
                        f"complete ECharts option JSON for each chart section."
            ),
        ]

        # ReAct loop — data extraction only (no template search)
        max_iterations = 5
        for _ in range(max_iterations):
            response = await llm_with_tools.ainvoke(messages)
            messages.append(response)

            if not response.tool_calls:
                break

            for tool_call in response.tool_calls:
                tool_fn = tool_map.get(tool_call["name"])
                if tool_fn:
                    tool_result = tool_fn.invoke(tool_call["args"])
                else:
                    tool_result = f"Unknown tool: {tool_call['name']}"
                messages.append(ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_call["id"],
                ))

        # Structured output — LLM generates VisualReport with echarts_option
        structured_llm = _get_viz_llm().with_structured_output(VisualReport)

        @llm_retry()
        async def _generate_report():
            return await structured_llm.ainvoke(
                messages + [HumanMessage(
                    content="Now generate the VisualReport. For each chart section, "
                            "write the complete ECharts option JSON in echarts_option. "
                            "Include all extracted data values, metrics, and insights."
                )]
            )

        report: VisualReport = await _generate_report()

        # Build visual_report — echarts_option is already complete, no build step
        visual_report = {
            "title": report.title,
            "subtitle": report.subtitle,
            "metrics": [m.model_dump() for m in report.metrics],
            "sections": [
                {
                    "heading": s.heading,
                    "narrative": s.narrative,
                    "priority": s.priority,
                    "echarts_option": s.echarts_option,
                }
                for s in report.sections
            ],
            "insights": report.insights,
        }

        logger.info("visualizer_success", plan_index=plan_index,
                     section_count=len(report.sections))

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
