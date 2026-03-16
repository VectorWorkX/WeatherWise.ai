"""Visualizer agent node — designs visual stories from weather data.

The visualizer is an analyst, not a plotter. It:
1. Analyzes the data shape and query intent
2. Decides how many chart sections the story needs (1 for simple, 3+ for complex)
3. For each section, picks the chart type that best reveals the pattern
4. Extracts data programmatically using run_calculation (no manual row skipping)
5. Generates metrics, insights, and narratives that go beyond the user's question
6. Outputs structured JSON that the frontend renders with ECharts

Uses run_calculation for reliable data extraction from tables, and
search_chart_template for finding the best visual design from Pinecone.
"""

from __future__ import annotations

import json
import os
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field

from weatherwise.logging import get_logger
from weatherwise.retry import llm_retry
from weatherwise.state import VisualizePayload
from weatherwise.tools.chart_templates import build_echarts_option, discover_chart_templates
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


class ChartSpec(BaseModel):
    type: str = Field(description="Chart type: line, bar, pie, donut, radar, gauge, scatter, heatmap")
    search_hint: str = Field(description="Visual description for template search — describe the desired look, not the data")
    title: str
    data: list[dict[str, Any]] = Field(default_factory=list)
    x_field: str = ""
    y_fields: list[str] = Field(default_factory=list)
    x_label: str = ""
    y_label: str = ""
    y_field: str = ""
    value_field: str = ""
    unit: str = ""
    smooth: bool = False
    area: bool = False
    stacked: bool = False
    horizontal: bool = False
    show_labels: bool = False
    dual_axis: bool = False
    mood: str = "neutral"
    metrics: list[str] = Field(default_factory=list)
    max_values: list[float] = Field(default_factory=list)
    min: float | None = None
    max: float | None = None
    reference_lines: list[dict[str, Any]] = Field(default_factory=list)


class ReportSection(BaseModel):
    heading: str
    narrative: str
    priority: int = Field(default=1, description="1 = full width, 2 = half width (side by side)")
    chart: ChartSpec


class VisualReport(BaseModel):
    title: str
    subtitle: str = ""
    metrics: list[MetricCard] = Field(default_factory=list)
    sections: list[ReportSection] = Field(min_length=1)
    insights: list[str] = Field(min_length=1)


# ---------------------------------------------------------------------------
# Visualizer LLM
# ---------------------------------------------------------------------------

VISUALIZER_SYSTEM_PROMPT = """You are a weather data visualization analyst. Your job is to create
insightful visual reports that tell a complete story about the data.

WORKFLOW (follow this order strictly):

1. ANALYZE: Study the weather data. Identify what patterns exist (trends, comparisons,
   distributions, anomalies). Decide what chart types would best reveal these patterns
   and how many sections the story needs. Use ONE chart for simple data, multiple ONLY
   if different aspects need different chart types.

2. DISCOVER TEMPLATES: Call discover_chart_templates with your visual intent for each
   chart. You can search for multiple templates in ONE call (bundle search).
   Read the returned data_mapping carefully — it tells you EXACTLY:
   - WHERE data goes (series_paths, x_axis_path, category_path)
   - HOW MANY series the template supports
   - What structure the template expects
   You MUST conform your data output to match these mapping requirements.

3. EXTRACT DATA: Use run_calculation to programmatically extract numeric values from
   the weather data. NEVER manually list values from tables — write code that parses
   the table to ensure no values are missed.

4. GENERATE REPORT: Output the VisualReport with data shaped to match each template's
   data_mapping. For each chart section, include the search_hint you used so the
   builder can find the same template.

UNIVERSAL DATA FORMATS — use these standard shapes in your chart data:

For categorical/time-series data (line, bar, scatter):
  data: [{"label": "Mar 15", "Delhi": 29.9, "San Jose": 27.7}, ...]
  x_field: "label"
  (y_fields will be auto-detected from numeric keys if you omit them)

For single-value categories (pie, donut, gauge):
  data: [{"label": "Sunny Days", "value": 15}, {"label": "Rainy Days", "value": 8}]

For multi-dimensional profiles (radar):
  data: [{"group": "Delhi", "metrics": {"Temp": 24, "Humidity": 55, "Wind": 12}}]
  metrics: ["Temp", "Humidity", "Wind"]
  max_values: [50, 100, 60]

For matrix data (heatmap):
  data: [{"row": "Monday", "column": "6am", "value": 15}]
  x_field: "row", y_field: "column", value_field: "value"

CHART SELECTION — choose based on what the data reveals:
- "line": data changes over time (trends, forecasts)
- "bar": comparing values across categories
- "donut": parts of a whole (distribution)
- "radar": comparing multiple metrics across entities (needs 3+ metrics to be meaningful)
- "gauge": single current reading against a range
- "scatter": relationship between two variables
- "heatmap": intensity across two dimensions

If the user explicitly requests a specific chart type, use that type.
If the data doesn't fit the chosen chart type (e.g., only 1 metric for radar), choose
a chart type that matches the data's actual shape.

THINK BEYOND THE QUESTION:
- Add reference lines for means or thresholds when relevant
- Highlight which entity "wins" in comparisons
- Set mood to "danger" for severe weather / unhealthy AQI
- Set dual_axis when comparing different units
- Include at least 3 insights that go beyond restating the numbers
- Include every detail: variable names, units, labels, date ranges

METRICS: Highlight key numbers as metric cards. Include the unit for each.

PRIORITY: Set priority=1 for the main chart (full width), priority=2 for supporting
charts (displayed side-by-side in the frontend)."""

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
    """Design a visual story from fetched weather data.

    Receives a VisualizePayload (via Send) with fetched_weather and calculations.
    Uses LLM to analyze data, pick chart types, extract values, and generate
    a structured VisualReport. Then builds complete ECharts options for each chart.
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

        # Use LLM with tools to discover templates and extract data
        llm = _get_viz_llm()
        _tools = [discover_chart_templates, run_calculation]
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
                        f"Follow the workflow: first call discover_chart_templates to find "
                        f"the best visual designs for your charts, then use run_calculation "
                        f"to extract data values, then generate the VisualReport with data "
                        f"shaped to match each template's data_mapping requirements."
            ),
        ]

        # ReAct loop — template discovery + data extraction + report generation
        max_iterations = 8
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

        # Now get structured output from the LLM
        structured_llm = _get_viz_llm().with_structured_output(VisualReport)

        @llm_retry()
        async def _generate_report():
            return await structured_llm.ainvoke(
                messages + [HumanMessage(
                    content="Based on your analysis above, generate the VisualReport "
                            "structured output now. Include all data values, metrics, "
                            "and insights. Make sure every chart has complete data arrays."
                )]
            )

        report: VisualReport = await _generate_report()

        # Build complete ECharts options for each section
        used_features: list[str] = []
        sections_with_options = []

        for section in report.sections:
            chart_spec = section.chart.model_dump()

            # Build ECharts option — handles template search, validation, merge, formatting
            echarts_option = build_echarts_option(
                chart_spec,
                exclude_features=used_features,
            )

            # Track used features for diversity in multi-chart reports
            # (next search will try to find a different style)
            # Features come from the template search result
            # We approximate by tracking chart types used
            used_features.append(chart_spec.get("type", ""))

            sections_with_options.append({
                "heading": section.heading,
                "narrative": section.narrative,
                "priority": section.priority,
                "echarts_option": echarts_option,
            })

        visual_report = {
            "title": report.title,
            "subtitle": report.subtitle,
            "metrics": [m.model_dump() for m in report.metrics],
            "sections": sections_with_options,
            "insights": report.insights,
        }

        logger.info("visualizer_success", plan_index=plan_index,
                     section_count=len(sections_with_options))

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
