"""Visualizer agent node — generates visual stories from weather data.

Single structured output call: the LLM reads structured JSON data directly
and generates a complete VisualReport with ECharts options. No ReAct loop,
no code execution for parsing — the LLM reads the data and builds the chart.

The frontend renders echarts_option directly with theme="dark".
"""

from __future__ import annotations

import json
import os
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, ValidationError, field_validator

from weatherwise.logging import get_logger
from weatherwise.retry import llm_retry
from weatherwise.state import VisualizePayload

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Structured output models
# ---------------------------------------------------------------------------

def _parse_json_string(v: Any) -> Any:
    """Parse a value that might be a JSON string instead of a parsed object.

    The Anthropic API sometimes serializes deeply nested fields as JSON strings
    in tool_use responses. This handles both cases transparently.
    """
    if isinstance(v, str):
        try:
            return json.loads(v)
        except (json.JSONDecodeError, ValueError):
            return v
    return v


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

    @field_validator("echarts_option", mode="before")
    @classmethod
    def parse_echarts_string(cls, v: Any) -> Any:
        return _parse_json_string(v)


class VisualReport(BaseModel):
    title: str
    subtitle: str = ""
    metrics: list[MetricCard] = Field(default_factory=list)
    sections: list[ReportSection] = Field(min_length=1)
    insights: list[str] = Field(min_length=1)

    @field_validator("sections", mode="before")
    @classmethod
    def parse_sections_string(cls, v: Any) -> Any:
        return _parse_json_string(v)

    @field_validator("metrics", mode="before")
    @classmethod
    def parse_metrics_string(cls, v: Any) -> Any:
        return _parse_json_string(v)

    @field_validator("insights", mode="before")
    @classmethod
    def parse_insights_string(cls, v: Any) -> Any:
        return _parse_json_string(v)


# ---------------------------------------------------------------------------
# Visualizer LLM
# ---------------------------------------------------------------------------

VISUALIZER_SYSTEM_PROMPT = """You are a weather data visualization analyst. Your job is to create \
visual reports that tell a complete story — not just plot numbers. A basic chart that shows \
a single value is a waste. Every chart should reveal something the viewer cannot get from text.

VISUALIZATION PHILOSOPHY:
Never settle for the minimum chart. Always think one level deeper:
- If the query asks for a mean, show the DISTRIBUTION with the mean as a reference line — \
the viewer sees both the answer AND the underlying pattern.
- If comparing two locations, overlay them on the SAME chart axes so differences are \
immediately visible. Use separate series with distinct colors, not separate charts.
- If showing a forecast, include the recent actuals too — context makes the forecast meaningful.
- If the data has a time dimension, show every data point as a series and annotate the key \
statistics (mean, min, max) as markLines or markPoints on the chart. One rich chart beats \
three simple ones.
- If data has multiple metrics (temperature, humidity, wind, pressure), consider whether a \
multi-axis chart, a radar overlay, or a parallel coordinates view tells the comparison story \
better than separate charts.

THINK BEFORE YOU BUILD:
1. What is the MOST INSIGHTFUL way to show this data? Not the simplest — the most revealing.
2. What patterns, anomalies, or relationships exist? Trends, cycles, outliers, correlations?
3. How many chart sections does this story ACTUALLY need? Sometimes one dense, rich chart \
with multiple series and annotations tells the story better than three sparse charts. \
Other times, different dimensions genuinely need separate views. Use judgment.
4. What reference context makes the chart self-explanatory? Threshold lines (freezing point, \
record highs, historical averages), annotations on outliers, shaded ranges for normal bands.

DATA ACCESS:
The weather data is provided as structured JSON in your prompt. Read it directly. \
The JSON is a dict keyed by location (e.g., data["phoenix_0"]). Each value is the \
structured API response with metadata and records.

ACCURACY: Before generating any chart, verify the data you are about to use. Count the records, \
check the date range, and confirm the values match what is in the JSON. Do not assume or \
approximate — if you are unsure about a value, double-check it in the JSON.

CHART TYPE SELECTION — choose the type that reveals the most insight:
You have the FULL ECharts library available. Choose the chart type that best fits the data \
shape and the question being asked. Do not default to line/bar when another type tells a \
better story. Let the data guide you:
- Time-series with many points: line chart with area fill, smooth curves, and dataZoom \
for exploration
- Single current-condition metric (wind speed, UV, AQI): gauge chart — instant readability
- Comparing multi-metric profiles across locations: radar chart — one glance shows which \
city is hotter, windier, more humid
- Daily high/low ranges: candlestick or custom range bar — shows spread, not just averages
- Long-period anomalies (30 days of deviation): heatmap or calendar view — patterns emerge
- Distributions and density: scatter plot with trend line or box-plot style markings
- Part-of-whole composition (precipitation types, wind direction): pie/donut or polar bar
- Relationships between metrics: scatter with regression or bubble chart

INTERACTIVITY AND POLISH:
Make every chart interactive and polished. Include these ECharts features:
- animation: true, animationDuration: 800, animationEasing: "cubicOut" — smooth entrance
- tooltip with trigger: "axis" for time-series, "item" for categorical. Do NOT use custom \
tooltip formatters — ECharts defaults are clean and always accurate.
- For time-series with 10+ data points: include dataZoom: [{type: "slider", height: 20}, \
{type: "inside"}] so users can scroll and zoom through the data
- For temperature data: use visualMap with inRange colors from cool (#4575b4) to warm (#d73027) \
to make temperature patterns instantly visible through color
- Use markLine for meaningful thresholds: freezing point (0°C/32°F), record highs/lows if \
known, averages. These reference lines add context without extra text
- Use markPoint to annotate outliers or notable values (min, max) when they tell a story
- Include toolbox: {feature: {saveAsImage: {title: "Save"}}} so users can export charts
- Legend should be interactive (click to toggle series visibility). Position it where it \
does not collide with axis labels — typically top-right or bottom-center

STYLING (dark theme):
The frontend applies a dark theme automatically — do not set backgroundColor. \
Set text colors to light values (#ccc for primary, #999 for secondary). \
Use rich, distinct colors for multiple series — avoid similar hues that blend on dark \
backgrounds. Good palette: #60a5fa (blue), #f97316 (orange), #22c55e (green), \
#f43f5e (rose), #a78bfa (violet), #facc15 (yellow).

When comparing values with different units or scales, use dual yAxis with clear labeling — \
or separate charts if dual-axis would be confusing. Ensure grid.containLabel: true always.

CRITICAL: Your charts must be based entirely on the actual fetched data provided to you. \
If the data contains zero records or no values for a metric, do NOT create charts for it — \
report the gap instead. Never fill in charts with general reference information or \
training knowledge. If the data is a map URL (not weather records with numeric values), \
do not create ECharts charts — the map itself is the visualization.

DATA ACCURACY: Represent the complete dataset accurately. For small datasets, include every \
data point. For large or multi-location datasets, summarize intelligently — use aggregates, \
comparisons, or rankings to keep charts readable and within a reasonable size.

METRICS: Highlight key numbers as metric cards with units and contextual detail.
INSIGHTS: Provide meaningful analytical observations — not restatements of numbers, but \
what the patterns MEAN (e.g., "Temperature variance is 3x higher in Delhi than San Jose, \
suggesting more volatile weather patterns").
PRIORITY: Use priority=1 for all sections (full width, stacked vertically)."""

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


# ---------------------------------------------------------------------------
# Visualizer node
# ---------------------------------------------------------------------------

async def visualizer_agent(state: VisualizePayload) -> dict[str, Any]:
    """Generate a visual story from fetched weather data.

    Single structured output call: the LLM reads raw JSON data directly
    and generates a VisualReport with complete ECharts options.
    No ReAct loop, no code execution — the LLM reads and builds.
    """
    query = state.get("query", "")
    plan_index = state["plan_index"]
    fetched_weather = state.get("fetched_weather", {})
    calculations = state.get("calculations", [])

    logger.info("visualizer_start", query=query, plan_index=plan_index)

    try:
        # Build data context as a JSON dict keyed by location
        data_dict: dict[str, Any] = {}
        for loc, weather_data in fetched_weather.items():
            if isinstance(weather_data, dict):
                raw = weather_data.get("raw_data") or weather_data.get("raw_response", str(weather_data))
            else:
                raw = str(weather_data)
            try:
                data_dict[loc] = json.loads(raw) if isinstance(raw, str) else raw
            except (json.JSONDecodeError, ValueError):
                data_dict[loc] = raw
        # Check if we have any structured data to visualize.
        # When fetchers fail, data_dict values are plain strings (error messages)
        # instead of parsed dicts with weather records. If no entry parsed into
        # a dict, there is nothing meaningful to chart.
        has_structured_data = any(isinstance(v, dict) for v in data_dict.values())
        if not has_structured_data and not calculations:
            logger.warning(
                "visualizer_skipped_no_data",
                plan_index=plan_index,
                data_types=[type(v).__name__ for v in data_dict.values()],
            )
            return {
                "visual_report": {},
                "completed_indices": [plan_index],
            }

        data_summary = json.dumps(data_dict)

        calc_context = ""
        if calculations:
            calc_context = "\n\nCalculation results:\n" + "\n".join(
                f"  - {c}" for c in calculations
            )

        # Single structured output call — LLM reads JSON directly
        structured_llm = _get_viz_llm().with_structured_output(VisualReport)

        structured_context = (
            f"Query: {query}\n\n"
            f"Weather data (JSON):\n{data_summary}"
            f"{calc_context}\n\n"
            f"Generate the VisualReport with complete ECharts option JSON "
            f"for each chart section. Include metrics and insights. "
            f"Use every data point from the JSON — do not omit any."
        )

        base_messages = [
            SystemMessage(content=[
                {"type": "text", "text": VISUALIZER_SYSTEM_PROMPT,
                 "cache_control": {"type": "ephemeral"}},
            ]),
            HumanMessage(content=structured_context),
        ]

        @llm_retry()
        async def _generate_report(messages):
            return await structured_llm.ainvoke(messages)

        # Retry on ValidationError with the error details included so the LLM
        # can correct the specific issue (e.g., malformed ECharts JSON, missing
        # required fields) instead of blindly regenerating the same output.
        try:
            report: VisualReport = await _generate_report(base_messages)
        except ValidationError as ve:
            logger.warning("visualizer_validation_retry", plan_index=plan_index,
                           error=str(ve))
            retry_messages = base_messages + [
                HumanMessage(content=(
                    f"Your previous response failed validation with this error:\n"
                    f"{ve}\n\n"
                    f"Please fix the issue and regenerate the VisualReport. "
                    f"Ensure all echarts_option fields are valid JSON objects, "
                    f"sections has at least 1 item, and insights has at least 1 item."
                )),
            ]
            report = await _generate_report(retry_messages)

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
