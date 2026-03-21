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
visual reports that tell a complete story — not just plot numbers.

THINK BEFORE YOU BUILD:
1. What patterns exist in this data? Look for any meaningful structure — trends, \
comparisons, distributions, anomalies, correlations, or anything else worth showing.
2. What would a viewer learn from seeing this visually that they can't get from text?
3. What data SUPPORTS the insight? Always show the underlying data, not just the summary.
4. How many chart sections does this story need? Use your judgment — sometimes one chart \
is enough, sometimes the data has multiple dimensions that each deserve their own view.
5. What context makes each chart self-explanatory? Consider reference lines, annotations, \
axis labels with units, and any other elements that help the viewer understand \
the chart without reading surrounding text.

DATA ACCESS:
The weather data is provided as structured JSON in your prompt. Read it directly. \
The JSON is a dict keyed by location (e.g., data["phoenix_0"]). Each value is the \
structured API response with metadata and records.

ACCURACY: Before generating any chart, verify the data you are about to use. Count the records, \
check the date range, and confirm the values match what is in the JSON. Do not assume or \
approximate — if you are unsure about a value, double-check it in the JSON.

GENERATING CHARTS:
For each chart section, generate a COMPLETE ECharts option JSON object in the \
echarts_option field. This must be valid JSON that ECharts can render directly. \
Include title, tooltip, appropriate axes with labels and units, and series with data. \
The frontend applies a dark theme automatically — do not set background colors, \
but do set text colors to light values (#ccc, #999) for readability.

Make charts visually rich and professional. When comparing values with different \
units or scales (for example, temperature and rainfall), use separate charts or \
dual axes to keep both readable. Ensure chart elements (title, legend, axis labels, \
annotations) do not overlap — position the legend where it does not collide with \
axis labels or other elements. Do not use custom tooltip formatters — ECharts \
default tooltips are clean and always accurate. Just set trigger type (axis or item).

If the user explicitly requests a specific chart type, use that type.

CRITICAL: Your charts must be based entirely on the actual fetched data provided to you. \
If the data contains zero records or no values for a metric, do NOT create charts for it — \
report the gap instead. Never fill in charts with general reference information or \
training knowledge. If the data is a map URL (not weather records with numeric values), \
do not create ECharts charts — the map itself is the visualization.

DATA ACCURACY: Represent the complete dataset accurately. For small datasets, include every \
data point. For large or multi-location datasets, summarize intelligently — use aggregates, \
comparisons, or rankings to keep charts readable and within a reasonable size.

METRICS: Highlight key numbers as metric cards with units.
INSIGHTS: Provide meaningful analytical observations about what the data reveals.
PRIORITY: priority=1 for the main chart (full width), priority=2 for supporting \
charts (side-by-side)."""

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

        @llm_retry()
        async def _generate_report():
            return await structured_llm.ainvoke([
                SystemMessage(content=[
                    {"type": "text", "text": VISUALIZER_SYSTEM_PROMPT,
                     "cache_control": {"type": "ephemeral"}},
                ]),
                HumanMessage(content=structured_context),
            ])

        # Retry once on ValidationError (malformed ECharts JSON on first attempt)
        try:
            report: VisualReport = await _generate_report()
        except ValidationError:
            logger.warning("visualizer_validation_retry", plan_index=plan_index)
            report = await _generate_report()

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
