"""Chart template tools — Pinecone search, shape validation, template merging.

Layer 2 (Stylist): Searches Pinecone for the best ECharts design template
    matching a visual description, with hard filtering on chart_type.
Layer 3 (Builder): Validates data shape, merges data into template via
    data_mapping paths, and injects unit formatters.

Falls back to static baseline templates when Pinecone is unavailable.
"""

from __future__ import annotations

import json
import os
import re
from copy import deepcopy
from typing import Any

from langchain_core.tools import tool

from weatherwise.logging import get_logger
from weatherwise.tools._baseline_templates import BASELINE_TEMPLATES

logger = get_logger(__name__)

# Mood-based color palettes
_MOOD_PALETTES = {
    "danger": ["#F44336", "#FF5722", "#FF9800", "#FFC107", "#FFEB3B"],
    "calm": ["#2196F3", "#03A9F4", "#00BCD4", "#009688", "#4CAF50"],
    "neutral": ["#5470c6", "#91cc75", "#fac858", "#ee6666", "#73c0de",
                "#3ba272", "#fc8452", "#9a60b4", "#ea7ccc"],
}

# Shape requirements per chart type — used by the validator
_SHAPE_REQUIREMENTS = {
    "line": {"needs": ["x_field", "y_fields"]},
    "bar": {"needs": ["x_field", "y_fields"]},
    "scatter": {"needs": ["x_field", "y_fields"]},
    "pie": {"needs_data_keys": ["name", "value"]},
    "donut": {"needs_data_keys": ["name", "value"]},
    "radar": {"needs": ["metrics"]},
    "gauge": {"needs_data_keys": ["name", "value"]},
    "heatmap": {"needs": ["x_field", "y_field", "value_field"]},
}

# ---------------------------------------------------------------------------
# Pinecone search (Layer 2)
# ---------------------------------------------------------------------------

_vectorstore_cache = None


def _get_chart_vectorstore():
    """Return a Pinecone vectorstore for chart template search."""
    global _vectorstore_cache
    if _vectorstore_cache is None:
        from langchain_openai import OpenAIEmbeddings
        from langchain_pinecone import PineconeVectorStore

        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.environ["OPENAI_API_KEY"],
        )
        _vectorstore_cache = PineconeVectorStore(
            index_name=os.environ["PINECONE_INDEX_NAME"],
            embedding=embeddings,
            pinecone_api_key=os.environ["PINECONE_API_KEY"],
            namespace="visual_templates",
        )
    return _vectorstore_cache


def search_chart_template(
    search_hint: str,
    chart_type: str,
    exclude_features: list[str] | None = None,
) -> dict:
    """Search Pinecone for the best matching ECharts template.

    Uses hard filtering on chart_type to prevent semantic drift.
    Falls back to baseline template if Pinecone fails or returns no match.

    Args:
        search_hint: Natural language description of desired visual style.
        chart_type: Chart type to filter on (line, bar, pie, etc.).
        exclude_features: Visual feature tags to exclude (for multi-chart diversity).

    Returns:
        Dict with keys: echarts_template, data_mapping, visual_features
    """
    try:
        vectorstore = _get_chart_vectorstore()

        # Build metadata filter — hard filter on chart_type
        metadata_filter = {
            "source_type": {"$eq": "chart_template"},
            "chart_type": {"$eq": chart_type},
        }

        results = vectorstore.similarity_search_with_score(
            query=search_hint,
            k=3,  # get top 3 for diversity filtering
            filter=metadata_filter,
        )

        # Filter by relevance threshold
        RELEVANCE_THRESHOLD = 0.3
        results = [(doc, score) for doc, score in results if score >= RELEVANCE_THRESHOLD]

        if results:
            # If exclude_features specified, prefer templates without those features
            if exclude_features:
                for doc, score in results:
                    features = doc.metadata.get("visual_features", [])
                    if not any(f in features for f in exclude_features):
                        # Found a template with different style
                        return _parse_template_metadata(doc.metadata)

            # Use top match
            doc, score = results[0]
            return _parse_template_metadata(doc.metadata)

    except Exception as e:
        logger.warning("chart_template_search_failed", error=str(e))

    # Fallback to baseline
    logger.info("chart_template_fallback", chart_type=chart_type)
    return _get_baseline(chart_type)


def _parse_template_metadata(metadata: dict) -> dict:
    """Parse stringified JSON fields from Pinecone metadata."""
    return {
        "echarts_template": json.loads(metadata["echarts_template"]),
        "data_mapping": json.loads(metadata["data_mapping"]),
        "visual_features": metadata.get("visual_features", []),
    }


def _get_baseline(chart_type: str) -> dict:
    """Get a static baseline template for the given chart type."""
    baseline = BASELINE_TEMPLATES.get(chart_type)
    if baseline:
        return {
            "echarts_template": deepcopy(baseline["echarts_template"]),
            "data_mapping": deepcopy(baseline["data_mapping"]),
            "visual_features": ["baseline"],
        }
    # Ultimate fallback — bar chart
    baseline = BASELINE_TEMPLATES["bar"]
    return {
        "echarts_template": deepcopy(baseline["echarts_template"]),
        "data_mapping": deepcopy(baseline["data_mapping"]),
        "visual_features": ["baseline"],
    }


# ---------------------------------------------------------------------------
# Shape Validator (Layer 3 - Step A)
# ---------------------------------------------------------------------------

def validate_shape(chart_spec: dict) -> dict:
    """Validate data shape matches chart type. Auto-downgrade if incompatible.

    Returns the (possibly modified) chart_spec with a valid chart type.
    """
    chart_type = chart_spec.get("type", "bar")
    reqs = _SHAPE_REQUIREMENTS.get(chart_type, {})

    # Check required fields
    if "needs" in reqs:
        missing = [f for f in reqs["needs"] if not chart_spec.get(f)]
        if missing:
            logger.warning("shape_downgrade", from_type=chart_type, to_type="bar",
                           reason=f"missing fields: {missing}")
            chart_spec = {**chart_spec, "type": "bar"}

    # Check data key requirements (pie, donut, gauge)
    if "needs_data_keys" in reqs:
        data = chart_spec.get("data", [])
        if not data:
            # Empty data for types that require it — downgrade
            logger.warning("shape_downgrade", from_type=chart_type, to_type="bar",
                           reason="empty data array")
            chart_spec = {**chart_spec, "type": "bar"}
        elif isinstance(data[0], dict):
            required_keys = reqs["needs_data_keys"]
            if not all(k in data[0] for k in required_keys):
                logger.warning("shape_downgrade", from_type=chart_type, to_type="bar",
                               reason=f"data missing keys: {required_keys}")
                chart_spec = {**chart_spec, "type": "bar"}

    return chart_spec


# ---------------------------------------------------------------------------
# Nested path utilities
# ---------------------------------------------------------------------------

_ARRAY_INDEX_RE = re.compile(r"^(.+)\[(\d+)\]$")


def _resolve_path(obj: Any, path: str) -> tuple[Any, str]:
    """Resolve a dot-notation path like 'series[0].data' to (parent_obj, final_key).

    Returns the parent object and the final key/index to set.
    """
    parts = path.split(".")
    current = obj

    for i, part in enumerate(parts[:-1]):
        match = _ARRAY_INDEX_RE.match(part)
        if match:
            key, idx = match.group(1), int(match.group(2))
            current = current[key][idx]
        else:
            current = current[part]

    final_part = parts[-1]
    match = _ARRAY_INDEX_RE.match(final_part)
    if match:
        key, idx = match.group(1), int(match.group(2))
        return current[key], idx

    return current, final_part


def set_nested_value(obj: dict, path: str, value: Any) -> None:
    """Set a value at a dot-notation path in a nested dict.

    Supports array indexing: 'series[0].data' sets obj['series'][0]['data'].
    """
    if not path:
        return
    parent, key = _resolve_path(obj, path)
    if isinstance(parent, list):
        parent[int(key)] = value
    else:
        parent[key] = value


def get_nested_value(obj: dict, path: str) -> Any:
    """Get a value at a dot-notation path in a nested dict."""
    current = obj
    for part in path.split("."):
        match = _ARRAY_INDEX_RE.match(part)
        if match:
            key, idx = match.group(1), int(match.group(2))
            current = current[key][idx]
        else:
            current = current[part]
    return current


# ---------------------------------------------------------------------------
# Template Merger (Layer 3 - Step B)
# ---------------------------------------------------------------------------

def merge_with_template(
    chart_spec: dict,
    template: dict,
    data_mapping: dict,
) -> dict:
    """Merge chart spec data into the ECharts template using data_mapping paths.

    Template owns: styling (colors, gradients, animations, emphasis effects)
    ChartSpec owns: data arrays, title, axis labels, reference lines, units

    Handles dynamic multi-series: if data has more series than template,
    clones the last series with its styling for additional data.
    """
    option = deepcopy(template)
    chart_type = chart_spec.get("type", "bar")

    # --- Title ---
    title_path = data_mapping.get("title_path")
    if title_path and chart_spec.get("title"):
        set_nested_value(option, title_path, chart_spec["title"])

    # --- Subtitle ---
    if chart_spec.get("subtitle"):
        if "title" in option and isinstance(option["title"], dict):
            option["title"]["subtext"] = chart_spec["subtitle"]

    # --- Cartesian charts (line, bar, scatter) ---
    if chart_type in ("line", "bar", "scatter"):
        _merge_cartesian(option, chart_spec, data_mapping)

    # --- Pie / Donut ---
    elif chart_type in ("pie", "donut"):
        _merge_pie(option, chart_spec, data_mapping)

    # --- Radar ---
    elif chart_type == "radar":
        _merge_radar(option, chart_spec, data_mapping)

    # --- Gauge ---
    elif chart_type == "gauge":
        _merge_gauge(option, chart_spec, data_mapping)

    # --- Heatmap ---
    elif chart_type == "heatmap":
        _merge_heatmap(option, chart_spec, data_mapping)

    # --- Legend ---
    legend_path = data_mapping.get("legend_path")
    if legend_path and chart_type in ("line", "bar", "scatter", "radar"):
        y_fields = chart_spec.get("y_fields", [])
        if len(y_fields) > 1:
            set_nested_value(option, legend_path, y_fields)

    # --- Mood palette ---
    mood = chart_spec.get("mood")
    if mood and mood in _MOOD_PALETTES:
        option["color"] = _MOOD_PALETTES[mood]

    return option


def _merge_cartesian(option: dict, spec: dict, mapping: dict) -> None:
    """Fill cartesian chart (line/bar/scatter) with data."""
    x_field = spec.get("x_field")
    y_fields = spec.get("y_fields", [])
    data = spec.get("data", [])
    series_paths = mapping.get("series_paths", [])

    # X-axis data
    x_path = mapping.get("x_axis_path")
    if x_path and x_field:
        set_nested_value(option, x_path, [d.get(x_field, "") for d in data])

    # Y-axis label
    y_name_path = mapping.get("y_axis_name_path")
    if y_name_path and spec.get("y_label"):
        set_nested_value(option, y_name_path, spec["y_label"])

    # X-axis label
    if spec.get("x_label") and "xAxis" in option:
        x_axis = option["xAxis"]
        if isinstance(x_axis, list):
            x_axis[0]["name"] = spec["x_label"]
        elif isinstance(x_axis, dict):
            x_axis["name"] = spec["x_label"]

    # Series data — fill existing paths, clone for extras
    for i, field in enumerate(y_fields):
        values = [d.get(field) for d in data]

        if i < len(series_paths):
            set_nested_value(option, series_paths[i], values)
            # Set series name — replace only the LAST .data segment
            name_path = series_paths[i].rsplit(".data", 1)
            name_path = ".name".join(name_path) if len(name_path) == 2 else series_paths[i]
            try:
                set_nested_value(option, name_path, field)
            except (KeyError, IndexError):
                pass
        else:
            # Dynamic multi-series: clone last series with its styling
            if option.get("series"):
                last_series = deepcopy(option["series"][-1])
                last_series["data"] = values
                last_series["name"] = field
                option["series"].append(last_series)

    # Reference lines → markLine on first series
    ref_lines = spec.get("reference_lines", [])
    if ref_lines and option.get("series"):
        option["series"][0]["markLine"] = {
            "symbol": "none",
            "silent": True,
            "data": [
                {
                    **({"yAxis": rl["value"]} if rl.get("axis", "y") == "y" else {"xAxis": rl["value"]}),
                    "name": rl.get("label", ""),
                    "label": {"formatter": rl.get("label", ""), "position": "insideEndTop"},
                    "lineStyle": {"type": rl.get("style", "dashed"), "color": "#ff4444"},
                }
                for rl in ref_lines
            ],
        }

    # Horizontal bars
    if spec.get("horizontal") and "xAxis" in option and "yAxis" in option:
        option["xAxis"], option["yAxis"] = option["yAxis"], option["xAxis"]

    # Stacked
    if spec.get("stacked") and option.get("series"):
        for s in option["series"]:
            s["stack"] = "total"

    # Dual axis
    if spec.get("dual_axis") and option.get("series") and len(option.get("series", [])) >= 2:
        y_axis = option.get("yAxis", {})
        if isinstance(y_axis, dict):
            option["yAxis"] = [y_axis, deepcopy(y_axis)]
        if isinstance(option["yAxis"], list) and len(option["yAxis"]) >= 2:
            option["series"][1]["yAxisIndex"] = 1

    # Show labels
    if spec.get("show_labels") and option.get("series"):
        for s in option["series"]:
            s.setdefault("label", {})["show"] = True
            s["label"].setdefault("position", "top")


def _merge_pie(option: dict, spec: dict, mapping: dict) -> None:
    """Fill pie/donut chart with data."""
    series_paths = mapping.get("series_paths", ["series[0].data"])
    if series_paths:
        set_nested_value(option, series_paths[0], spec.get("data", []))


def _merge_radar(option: dict, spec: dict, mapping: dict) -> None:
    """Fill radar chart with data."""
    category_path = mapping.get("category_path")
    if category_path and spec.get("metrics"):
        max_values = spec.get("max_values", [100] * len(spec["metrics"]))
        indicators = [
            {"name": m, "max": mv}
            for m, mv in zip(spec["metrics"], max_values)
        ]
        set_nested_value(option, category_path, indicators)

    series_paths = mapping.get("series_paths", ["series[0].data"])
    if series_paths and spec.get("data"):
        radar_data = [
            {"value": d.get("values", []), "name": d.get("name", "")}
            for d in spec["data"]
        ]
        set_nested_value(option, series_paths[0], radar_data)

    # Legend
    legend_path = mapping.get("legend_path")
    if legend_path and spec.get("data"):
        legend_names = [d.get("name", "") for d in spec["data"]]
        set_nested_value(option, legend_path, legend_names)


def _merge_gauge(option: dict, spec: dict, mapping: dict) -> None:
    """Fill gauge chart with data."""
    series_paths = mapping.get("series_paths", ["series[0].data"])
    if series_paths:
        set_nested_value(option, series_paths[0], spec.get("data", []))

    # Min/max
    if spec.get("min") is not None and option.get("series"):
        option["series"][0]["min"] = spec["min"]
    if spec.get("max") is not None and option.get("series"):
        option["series"][0]["max"] = spec["max"]


def _merge_heatmap(option: dict, spec: dict, mapping: dict) -> None:
    """Fill heatmap chart with data."""
    data = spec.get("data", [])
    x_field = spec.get("x_field", "")
    y_field = spec.get("y_field", "")
    value_field = spec.get("value_field", "")

    if not (x_field and y_field and value_field):
        return

    x_labels = list(dict.fromkeys(d.get(x_field, "") for d in data))
    y_labels = list(dict.fromkeys(d.get(y_field, "") for d in data))

    x_path = mapping.get("x_axis_path")
    if x_path:
        set_nested_value(option, x_path, x_labels)

    # Set y-axis data
    if "yAxis" in option:
        y_axis = option["yAxis"]
        if isinstance(y_axis, dict):
            y_axis["data"] = y_labels
        elif isinstance(y_axis, list) and y_axis:
            y_axis[0]["data"] = y_labels

    # Transform data to [x_index, y_index, value] format
    # Use dict lookups instead of index() to avoid ValueError on missing labels
    x_lookup = {label: i for i, label in enumerate(x_labels)}
    y_lookup = {label: i for i, label in enumerate(y_labels)}
    heatmap_data = []
    for d in data:
        x_val = d.get(x_field, "")
        y_val = d.get(y_field, "")
        if x_val in x_lookup and y_val in y_lookup:
            heatmap_data.append([x_lookup[x_val], y_lookup[y_val], d.get(value_field, 0)])

    series_paths = mapping.get("series_paths", ["series[0].data"])
    if series_paths:
        set_nested_value(option, series_paths[0], heatmap_data)

    # Update visualMap range
    all_values = [d.get(value_field, 0) for d in data]
    if all_values and "visualMap" in option:
        option["visualMap"]["min"] = min(all_values)
        option["visualMap"]["max"] = max(all_values)


# ---------------------------------------------------------------------------
# Formatter Injector (Layer 3 - Step C)
# ---------------------------------------------------------------------------

def inject_formatters(option: dict, unit: str) -> dict:
    """Replace {UNIT} placeholders throughout the template with the actual unit.

    This handles the "Ghost Formatter" problem — JS functions stripped during
    template extraction are replaced with {UNIT} placeholders. This function
    injects the real unit string everywhere.
    """
    if not unit:
        return option

    # Stringify → replace → parse back
    option_str = json.dumps(option)
    option_str = option_str.replace("{UNIT}", unit)
    option_str = option_str.replace("{unit}", unit)
    return json.loads(option_str)


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

def build_echarts_option(
    chart_spec: dict,
    exclude_features: list[str] | None = None,
) -> dict:
    """Full pipeline: validate → search template → merge → inject formatters.

    This is the main entry point for the visualizer agent. Takes a ChartSpec
    (from the LLM) and returns a complete ECharts option object ready for
    frontend rendering.

    Args:
        chart_spec: Chart specification from the visualizer LLM.
        exclude_features: Visual features to avoid (for multi-chart diversity).

    Returns:
        Complete ECharts option dict.
    """
    # Step A: Validate shape, auto-downgrade if needed
    chart_spec = validate_shape(chart_spec)

    # Step B: Search for best matching template
    chart_type = chart_spec.get("type", "bar")
    search_hint = chart_spec.get("search_hint", chart_spec.get("title", ""))
    template_data = search_chart_template(search_hint, chart_type, exclude_features)

    template = template_data["echarts_template"]
    data_mapping = template_data["data_mapping"]

    # Step C: Merge data into template
    option = merge_with_template(chart_spec, template, data_mapping)

    # Step D: Inject unit formatters
    unit = chart_spec.get("unit", "")
    option = inject_formatters(option, unit)

    logger.info("echarts_option_built",
                chart_type=chart_type,
                template_features=template_data.get("visual_features", []))

    return option
