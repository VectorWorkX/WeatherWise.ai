"""One-time indexing pipeline for ECharts visualization templates.

Fetches curated ECharts examples from the Apache ECharts GitHub repo,
processes each with an LLM to extract:
  1. Clean JSON template (JS functions replaced with {VALUE} {UNIT} placeholders)
  2. Data mapping paths (where data goes in the template structure)
  3. Semantic visual descriptor + visual feature tags

Indexes everything in Pinecone namespace "visual_templates" for the
visualizer agent to search at runtime.

Also generates static baseline templates as a Python module for fallback
when Pinecone is unavailable.

Usage::

    python scripts/build_chart_templates.py --all
    python scripts/build_chart_templates.py --dry-run 5
    python scripts/build_chart_templates.py --id line-smooth
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Curated list of 75 ECharts examples — distributed across chart types
# ---------------------------------------------------------------------------
CHART_EXAMPLES: list[dict[str, str]] = [
    # --- LINE variants (15) ---
    {"id": "line-simple", "chart_type": "line"},
    {"id": "line-smooth", "chart_type": "line"},
    {"id": "line-stack", "chart_type": "line"},
    {"id": "line-marker", "chart_type": "line"},
    {"id": "line-step", "chart_type": "line"},
    {"id": "line-gradient", "chart_type": "line"},
    {"id": "line-easing", "chart_type": "line"},
    {"id": "line-pen", "chart_type": "line"},
    {"id": "line-race", "chart_type": "line"},
    {"id": "area-basic", "chart_type": "line"},
    {"id": "area-stack", "chart_type": "line"},
    {"id": "area-stack-gradient", "chart_type": "line"},
    {"id": "area-pieces", "chart_type": "line"},
    {"id": "bump-chart", "chart_type": "line"},
    {"id": "cycle-plot", "chart_type": "line"},

    # --- BAR variants (13) ---
    {"id": "bar-simple", "chart_type": "bar"},
    {"id": "bar-background", "chart_type": "bar"},
    {"id": "bar-data-color", "chart_type": "bar"},
    {"id": "bar-gradient", "chart_type": "bar"},
    {"id": "bar-stack", "chart_type": "bar"},
    {"id": "bar-stack-borderRadius", "chart_type": "bar"},
    {"id": "bar-y-category", "chart_type": "bar"},
    {"id": "bar-y-category-stack", "chart_type": "bar"},
    {"id": "bar-negative2", "chart_type": "bar"},
    {"id": "bar-waterfall", "chart_type": "bar"},
    {"id": "bar-waterfall2", "chart_type": "bar"},
    {"id": "bar-label-rotation", "chart_type": "bar"},
    {"id": "bar-polar-label-radial", "chart_type": "bar"},

    # --- PIE / DONUT variants (9) ---
    {"id": "pie-simple", "chart_type": "pie"},
    {"id": "pie-doughnut", "chart_type": "donut"},
    {"id": "pie-borderRadius", "chart_type": "donut"},
    {"id": "pie-roseType", "chart_type": "pie"},
    {"id": "pie-roseType-simple", "chart_type": "pie"},
    {"id": "pie-half-donut", "chart_type": "donut"},
    {"id": "pie-custom", "chart_type": "pie"},
    {"id": "pie-pattern", "chart_type": "pie"},
    {"id": "pie-padAngle", "chart_type": "pie"},

    # --- RADAR variants (5) ---
    {"id": "radar", "chart_type": "radar"},
    {"id": "radar-aqi", "chart_type": "radar"},
    {"id": "radar-custom", "chart_type": "radar"},
    {"id": "radar-multiple", "chart_type": "radar"},
    {"id": "parallel-aqi", "chart_type": "radar"},

    # --- GAUGE variants (7) ---
    {"id": "gauge", "chart_type": "gauge"},
    {"id": "gauge-simple", "chart_type": "gauge"},
    {"id": "gauge-speed", "chart_type": "gauge"},
    {"id": "gauge-progress", "chart_type": "gauge"},
    {"id": "gauge-stage", "chart_type": "gauge"},
    {"id": "gauge-grade", "chart_type": "gauge"},
    {"id": "gauge-multi", "chart_type": "gauge"},

    # --- SCATTER variants (6) ---
    {"id": "scatter-simple", "chart_type": "scatter"},
    {"id": "scatter-effect", "chart_type": "scatter"},
    {"id": "scatter-linear-regression", "chart_type": "scatter"},
    {"id": "scatter-exponential-regression", "chart_type": "scatter"},
    {"id": "scatter-polynomial-regression", "chart_type": "scatter"},
    {"id": "scatter-clustering", "chart_type": "scatter"},

    # --- HEATMAP variants (5) ---
    {"id": "heatmap-cartesian", "chart_type": "heatmap"},
    {"id": "heatmap-large", "chart_type": "heatmap"},
    {"id": "heatmap-large-piecewise", "chart_type": "heatmap"},
    {"id": "calendar-heatmap", "chart_type": "heatmap"},
    {"id": "calendar-simple", "chart_type": "heatmap"},

    # --- CANDLESTICK variants (3) ---
    {"id": "candlestick-simple", "chart_type": "candlestick"},
    {"id": "candlestick-sh", "chart_type": "candlestick"},
    {"id": "candlestick-large", "chart_type": "candlestick"},

    # --- MIXED / COMBO / DATASET (5) ---
    {"id": "data-transform-sort-bar", "chart_type": "bar"},
    {"id": "dataset-encode0", "chart_type": "bar"},
    {"id": "dataset-encode1", "chart_type": "line"},
    {"id": "dataset-default", "chart_type": "bar"},
    {"id": "data-transform-filter", "chart_type": "line"},

    # --- MATRIX / CORRELATION (4) ---
    {"id": "matrix-simple", "chart_type": "heatmap"},
    {"id": "matrix-correlation-heatmap", "chart_type": "heatmap"},
    {"id": "matrix-correlation-scatter", "chart_type": "scatter"},
    {"id": "matrix-covariance", "chart_type": "heatmap"},

    # --- PARALLEL (3) ---
    {"id": "parallel-simple", "chart_type": "parallel"},

    # --- ADDITIONAL (2) ---
    {"id": "bar-tick-align", "chart_type": "bar"},
    {"id": "line-in-cartesian-coordinate-system", "chart_type": "line"},
]

# Verify count
assert len(CHART_EXAMPLES) == 75, f"Expected 75 examples, got {len(CHART_EXAMPLES)}"


# ---------------------------------------------------------------------------
# GitHub API helper
# ---------------------------------------------------------------------------

def fetch_example_source(example_id: str) -> str | None:
    """Fetch an ECharts example's TypeScript source from the GitHub repo."""
    path = f"public/examples/ts/{example_id}.ts"
    try:
        result = subprocess.run(
            ["gh", "api", f"repos/apache/echarts-examples/contents/{path}",
             "--jq", ".content"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            # Try .js extension as fallback
            path_js = f"public/examples/ts/{example_id}.js"
            result = subprocess.run(
                ["gh", "api", f"repos/apache/echarts-examples/contents/{path_js}",
                 "--jq", ".content"],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode != 0:
                print(f"  [SKIP] Could not fetch {example_id}: {result.stderr.strip()}")
                return None

        import base64
        content = base64.b64decode(result.stdout.strip()).decode("utf-8")
        return content

    except Exception as e:
        print(f"  [ERROR] Failed to fetch {example_id}: {e}")
        return None


# ---------------------------------------------------------------------------
# LLM processing
# ---------------------------------------------------------------------------

def _get_llm():
    """Return a Claude Haiku instance for template processing."""
    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        api_key=os.environ["ANTHROPIC_API_KEY"],
        max_tokens=4096,
        temperature=0,
    )


def process_example(source_code: str, example_id: str, chart_type: str) -> dict | None:
    """Use LLM to extract template, data mapping, descriptor, and features."""
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = _get_llm()

    prompt = f"""You are an ECharts expert. Analyze this ECharts example source code and extract three things.

SOURCE CODE:
```
{source_code}
```

Return a JSON object with exactly these keys:

1. "echarts_template": The ECharts option object as clean JSON.
   - Replace ALL sample/demo data arrays with empty arrays []
   - Replace ALL JavaScript functions (formatters, callbacks) with string placeholders using {{VALUE}} and {{UNIT}} patterns
   - Keep ALL styling properties (colors, gradients, shadows, animations, emphasis effects, borderRadius, etc.)
   - Keep markPoint, markLine, markArea configurations but empty their data arrays
   - Ensure the JSON is valid (no trailing commas, no JS syntax)

2. "data_mapping": An object describing WHERE data goes in the template.
   Use dot-notation paths. Include:
   - "title_path": path to the title text (e.g., "title.text")
   - "x_axis_path": path to x-axis data array (e.g., "xAxis.data" or "xAxis[0].data"), null if not applicable
   - "y_axis_name_path": path to y-axis label (e.g., "yAxis.name"), null if not applicable
   - "series_paths": list of paths to series data arrays (e.g., ["series[0].data", "series[1].data"])
   - "legend_path": path to legend data array (e.g., "legend.data"), null if not applicable
   - "category_path": for radar charts, path to indicator array, null otherwise

3. "description": A visual description of what this chart LOOKS like.
   Describe ONLY visual properties: layout, colors, gradients, animations, emphasis effects, spacing, typography.
   Do NOT mention what kind of data it shows or any domain context.
   Example: "Smooth line chart with semi-transparent gradient area fill transitioning from vivid blue to transparent. Features rounded markPoints at extremes and a dashed horizontal reference line. Tooltip uses cross-style axis pointer."

4. "visual_features": A list of 3-6 tags describing the visual style.
   Choose from: gradient, smooth, animated, minimal, flat, high-contrast, rounded, stacked, polar, 3D, filled-area, step, dashed, shadow, glow, transparent, outlined, labeled, icon-markers, data-zoom, progress-bar, multi-axis, rich-label

Return ONLY valid JSON. No markdown, no explanation."""

    try:
        response = llm.invoke([
            SystemMessage(content="You are a precise JSON extraction tool. Return only valid JSON."),
            HumanMessage(content=prompt),
        ])

        content = response.content.strip()
        # Strip markdown code blocks if present
        if content.startswith("```"):
            content = content.split("\n", 1)[1]
            if content.endswith("```"):
                content = content.rsplit("```", 1)[0]
            content = content.strip()

        result = json.loads(content)

        # Validate required keys
        required = ["echarts_template", "data_mapping", "description", "visual_features"]
        for key in required:
            if key not in result:
                print(f"  [WARN] Missing key '{key}' in LLM output for {example_id}")
                return None

        return result

    except json.JSONDecodeError as e:
        print(f"  [ERROR] Invalid JSON from LLM for {example_id}: {e}")
        return None
    except Exception as e:
        print(f"  [ERROR] LLM processing failed for {example_id}: {e}")
        return None


# ---------------------------------------------------------------------------
# Pinecone upsert
# ---------------------------------------------------------------------------

def _get_pinecone_index():
    """Return the Pinecone index instance."""
    from pinecone import Pinecone
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    return pc.Index(os.environ["PINECONE_INDEX_NAME"])


def _get_embeddings():
    """Return the OpenAI embeddings instance."""
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.environ["OPENAI_API_KEY"],
    )


def upsert_template(
    index,
    embeddings,
    example_id: str,
    chart_type: str,
    processed: dict,
) -> None:
    """Embed the descriptor and upsert to Pinecone with stringified metadata."""

    description = processed["description"]
    embedding = embeddings.embed_query(description)

    # Pinecone does NOT allow nested objects in metadata.
    # Stringify complex objects before upserting.
    metadata = {
        "source_type": "chart_template",
        "chart_type": chart_type,
        "example_id": example_id,
        "description": description,
        "visual_features": processed["visual_features"],  # list of strings is OK
        "echarts_template": json.dumps(processed["echarts_template"]),  # STRINGIFIED
        "data_mapping": json.dumps(processed["data_mapping"]),  # STRINGIFIED
    }

    vector_id = f"chart_template_{example_id}"

    index.upsert(
        vectors=[{
            "id": vector_id,
            "values": embedding,
            "metadata": metadata,
        }],
        namespace="visual_templates",
    )

    print(f"  [OK] Upserted {example_id} (chart_type={chart_type}, "
          f"features={processed['visual_features']})")


# ---------------------------------------------------------------------------
# Baseline template generation (static fallback)
# ---------------------------------------------------------------------------

BASELINE_CHART_TYPES = ["line", "bar", "pie", "donut", "radar", "gauge", "scatter", "heatmap"]


def generate_baseline_templates(all_processed: dict[str, dict]) -> None:
    """Generate _baseline_templates.py with one fallback template per chart type.

    Picks the simplest (first) processed template for each chart type.
    """
    baselines: dict[str, dict] = {}

    for chart_type in BASELINE_CHART_TYPES:
        # Find the first processed example of this chart type
        for ex in CHART_EXAMPLES:
            if ex["chart_type"] == chart_type and ex["id"] in all_processed:
                processed = all_processed[ex["id"]]
                baselines[chart_type] = {
                    "echarts_template": processed["echarts_template"],
                    "data_mapping": processed["data_mapping"],
                }
                break

        if chart_type not in baselines:
            print(f"  [WARN] No baseline template found for chart_type={chart_type}")

    # Write Python module
    output_path = PROJECT_ROOT / "src" / "weatherwise" / "tools" / "_baseline_templates.py"
    with open(output_path, "w") as f:
        f.write('"""Static baseline ECharts templates — fallback when Pinecone is unavailable.\n\n')
        f.write('Auto-generated by scripts/build_chart_templates.py. Do not edit manually.\n')
        f.write('"""\n\n')
        f.write("import json\n\n")
        f.write("# Each entry: {echarts_template: dict, data_mapping: dict}\n")
        f.write(f"BASELINE_TEMPLATES: dict[str, dict] = json.loads('''\n")
        f.write(json.dumps(baselines, indent=2))
        f.write("\n''')\n")

    print(f"  [OK] Generated baseline templates at {output_path}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    examples: list[dict[str, str]],
    dry_run: bool = False,
) -> None:
    """Run the full indexing pipeline."""
    print(f"\n{'='*60}")
    print(f"ECharts Template Indexing Pipeline")
    print(f"Processing {len(examples)} examples")
    print(f"{'='*60}\n")

    if not dry_run:
        index = _get_pinecone_index()
        embeddings = _get_embeddings()
    else:
        index = None
        embeddings = None

    all_processed: dict[str, dict] = {}
    success = 0
    failed = 0

    for i, example in enumerate(examples):
        eid = example["id"]
        ctype = example["chart_type"]
        print(f"[{i+1}/{len(examples)}] {eid} (chart_type={ctype})")

        # Step 1: Fetch source from GitHub
        source = fetch_example_source(eid)
        if not source:
            failed += 1
            continue

        # Step 2: LLM processing
        processed = process_example(source, eid, ctype)
        if not processed:
            failed += 1
            continue

        all_processed[eid] = processed

        if dry_run:
            print(f"  [DRY RUN] description: {processed['description'][:80]}...")
            print(f"  [DRY RUN] features: {processed['visual_features']}")
            print(f"  [DRY RUN] mapping: {json.dumps(processed['data_mapping'], indent=2)}")
        else:
            # Step 3: Upsert to Pinecone
            try:
                upsert_template(index, embeddings, eid, ctype, processed)
                success += 1
            except Exception as e:
                print(f"  [ERROR] Pinecone upsert failed for {eid}: {e}")
                failed += 1
                continue

        # Rate limiting — be nice to GitHub API and Anthropic
        time.sleep(1.0)

    # Generate baseline templates
    if all_processed:
        print(f"\nGenerating baseline templates...")
        generate_baseline_templates(all_processed)

    print(f"\n{'='*60}")
    print(f"Pipeline complete: {success} succeeded, {failed} failed")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Index ECharts templates to Pinecone")
    parser.add_argument("--all", action="store_true", help="Process all 75 examples")
    parser.add_argument("--dry-run", type=int, metavar="N",
                        help="Process first N examples without upserting (test mode)")
    parser.add_argument("--id", type=str, help="Process a single example by ID")

    args = parser.parse_args()

    if args.id:
        examples = [ex for ex in CHART_EXAMPLES if ex["id"] == args.id]
        if not examples:
            print(f"Unknown example ID: {args.id}")
            print(f"Available: {[e['id'] for e in CHART_EXAMPLES]}")
            sys.exit(1)
        run_pipeline(examples)

    elif args.dry_run:
        run_pipeline(CHART_EXAMPLES[:args.dry_run], dry_run=True)

    elif args.all:
        run_pipeline(CHART_EXAMPLES)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
