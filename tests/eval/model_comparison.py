"""Multi-model comparison runner for WeatherWise.ai.

Runs the same golden queries through the WeatherWise graph with different
LLM backends, judges all responses, and prints a comparison table.

Usage::

    python tests/eval/model_comparison.py
    python tests/eval/model_comparison.py --models claude-sonnet gemini-flash
    python tests/eval/model_comparison.py --judge-model gpt-4o --pairwise
    python tests/eval/model_comparison.py --langsmith
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from code_assertions import AssertionResult, run_assertions  # noqa: E402
from golden_responses import GOLDEN_RESPONSES, GoldenResponse  # noqa: E402
from langchain_core.messages import HumanMessage  # noqa: E402
from llm_judge import (  # noqa: E402
    BinaryVerdict,
    DimensionScore,
    JudgeVerdict,
    PairwiseVerdict,
    evaluate_criteria,
    evaluate_response,
    pairwise_evaluate_debiased,
)
from model_swap import reset_agent_llms, swap_agent_llms  # noqa: E402
from report import generate_html_report  # noqa: E402
from utils.model_registry import list_models  # noqa: E402


# ---------------------------------------------------------------------------
# Comparison result models
# ---------------------------------------------------------------------------


class ModelResult:
    """Result of evaluating one model on one query."""

    def __init__(
        self,
        model_key: str,
        golden: GoldenResponse,
        response: str,
        verdict: JudgeVerdict,
        assertions: Optional[List[AssertionResult]] = None,
        binary_verdict: Optional[BinaryVerdict] = None,
    ):
        self.model_key = model_key
        self.golden = golden
        self.response = response
        self.verdict = verdict
        self.assertions = assertions or []
        self.binary_verdict = binary_verdict


# ---------------------------------------------------------------------------
# Core comparison logic
# ---------------------------------------------------------------------------


async def run_single_model(
    model_key: str,
    goldens: List[GoldenResponse],
    judge_model_key: str = "gpt-4o",
) -> List[ModelResult]:
    """Run all golden queries through the graph with a specific model backend."""
    from graph.builder import build_graph
    from tools.math_tools import calculator
    from tools.mcp_bridge import get_mcp_client
    from tools.rag_tool import search_knowledge_base
    from tools.retry_wrapper import with_retry

    # Swap agent LLMs to the target model
    swap_agent_llms(model_key)

    # Build graph
    client = get_mcp_client()
    mcp_tools = with_retry(await client.get_tools())
    graph = build_graph(mcp_tools, search_knowledge_base, calculator)

    results = []
    for golden in goldens:
        print(f"  [{model_key}] Running: {golden.query[:60]}...")
        try:
            output = await graph.ainvoke({"messages": [HumanMessage(content=golden.query)]})
            response_text = output["messages"][-1].content
        except Exception as e:
            response_text = f"ERROR: {e}"

        # 1. Deterministic code assertions (fast, no LLM cost)
        assertions = run_assertions(response_text, golden)

        # 2. LLM judge — dimensional scoring
        try:
            verdict = evaluate_response(
                query=golden.query,
                actual_response=response_text,
                reference=golden,
                judge_model_key=judge_model_key,
            )
        except Exception as e:
            verdict = JudgeVerdict(
                dimension_scores=[DimensionScore(dimension="error", score=1, reasoning=str(e))],
                overall_score=0.0,
                overall_assessment=f"Judge failed: {e}",
                pass_fail="fail",
            )

        # 3. Binary PASS/FAIL per success criterion
        binary_verdict = None
        try:
            binary_verdict = evaluate_criteria(
                query=golden.query,
                actual_response=response_text,
                reference=golden,
                judge_model_key=judge_model_key,
            )
        except Exception as e:
            print(f"    [WARN] Binary eval failed: {e}")

        results.append(ModelResult(
            model_key, golden, response_text, verdict, assertions, binary_verdict
        ))

    # Reset singletons
    reset_agent_llms()

    return results


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def print_comparison_table(all_results: Dict[str, List[ModelResult]]) -> None:
    """Print a formatted comparison table to stdout."""
    models = list(all_results.keys())
    buckets = [g.bucket for g in GOLDEN_RESPONSES]

    # Header
    header = f"{'Bucket':<15}"
    for model in models:
        header += f" | {model:<20}"
    print("\n" + "=" * len(header))
    print("WEATHERWISE.AI MODEL COMPARISON")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    # Per-bucket scores
    model_totals: Dict[str, List[float]] = {m: [] for m in models}
    for i, bucket in enumerate(buckets):
        row = f"{bucket:<15}"
        for model in models:
            result = all_results[model][i]
            score = result.verdict.overall_score
            pf = result.verdict.pass_fail.upper()
            row += f" | {score:>5.2f} ({pf:<4})"
            model_totals[model].append(score)
            row += " " * max(0, 14 - len(f"{score:>5.2f} ({pf:<4})"))
        print(row)

    # Averages
    print("-" * len(header))
    avg_row = f"{'AVERAGE':<15}"
    for model in models:
        avg = sum(model_totals[model]) / len(model_totals[model]) if model_totals[model] else 0
        avg_row += f" | {avg:>5.2f}{'':>15}"
    print(avg_row)
    print("=" * len(header))

    # Dimension breakdown per model
    for model in models:
        print(f"\n--- {model} Dimension Breakdown ---")
        for result in all_results[model]:
            print(f"  {result.golden.bucket}:")
            for ds in result.verdict.dimension_scores:
                print(f"    {ds.dimension:<22} {ds.score}/5  {ds.reasoning}")

    # Code assertions
    for model in models:
        print(f"\n--- {model} Code Assertions ---")
        for result in all_results[model]:
            print(f"  {result.golden.bucket}:")
            for a in result.assertions:
                status = "PASS" if a.passed else "FAIL"
                print(f"    [{status}] {a.name}: {a.detail}")

    # Binary criteria
    for model in models:
        print(f"\n--- {model} Binary Criteria ---")
        for result in all_results[model]:
            bv = result.binary_verdict
            if bv is None:
                print(f"  {result.golden.bucket}: [SKIPPED]")
                continue
            status = "PASS" if bv.overall_pass else "FAIL"
            print(
                f"  {result.golden.bucket}: [{status}] "
                f"Must: {bv.must_pass_count} pass / {bv.must_fail_count} fail | "
                f"Should: {bv.should_pass_count}/{bv.should_total_count}"
            )
            for cr in bv.criteria_results:
                tag = "MUST" if cr.is_must else "SHOULD"
                mark = "PASS" if cr.passed else "FAIL"
                print(f"    [{tag}][{mark}] {cr.criterion[:80]}")
                print(f"             {cr.reasoning}")


# ---------------------------------------------------------------------------
# Pairwise comparison
# ---------------------------------------------------------------------------


def run_pairwise_comparison(
    all_results: Dict[str, List[ModelResult]],
    judge_model_key: str = "gpt-4o",
) -> List[Dict]:
    """Run pairwise head-to-head comparison between the first two models.

    Args:
        all_results: Mapping of model_key -> list of ModelResult.
        judge_model_key: Model key for the judge LLM.

    Returns:
        List of dicts with pairwise verdicts per golden query.
    """
    models = list(all_results.keys())
    if len(models) < 2:
        print("Pairwise comparison requires at least 2 models.")
        return []

    model_a, model_b = models[0], models[1]
    pairwise_results = []

    for i, golden in enumerate(GOLDEN_RESPONSES):
        result_a = all_results[model_a][i]
        result_b = all_results[model_b][i]

        print(f"  Pairwise [{model_a} vs {model_b}]: {golden.query[:50]}...")
        try:
            verdict = pairwise_evaluate_debiased(
                query=golden.query,
                response_a=result_a.response,
                response_b=result_b.response,
                reference=golden,
                judge_model_key=judge_model_key,
            )
        except Exception as e:
            verdict = PairwiseVerdict(
                winner="tie",
                reasoning=f"Pairwise judge failed: {e}",
                confidence=0.0,
            )
        pairwise_results.append({
            "bucket": golden.bucket,
            "model_a": model_a,
            "model_b": model_b,
            "verdict": verdict,
        })

    return pairwise_results


def print_pairwise_table(pairwise_results: List[Dict]) -> None:
    """Print pairwise comparison results."""
    if not pairwise_results:
        return

    model_a = pairwise_results[0]["model_a"]
    model_b = pairwise_results[0]["model_b"]

    print(f"\n{'=' * 70}")
    print(f"PAIRWISE COMPARISON: {model_a} vs {model_b}")
    print(f"{'=' * 70}")
    print(f"{'Bucket':<15} | {'Winner':<15} | {'Conf':<6} | Reasoning")
    print(f"{'-' * 70}")

    wins: Dict[str, int] = {model_a: 0, model_b: 0, "tie": 0}
    for pr in pairwise_results:
        v = pr["verdict"]
        if v.winner == "model_a":
            winner_name = model_a
            wins[model_a] += 1
        elif v.winner == "model_b":
            winner_name = model_b
            wins[model_b] += 1
        else:
            winner_name = "TIE"
            wins["tie"] += 1
        print(f"{pr['bucket']:<15} | {winner_name:<15} | {v.confidence:<6.1f} | {v.reasoning[:50]}...")

    print(f"{'-' * 70}")
    print(f"Score: {model_a}={wins[model_a]} | {model_b}={wins[model_b]} | Ties={wins['tie']}")
    print(f"{'=' * 70}")


# ---------------------------------------------------------------------------
# LangSmith integration
# ---------------------------------------------------------------------------


def upload_to_langsmith(all_results: Dict[str, List[ModelResult]]) -> None:
    """Upload comparison results to LangSmith as experiments."""
    try:
        from langsmith import Client
        from langsmith.evaluation import evaluate
    except ImportError:
        print("langsmith not installed. Skipping upload.")
        return

    if not os.environ.get("LANGCHAIN_API_KEY"):
        print("LANGCHAIN_API_KEY not set. Skipping LangSmith upload.")
        return

    client = Client()
    dataset_name = "weatherwise-quality-golden"

    # Create or get dataset
    try:
        dataset = client.read_dataset(dataset_name=dataset_name)
    except Exception:
        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description="Golden quality evaluation cases for WeatherWise.ai",
        )
        for golden in GOLDEN_RESPONSES:
            client.create_example(
                inputs={"query": golden.query, "bucket": golden.bucket},
                outputs={
                    "reference_response": golden.reference_response,
                    "success_criteria": golden.success_criteria,
                },
                dataset_id=dataset.id,
            )
        print(f"Created LangSmith dataset: {dataset_name}")

    for model_key, results in all_results.items():
        prefix = f"weatherwise-{model_key}-{datetime.now().strftime('%Y%m%d')}"
        print(f"Uploading experiment: {prefix}")

        def make_predictor(model_results):
            result_map = {r.golden.query: r.response for r in model_results}
            def predict(inputs):
                return {"response": result_map.get(inputs["query"], "NOT_FOUND")}
            return predict

        def overall_quality(run, example):
            response = run.outputs.get("response", "")
            golden = next((g for g in GOLDEN_RESPONSES if g.query == example.inputs["query"]), None)
            if not golden:
                return {"key": "overall_quality", "score": 0.0}
            verdict = evaluate_response(query=example.inputs["query"], actual_response=response, reference=golden)
            return {"key": "overall_quality", "score": verdict.overall_score / 5.0}

        evaluate(
            make_predictor(results),
            data=dataset_name,
            evaluators=[overall_quality],
            experiment_prefix=prefix,
        )

    print("LangSmith upload complete.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main():
    parser = argparse.ArgumentParser(description="WeatherWise.ai Model Comparison")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["claude-sonnet"],
        help=f"Model keys to compare. Available: {list_models()}",
    )
    parser.add_argument(
        "--judge-model",
        default="gpt-4o",
        help="Model key for the LLM judge (default: gpt-4o)",
    )
    parser.add_argument(
        "--pairwise",
        action="store_true",
        help="Run pairwise comparison between the first two models",
    )
    parser.add_argument(
        "--langsmith",
        action="store_true",
        help="Upload results to LangSmith",
    )
    parser.add_argument(
        "--html",
        nargs="?",
        const="reports/eval_report.html",
        default=None,
        metavar="PATH",
        help="Generate a standalone HTML report (default: reports/eval_report.html)",
    )
    args = parser.parse_args()

    print(f"Models to compare: {args.models}")
    print(f"Judge model: {args.judge_model}")
    print(f"Golden queries: {len(GOLDEN_RESPONSES)}")

    all_results: Dict[str, List[ModelResult]] = {}
    for model_key in args.models:
        print(f"\nRunning model: {model_key}")
        results = await run_single_model(
            model_key, GOLDEN_RESPONSES, judge_model_key=args.judge_model
        )
        all_results[model_key] = results

    print_comparison_table(all_results)

    pairwise_results: Optional[List[Dict]] = None
    if args.pairwise and len(args.models) >= 2:
        print("\nRunning pairwise comparison...")
        pairwise_results = run_pairwise_comparison(
            all_results, judge_model_key=args.judge_model
        )
        print_pairwise_table(pairwise_results)

    if args.html:
        generate_html_report(
            all_results,
            output_path=args.html,
            pairwise_results=pairwise_results,
        )

    if args.langsmith:
        upload_to_langsmith(all_results)


if __name__ == "__main__":
    asyncio.run(main())
