"""Evaluation framework for WeatherWise.ai routing and response quality.

Golden test cases for each intent bucket, plus evaluators for:
- Routing accuracy: does the supervisor classify the intent correctly?
- Response quality: LLM-as-Judge for output quality scoring.
- Code assertions: deterministic checks for response structure.
- Binary criteria: PASS/FAIL per success criterion.
- Pairwise comparison: head-to-head model comparison.
- Model comparison: compare Claude vs Gemini on the same queries.

Run with::

    python -m pytest tests/eval/test_evaluation.py -v

    # Unit tests only (no API keys):
    python -m pytest tests/eval/test_evaluation.py -v -k "not Accuracy and not Judge and not Binary and not Pairwise"

    # Integration tests (requires OPENAI_API_KEY):
    python -m pytest tests/eval/test_evaluation.py::TestJudgeModelSwitch -v
    python -m pytest tests/eval/test_evaluation.py::TestBinaryVerdict -v
    python -m pytest tests/eval/test_evaluation.py::TestPairwiseEvaluation -v

    # LangSmith evaluation (requires LANGCHAIN_API_KEY):
    python tests/eval/test_evaluation.py
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from graph.nodes.supervisor import AGENT_MAP, supervisor_node  # noqa: E402
from graph.state import RouteDecision  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402

from code_assertions import (  # noqa: E402
    check_citations,
    check_numeric_with_units,
    run_assertions,
)
from conftest import requires_anthropic, requires_openai  # noqa: E402
from golden_responses import GOLDEN_RESPONSES  # noqa: E402
from llm_judge import _classify_criterion  # noqa: E402
from rubrics import get_dimensions_for_bucket  # noqa: E402

# ---------------------------------------------------------------------------
# Golden test cases — (query, expected_intent, expected_agent)
# ---------------------------------------------------------------------------

GOLDEN_CASES: list[tuple[str, str, str]] = [
    # Bucket 1: Real-Time
    (
        "What is the wind speed and gust spread in San Francisco right now?",
        "real_time",
        "forecast_agent",
    ),
    (
        "What's the temperature in Tokyo?",
        "real_time",
        "forecast_agent",
    ),
    # Bucket 2: Comparative
    (
        "How does the current winter storm severity in Buffalo compare to Boston's accumulation?",
        "comparative",
        "forecast_agent",
    ),
    (
        "Is it warmer in NYC or LA right now?",
        "comparative",
        "forecast_agent",
    ),
    # Bucket 3: Lifestyle
    (
        "I'm planning a ski trip to Colorado in February. Based on current snowpack and wind forecasts, which resort town should I target?",
        "lifestyle",
        "forecast_agent",
    ),
    # Bucket 4: Safety
    (
        "My logistics fleet is traveling from Dallas to Denver between Feb 7th and 9th. Are there any severe weather advisories or icing risks along this route?",
        "safety",
        "forecast_agent",
    ),
    (
        "Are there any tornado warnings in Oklahoma right now?",
        "safety",
        "forecast_agent",
    ),
    # Bucket 5: Explanation
    (
        "Why did the 2023 Canadian wildfire smoke travel so far south into the US, and what atmospheric blocking patterns caused it?",
        "explanation",
        "research_agent",
    ),
    (
        "Why do heat domes form over the Midwest?",
        "explanation",
        "research_agent",
    ),
    # Bucket 6: Math
    (
        "Calculate the difference between Chicago's current wind chill and Miami's heat index in degrees Celsius.",
        "math",
        "analyst_agent",
    ),
    (
        "Sum the temperatures of Boston and NYC and convert to Kelvin.",
        "math",
        "analyst_agent",
    ),
]

# Off-topic queries — supervisor should classify as "off_topic"
OFF_TOPIC_CASES: list[str] = [
    "What is the GDP of France?",
    "Tell me a joke about cats.",
    "Who won the Super Bowl in 2024?",
    "What is the meaning of life?",
]

# Greeting queries — supervisor should classify as "greeting"
GREETING_CASES: list[str] = [
    "Hello",
    "What can you do?",
]

# Weather queries — supervisor should classify as a weather bucket
ON_TOPIC_CASES: list[str] = [
    "What's the weather in Chicago?",
    "Is it going to rain tomorrow in Seattle?",
    "Why do hurricanes form in the Atlantic?",
    "Compare snowfall in Denver and Salt Lake City.",
    "Can I take my kid out in the park today in San Jose?",
]

# Valid weather intent buckets
_WEATHER_BUCKETS = {"real_time", "comparative", "lifestyle", "safety", "explanation", "math"}


# ---------------------------------------------------------------------------
# Input filtering tests (LLM-based — requires ANTHROPIC_API_KEY)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set — skipping LLM-based input filtering tests",
)
class TestInputFiltering:
    """Test that the supervisor correctly classifies weather, greeting, and off-topic queries."""

    @pytest.mark.parametrize("query", ON_TOPIC_CASES)
    def test_weather_queries_routed(self, query: str):
        state = {"messages": [HumanMessage(content=query)]}
        result = supervisor_node(state)
        intent = result.get("intent", "")
        assert intent in _WEATHER_BUCKETS, (
            f"Weather query classified as '{intent}' instead of a weather bucket: {query}"
        )

    @pytest.mark.parametrize("query", OFF_TOPIC_CASES)
    def test_off_topic_redirected(self, query: str):
        state = {"messages": [HumanMessage(content=query)]}
        result = supervisor_node(state)
        assert result.get("intent") == "off_topic", (
            f"Off-topic query not redirected: {query} → {result.get('intent')}"
        )

    @pytest.mark.parametrize("query", GREETING_CASES)
    def test_greetings_handled(self, query: str):
        state = {"messages": [HumanMessage(content=query)]}
        result = supervisor_node(state)
        assert result.get("intent") == "greeting", (
            f"Greeting not recognized: {query} → {result.get('intent')}"
        )


# ---------------------------------------------------------------------------
# Routing tests
# ---------------------------------------------------------------------------


class TestRouting:
    """Test that AGENT_MAP routes intents to correct agents."""

    @pytest.mark.parametrize(
        "intent,expected_agent",
        [
            ("real_time", "forecast_agent"),
            ("comparative", "forecast_agent"),
            ("lifestyle", "forecast_agent"),
            ("safety", "forecast_agent"),
            ("explanation", "research_agent"),
            ("math", "analyst_agent"),
        ],
    )
    def test_agent_map(self, intent: str, expected_agent: str):
        assert AGENT_MAP[intent] == expected_agent


# ---------------------------------------------------------------------------
# Routing accuracy evaluator (requires ANTHROPIC_API_KEY)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set — skipping LLM-based routing test",
)
class TestRoutingAccuracy:
    """Test routing accuracy against golden cases using the actual supervisor LLM."""

    @pytest.mark.parametrize(
        "query,expected_intent,expected_agent",
        GOLDEN_CASES,
        ids=[c[0][:50] for c in GOLDEN_CASES],
    )
    def test_supervisor_routing(self, query: str, expected_intent: str, expected_agent: str):
        """Verify the supervisor classifies each golden query to the correct intent."""
        state = {"messages": [HumanMessage(content=query)]}
        result = supervisor_node(state)

        actual_intent = result.get("intent", "")
        actual_agent = AGENT_MAP.get(actual_intent, "")

        # Primary check: correct agent (more important than exact intent)
        assert actual_agent == expected_agent, (
            f"Query: {query}\n"
            f"Expected agent: {expected_agent} (intent: {expected_intent})\n"
            f"Got agent: {actual_agent} (intent: {actual_intent})"
        )


# ---------------------------------------------------------------------------
# Code assertions tests (P1)
# ---------------------------------------------------------------------------


class TestCodeAssertions:
    """Test deterministic code assertions against golden reference responses."""

    @pytest.mark.parametrize(
        "golden",
        GOLDEN_RESPONSES,
        ids=[g.bucket for g in GOLDEN_RESPONSES],
    )
    def test_golden_references_pass_assertions(self, golden):
        """The golden reference responses themselves should pass all assertions."""
        results = run_assertions(golden.reference_response, golden)
        for r in results:
            if "skipped" not in r.detail:
                assert r.passed, (
                    f"Assertion '{r.name}' failed on golden {golden.bucket}: {r.detail}"
                )

    def test_numeric_units_catches_missing_units(self):
        result = check_numeric_with_units("The weather is nice and warm today.")
        assert not result.passed

    def test_numeric_units_passes_with_units(self):
        result = check_numeric_with_units("Temperature is 72°F with winds at 15 mph.")
        assert result.passed

    def test_citations_catches_missing_sources(self):
        result = check_citations(
            "Heat domes form due to high pressure systems.", "explanation"
        )
        assert not result.passed

    def test_citations_passes_with_sources(self):
        result = check_citations(
            "[Source #1: NOAA Report, p.5] Heat domes form due to...", "explanation"
        )
        assert result.passed

    def test_citations_skips_non_explanation(self):
        result = check_citations("Temp is 72°F.", "real_time")
        assert result.passed
        assert "skipped" in result.detail


# ---------------------------------------------------------------------------
# Criterion classification tests (P3)
# ---------------------------------------------------------------------------


class TestCriterionClassification:
    """Test the Must/Should parsing logic for binary criteria evaluation."""

    @pytest.mark.parametrize(
        "criterion,expected_is_must",
        [
            ("Must use get_current_weather to fetch live wind data.", True),
            ("Must return both sustained wind speed and gust speed.", True),
            ("Should provide brief contextual interpretation.", False),
            ("Tone should be advisory and helpful.", False),
            ("All operands must be in consistent units (Celsius).", True),
            ("The mathematical operation is determined by the query.", False),
            ("Retrieved source documents must be traceable via doc_id.", True),
            ("Alert types assessed should include: winter storm warnings.", False),
            ("Urgency formatting must scale with severity.", True),
            (
                "When visual data was ingested, the response must include citations.",
                True,
            ),
            (
                "The explanation must address both parts of the question.",
                True,
            ),
        ],
    )
    def test_classify_criterion(self, criterion: str, expected_is_must: bool):
        assert _classify_criterion(criterion) == expected_is_must, (
            f"Criterion: {criterion!r}\n"
            f"Expected is_must={expected_is_must}, "
            f"got {_classify_criterion(criterion)}"
        )


# ---------------------------------------------------------------------------
# RAG metrics rubric tests (P4)
# ---------------------------------------------------------------------------


class TestRubrics:
    """Test rubric configuration for RAG-specific metrics."""

    def test_explanation_bucket_has_rag_dimensions(self):
        dims = get_dimensions_for_bucket("explanation")
        dim_names = {d.name for d in dims}
        assert "faithfulness" in dim_names
        assert "context_precision" in dim_names
        assert "context_recall" in dim_names

    def test_non_explanation_buckets_lack_rag_dimensions(self):
        for bucket in ["real_time", "comparative", "math", "safety", "lifestyle"]:
            dims = get_dimensions_for_bucket(bucket)
            dim_names = {d.name for d in dims}
            assert "faithfulness" not in dim_names, f"{bucket} should not have faithfulness"
            assert "context_precision" not in dim_names, f"{bucket} should not have context_precision"
            assert "context_recall" not in dim_names, f"{bucket} should not have context_recall"

    def test_explanation_has_all_expected_dimensions(self):
        dims = get_dimensions_for_bucket("explanation")
        dim_names = {d.name for d in dims}
        expected = {
            "factual_grounding", "completeness", "formatting",
            "scientific_accuracy", "citation_quality",
            "faithfulness", "context_precision", "context_recall",
        }
        assert dim_names == expected


# ---------------------------------------------------------------------------
# Integration tests (require API keys)
# ---------------------------------------------------------------------------


@requires_openai
class TestJudgeModelSwitch:
    """Test that the judge can use GPT-4o via model registry."""

    def test_gpt4o_judge_returns_verdict(self):
        from llm_judge import evaluate_response, reset_judge_llm

        reset_judge_llm()
        golden = GOLDEN_RESPONSES[0]  # real_time
        verdict = evaluate_response(
            query=golden.query,
            actual_response=golden.reference_response,
            reference=golden,
            judge_model_key="gpt-4o",
        )
        assert verdict.overall_score >= 1.0
        assert verdict.pass_fail in ("pass", "fail")
        assert len(verdict.dimension_scores) > 0
        reset_judge_llm()


@requires_openai
class TestBinaryVerdict:
    """Test binary PASS/FAIL criteria evaluation."""

    def test_golden_reference_passes_own_criteria(self):
        from llm_judge import evaluate_criteria, reset_judge_llm

        reset_judge_llm()
        golden = GOLDEN_RESPONSES[0]  # real_time
        bv = evaluate_criteria(
            query=golden.query,
            actual_response=golden.reference_response,
            reference=golden,
            judge_model_key="gpt-4o",
        )
        assert bv.must_fail_count == 0, (
            f"Golden reference failed Must criteria: "
            f"{[cr for cr in bv.criteria_results if cr.is_must and not cr.passed]}"
        )
        assert bv.overall_pass
        reset_judge_llm()


@requires_openai
class TestPairwiseEvaluation:
    """Test pairwise comparison with swap-debiasing."""

    def test_golden_beats_nonanswer(self):
        from llm_judge import pairwise_evaluate, reset_judge_llm

        reset_judge_llm()
        golden = GOLDEN_RESPONSES[0]  # real_time
        verdict = pairwise_evaluate(
            query=golden.query,
            response_a=golden.reference_response,
            response_b="I don't know the weather.",
            reference=golden,
            judge_model_key="gpt-4o",
        )
        assert verdict.winner == "model_a", (
            f"Golden reference should beat a non-answer. Got: {verdict.winner}"
        )
        reset_judge_llm()


# ---------------------------------------------------------------------------
# LangSmith evaluation runner (run as script, not pytest)
# ---------------------------------------------------------------------------


def run_langsmith_evaluation():
    """Run LangSmith-based evaluation with golden test cases.

    Requires: LANGCHAIN_API_KEY, LANGCHAIN_TRACING_V2=true
    Usage: python tests/eval/test_evaluation.py
    """
    try:
        from langsmith import Client
    except ImportError:
        print("langsmith package not installed. Install with: pip install langsmith")
        return

    if not os.environ.get("LANGCHAIN_API_KEY"):
        print("LANGCHAIN_API_KEY not set. Skipping LangSmith evaluation.")
        return

    client = Client()
    dataset_name = "weatherwise-routing-golden"

    # Create or update dataset
    try:
        dataset = client.read_dataset(dataset_name=dataset_name)
        print(f"Using existing dataset: {dataset_name}")
    except Exception:
        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description="Golden test cases for WeatherWise.ai intent routing",
        )
        print(f"Created dataset: {dataset_name}")

        # Add examples
        for query, expected_intent, expected_agent in GOLDEN_CASES:
            client.create_example(
                inputs={"query": query},
                outputs={
                    "expected_intent": expected_intent,
                    "expected_agent": expected_agent,
                },
                dataset_id=dataset.id,
            )
        print(f"Added {len(GOLDEN_CASES)} examples to dataset.")

    # Define evaluator
    def routing_accuracy(run, example) -> dict:
        """Check if the supervisor routed to the correct agent."""
        actual_intent = run.outputs.get("intent", "") if run.outputs else ""
        expected_agent = example.outputs["expected_agent"]
        actual_agent = AGENT_MAP.get(actual_intent, "")
        return {
            "key": "routing_accuracy",
            "score": 1.0 if actual_agent == expected_agent else 0.0,
        }

    # Define target function
    def predict(inputs: dict) -> dict:
        state = {"messages": [HumanMessage(content=inputs["query"])]}
        return supervisor_node(state)

    # Run evaluation
    from langsmith.evaluation import evaluate

    results = evaluate(
        predict,
        data=dataset_name,
        evaluators=[routing_accuracy],
        experiment_prefix="weatherwise-routing",
    )
    print(f"\nEvaluation complete. Results: {results}")


if __name__ == "__main__":
    run_langsmith_evaluation()
