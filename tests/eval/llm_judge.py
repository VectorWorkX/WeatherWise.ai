"""LLM-as-Judge evaluation engine for WeatherWise.ai response quality.

Uses an LLM judge (default: GPT-4o to avoid self-enhancement bias) with
structured output to score responses on multiple quality dimensions against
reference answers from scope.md.
"""

from __future__ import annotations

import re
import os
import sys
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from golden_responses import GoldenResponse  # noqa: E402
from rubrics import ScoringDimension, get_dimensions_for_bucket  # noqa: E402


# ---------------------------------------------------------------------------
# Structured output models for the judge
# ---------------------------------------------------------------------------


class DimensionScore(BaseModel):
    """Score for a single quality dimension."""

    dimension: str = Field(description="Name of the dimension scored")
    score: int = Field(ge=1, le=5, description="Score from 1 (worst) to 5 (best)")
    reasoning: str = Field(description="1-2 sentence justification for the score")


class JudgeVerdict(BaseModel):
    """Complete evaluation result from the LLM judge."""

    dimension_scores: List[DimensionScore] = Field(description="Scores for each dimension")
    overall_score: float = Field(description="Weighted average across dimensions (1.0-5.0)")
    overall_assessment: str = Field(description="2-3 sentence summary of the evaluation")
    pass_fail: Literal["pass", "fail"] = Field(
        description="'pass' if overall_score >= 3.5, 'fail' otherwise"
    )


# ---------------------------------------------------------------------------
# Judge prompt template
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """\
You are an expert quality evaluator for WeatherWise.ai, a professional weather \
intelligence system. You evaluate AI-generated weather responses against \
reference answers and success criteria.

## Important Rules
- Score based on SUBSTANCE, not exact wording. The actual response need not \
match the reference verbatim.
- A response can score 5/5 even if phrased differently, as long as it meets \
the success criteria.
- For "factual_grounding": weather values WILL differ from the reference \
(live data changes over time). Judge whether values appear to come from tool \
calls (not hallucinated) and whether the response cites specific numbers with \
units.
- Be strict on hallucination: any fabricated data point that clearly was not \
fetched from a tool is an automatic 1 for factual_grounding.
- Score each dimension independently using the provided rubrics.
"""

JUDGE_USER_TEMPLATE = """\
## User Query
{query}

## Reference Response (from scope.md — values are illustrative, not exact)
{reference_response}

## Success Criteria
{success_criteria}

## Actual Response to Evaluate
{actual_response}

## Scoring Rubrics
{rubric_text}

Score each applicable dimension. For overall_score, compute the weighted \
average. Set pass_fail to "pass" if overall_score >= 3.5, else "fail".
"""


# ---------------------------------------------------------------------------
# Judge LLM singleton
# ---------------------------------------------------------------------------

_judge_llm = None
_judge_model_key: Optional[str] = None


def _get_judge_llm(model_key: str = "gpt-4o"):
    """Return a cached LLM instance for judging.

    Uses model_registry to resolve the model_key. The singleton is
    invalidated if model_key changes between calls.

    Args:
        model_key: A registered model key (default: 'gpt-4o').
    """
    global _judge_llm, _judge_model_key
    if _judge_llm is None or _judge_model_key != model_key:
        from utils.model_registry import get_model

        _judge_llm = get_model(model_key, temperature=0, max_tokens=1024)
        _judge_model_key = model_key
    return _judge_llm


def reset_judge_llm() -> None:
    """Reset the judge singleton so the next call re-creates it."""
    global _judge_llm, _judge_model_key
    _judge_llm = None
    _judge_model_key = None


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------


def evaluate_response(
    query: str,
    actual_response: str,
    reference: GoldenResponse,
    dimensions: Optional[List[ScoringDimension]] = None,
    judge_model_key: str = "gpt-4o",
) -> JudgeVerdict:
    """Evaluate a response using the LLM judge.

    Args:
        query: The original user query.
        actual_response: The system's actual response to evaluate.
        reference: The golden reference response with success criteria.
        dimensions: Optional override for scoring dimensions.
                    Defaults to the dimensions applicable to the bucket.
        judge_model_key: Model key for the judge LLM (default: 'gpt-4o').

    Returns:
        A JudgeVerdict with per-dimension scores and overall assessment.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    if dimensions is None:
        dimensions = get_dimensions_for_bucket(reference.bucket)

    # Build rubric text
    rubric_parts = []
    for dim in dimensions:
        rubric_parts.append(f"### {dim.name}\n{dim.rubric}")
    rubric_text = "\n\n".join(rubric_parts)

    # Build success criteria text
    criteria_text = "\n".join(f"- {c}" for c in reference.success_criteria)

    # Build the judge prompt
    user_content = JUDGE_USER_TEMPLATE.format(
        query=query,
        reference_response=reference.reference_response,
        success_criteria=criteria_text,
        actual_response=actual_response,
        rubric_text=rubric_text,
    )

    judge = _get_judge_llm(judge_model_key).with_structured_output(JudgeVerdict)
    verdict = judge.invoke([
        SystemMessage(content=JUDGE_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ])

    # Recompute weighted average using bucket-specific weights
    total_weight = 0.0
    weighted_sum = 0.0
    for ds in verdict.dimension_scores:
        weight = reference.scoring_weights.get(ds.dimension, 1.0)
        weighted_sum += ds.score * weight
        total_weight += weight

    if total_weight > 0:
        verdict.overall_score = round(weighted_sum / total_weight, 2)
        verdict.pass_fail = "pass" if verdict.overall_score >= 3.5 else "fail"

    return verdict


# ---------------------------------------------------------------------------
# Binary PASS/FAIL per success criterion
# ---------------------------------------------------------------------------


class CriterionResult(BaseModel):
    """Result of evaluating a single success criterion."""

    criterion: str = Field(description="The criterion text being evaluated")
    is_must: bool = Field(description="True if this is a hard 'Must' requirement")
    passed: bool = Field(description="Whether this criterion was met")
    reasoning: str = Field(description="1-sentence justification for the PASS/FAIL decision")


class BinaryVerdict(BaseModel):
    """Aggregated binary evaluation across all success criteria."""

    criteria_results: List[CriterionResult] = Field(
        description="Individual results for each criterion"
    )
    must_pass_count: int = Field(description="Number of Must criteria that passed")
    must_fail_count: int = Field(description="Number of Must criteria that failed")
    should_pass_count: int = Field(description="Number of Should criteria that passed")
    should_total_count: int = Field(description="Total number of Should criteria")
    overall_pass: bool = Field(
        description="True if all Must criteria pass AND >= 75%% of Should criteria pass"
    )


class _CriteriaOutput(BaseModel):
    """Wrapper for structured output of criteria evaluation."""

    results: List[CriterionResult]


def _classify_criterion(criterion: str) -> bool:
    """Determine if a criterion is a 'Must' (hard) requirement.

    Rules:
    - Starts with 'Must' -> is_must = True
    - Starts with 'Should' -> is_must = False
    - Contains 'must' (case-insensitive) anywhere -> is_must = True
    - Otherwise -> is_must = False (treated as Should / soft)
    """
    stripped = criterion.strip()
    if stripped.startswith("Must"):
        return True
    if stripped.startswith("Should"):
        return False
    if re.search(r"\bmust\b", stripped, re.IGNORECASE):
        return True
    return False


CRITERIA_JUDGE_SYSTEM_PROMPT = """\
You are an expert evaluator for WeatherWise.ai. For each success criterion, \
determine if the actual response satisfies it.

## Rules
- Evaluate each criterion independently.
- For each criterion, output PASS or FAIL with a 1-sentence justification.
- For criteria about fetching live data: since you cannot verify actual API \
calls were made, check whether the response contains specific numeric data \
with units that appears to come from a data source (not generic/vague).
- Be strict on [MUST] criteria; be slightly more lenient on [SHOULD] criteria.
"""

CRITERIA_USER_TEMPLATE = """\
## User Query
{query}

## Reference Response (illustrative — exact values will differ)
{reference_response}

## Actual Response to Evaluate
{actual_response}

## Success Criteria to Evaluate
{criteria_list}

For each criterion above, determine if the actual response satisfies it. \
Output a result for each criterion with passed=true or passed=false and \
a 1-sentence reasoning.
"""


def evaluate_criteria(
    query: str,
    actual_response: str,
    reference: GoldenResponse,
    judge_model_key: str = "gpt-4o",
) -> BinaryVerdict:
    """Evaluate each success criterion as binary PASS/FAIL.

    Args:
        query: The original user query.
        actual_response: The system's actual response.
        reference: The golden reference with success_criteria list.
        judge_model_key: Model key for the judge LLM.

    Returns:
        A BinaryVerdict with per-criterion results and overall pass/fail.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    classified = [
        (c, _classify_criterion(c)) for c in reference.success_criteria
    ]
    criteria_list = "\n".join(
        f"{'[MUST]' if is_must else '[SHOULD]'} {c}"
        for c, is_must in classified
    )

    user_content = CRITERIA_USER_TEMPLATE.format(
        query=query,
        reference_response=reference.reference_response,
        actual_response=actual_response,
        criteria_list=criteria_list,
    )

    judge = _get_judge_llm(judge_model_key).with_structured_output(_CriteriaOutput)
    output = judge.invoke([
        SystemMessage(content=CRITERIA_JUDGE_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ])

    # Ensure is_must flags match our classification (don't trust the LLM's)
    for i, cr in enumerate(output.results):
        if i < len(classified):
            cr.is_must = classified[i][1]

    must_results = [r for r in output.results if r.is_must]
    should_results = [r for r in output.results if not r.is_must]
    must_pass = sum(1 for r in must_results if r.passed)
    must_fail = len(must_results) - must_pass
    should_pass = sum(1 for r in should_results if r.passed)
    should_total = len(should_results)

    should_threshold_met = (
        should_pass >= (0.75 * should_total) if should_total > 0 else True
    )
    overall_pass = (must_fail == 0) and should_threshold_met

    return BinaryVerdict(
        criteria_results=output.results,
        must_pass_count=must_pass,
        must_fail_count=must_fail,
        should_pass_count=should_pass,
        should_total_count=should_total,
        overall_pass=overall_pass,
    )


# ---------------------------------------------------------------------------
# Pairwise comparison
# ---------------------------------------------------------------------------


class PairwiseVerdict(BaseModel):
    """Result of comparing two model responses head-to-head."""

    winner: Literal["model_a", "model_b", "tie"] = Field(
        description="Which model produced the better response"
    )
    reasoning: str = Field(
        description="2-3 sentence justification for the winner decision"
    )
    per_dimension_winner: Dict[str, Literal["model_a", "model_b", "tie"]] = Field(
        default_factory=dict,
        description="Winner for each scoring dimension",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence in the verdict (reduced if swap-debiasing disagrees)",
    )


PAIRWISE_JUDGE_SYSTEM_PROMPT = """\
You are an expert quality evaluator for WeatherWise.ai. You are comparing two \
AI-generated weather responses to the same query. Determine which response is \
better overall and for each scoring dimension.

## Rules
- Focus on SUBSTANCE: accuracy, completeness, and usefulness.
- A response with correct data always beats one with plausible-sounding but \
fabricated data.
- If both responses are roughly equal in quality, output "tie".
- Score each dimension independently.
- Do NOT let response length bias your judgment — a concise correct answer \
beats a verbose incorrect one.
"""

PAIRWISE_USER_TEMPLATE = """\
## User Query
{query}

## Reference Response (illustrative)
{reference_response}

## Success Criteria
{success_criteria}

## Response A
{response_a}

## Response B
{response_b}

## Dimensions to Compare
{dimension_names}

For each dimension, state whether Response A or Response B is better (or tie). \
Then give an overall winner with reasoning.
"""


def pairwise_evaluate(
    query: str,
    response_a: str,
    response_b: str,
    reference: GoldenResponse,
    dimensions: Optional[List[ScoringDimension]] = None,
    judge_model_key: str = "gpt-4o",
) -> PairwiseVerdict:
    """Compare two responses head-to-head using the LLM judge.

    Args:
        query: The original user query.
        response_a: First model's response.
        response_b: Second model's response.
        reference: Golden reference with success criteria.
        dimensions: Scoring dimensions to compare on.
        judge_model_key: Model key for the judge LLM.

    Returns:
        PairwiseVerdict indicating the winner.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    if dimensions is None:
        dimensions = get_dimensions_for_bucket(reference.bucket)

    criteria_text = "\n".join(f"- {c}" for c in reference.success_criteria)
    dimension_names = ", ".join(d.name for d in dimensions)

    user_content = PAIRWISE_USER_TEMPLATE.format(
        query=query,
        reference_response=reference.reference_response,
        success_criteria=criteria_text,
        response_a=response_a,
        response_b=response_b,
        dimension_names=dimension_names,
    )

    judge = _get_judge_llm(judge_model_key).with_structured_output(
        PairwiseVerdict, method="function_calling"
    )
    verdict = judge.invoke([
        SystemMessage(content=PAIRWISE_JUDGE_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ])
    return verdict


_REVERSE_MAP = {"model_a": "model_b", "model_b": "model_a", "tie": "tie"}


def pairwise_evaluate_debiased(
    query: str,
    response_a: str,
    response_b: str,
    reference: GoldenResponse,
    dimensions: Optional[List[ScoringDimension]] = None,
    judge_model_key: str = "gpt-4o",
) -> PairwiseVerdict:
    """Pairwise comparison with swap-debiasing.

    Evaluates both (A, B) and (B, A) orderings. If results agree,
    confidence = 1.0. If they disagree, returns 'tie' with confidence = 0.5
    and a note about the inconsistency.
    """
    forward = pairwise_evaluate(
        query, response_a, response_b, reference, dimensions, judge_model_key
    )

    reverse = pairwise_evaluate(
        query, response_b, response_a, reference, dimensions, judge_model_key
    )

    reverse_mapped = _REVERSE_MAP[reverse.winner]

    if forward.winner == reverse_mapped:
        forward.confidence = 1.0
        return forward
    else:
        return PairwiseVerdict(
            winner="tie",
            reasoning=(
                f"POSITION BIAS DETECTED: Forward pass chose {forward.winner}, "
                f"reverse pass chose {reverse.winner} "
                f"(mapped: {reverse_mapped}). Defaulting to tie. "
                f"Forward: {forward.reasoning} | Reverse: {reverse.reasoning}"
            ),
            per_dimension_winner={},
            confidence=0.5,
        )
