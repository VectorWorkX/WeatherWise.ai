"""Scoring dimensions and rubric definitions for the LLM-as-Judge evaluator.

Each ScoringDimension defines a quality axis with a 1-5 rubric prompt that
the judge LLM uses to score responses.
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

from pydantic import BaseModel, Field


class ScoringDimension(BaseModel):
    """Definition of a single quality dimension for the LLM judge."""

    name: str = Field(description="Machine-readable dimension name")
    description: str = Field(description="Human-readable description")
    rubric: str = Field(description="Full rubric text given to the judge (1-5 scale)")
    score_range: Tuple[int, int] = Field(default=(1, 5))
    weight: float = Field(default=1.0, description="Default weight, overridable per bucket")


SCORING_DIMENSIONS: List[ScoringDimension] = [
    ScoringDimension(
        name="factual_grounding",
        description="Does the response match tool/API data with no hallucination?",
        rubric=(
            "Score 1-5 for FACTUAL GROUNDING:\n"
            "5: All weather values clearly come from tool calls, no fabricated data\n"
            "4: Values match tool output with minor formatting differences\n"
            "3: Most values correct, one minor unsupported claim\n"
            "2: Multiple incorrect or fabricated values\n"
            "1: Largely hallucinated, no evidence of tool data usage"
        ),
        weight=1.5,
    ),
    ScoringDimension(
        name="completeness",
        description="Did the response answer all parts of the query?",
        rubric=(
            "Score 1-5 for COMPLETENESS:\n"
            "5: Every part of the query addressed, all success criteria met\n"
            "4: All major parts addressed, one minor omission\n"
            "3: Most parts addressed, one significant gap\n"
            "2: Only partially answers the query, major aspects missing\n"
            "1: Fails to address the core question"
        ),
        weight=1.0,
    ),
    ScoringDimension(
        name="scientific_accuracy",
        description="Correct use of meteorological terminology and concepts?",
        rubric=(
            "Score 1-5 for SCIENTIFIC ACCURACY:\n"
            "5: All scientific terms used correctly, concepts accurately explained\n"
            "4: Terminology mostly correct, one minor imprecision\n"
            "3: Some terms used correctly, some misapplied or oversimplified\n"
            "2: Significant terminology errors or misconceptions\n"
            "1: Scientific content is largely incorrect or absent"
        ),
        weight=1.0,
    ),
    ScoringDimension(
        name="citation_quality",
        description="Are RAG sources properly cited and traceable?",
        rubric=(
            "Score 1-5 for CITATION QUALITY:\n"
            "5: All claims cite specific sources with page/section references\n"
            "4: Most claims cited, sources are identifiable\n"
            "3: Some citations present but incomplete or vague\n"
            "2: Minimal citations, most claims unsourced\n"
            "1: No citations despite using retrieved knowledge"
        ),
        weight=1.0,
    ),
    ScoringDimension(
        name="formatting",
        description="Proper markdown, units, structure for the query type?",
        rubric=(
            "Score 1-5 for FORMATTING:\n"
            "5: Excellent use of markdown, units on all values, clear structure\n"
            "4: Good formatting with minor inconsistencies\n"
            "3: Adequate but could be better organized\n"
            "2: Poor formatting, missing units, hard to parse\n"
            "1: Unformatted wall of text, no structure"
        ),
        weight=0.5,
    ),
    ScoringDimension(
        name="actionability",
        description="For advisory queries: are recommendations specific and useful?",
        rubric=(
            "Score 1-5 for ACTIONABILITY:\n"
            "5: Specific, implementable recommendations with clear justification\n"
            "4: Good recommendations with some justification\n"
            "3: Generic advice that could apply to any situation\n"
            "2: Vague suggestions without context\n"
            "1: No actionable guidance provided"
        ),
        weight=1.0,
    ),
    # RAG-specific metrics (explanation bucket only)
    ScoringDimension(
        name="faithfulness",
        description="Are all claims in the response supported by the retrieved RAG context?",
        rubric=(
            "Score 1-5 for FAITHFULNESS:\n"
            "5: Every factual claim is directly traceable to a cited source or retrieved document\n"
            "4: Nearly all claims supported, one minor inference beyond sources\n"
            "3: Most claims supported, some unsourced assertions or extrapolations\n"
            "2: Multiple claims lack source support, possible confabulation\n"
            "1: Response contains significant unsupported claims or contradicts retrieved sources"
        ),
        weight=1.5,
    ),
    ScoringDimension(
        name="context_precision",
        description="Are the retrieved documents relevant to the query?",
        rubric=(
            "Score 1-5 for CONTEXT PRECISION:\n"
            "5: All cited sources are directly relevant to the query topic; no noise\n"
            "4: Most sources relevant, one tangentially related\n"
            "3: Mix of relevant and irrelevant sources; some padding\n"
            "2: Mostly irrelevant sources with a few useful hits\n"
            "1: Retrieved context is almost entirely off-topic"
        ),
        weight=1.0,
    ),
    ScoringDimension(
        name="context_recall",
        description="Did retrieval find all information needed to answer the query?",
        rubric=(
            "Score 1-5 for CONTEXT RECALL:\n"
            "5: The retrieved context provides all information needed for a complete answer\n"
            "4: Most information present, one minor aspect requires external knowledge\n"
            "3: Adequate coverage but significant gaps requiring the LLM's own knowledge\n"
            "2: Major gaps in retrieval; the LLM had to rely heavily on parametric knowledge\n"
            "1: Retrieved context provides almost no useful information for the query"
        ),
        weight=1.0,
    ),
]

# Which dimensions apply to which buckets
BUCKET_APPLICABILITY: Dict[str, Set[str]] = {
    "real_time": {"factual_grounding", "completeness", "formatting"},
    "comparative": {"factual_grounding", "completeness", "formatting"},
    "lifestyle": {"factual_grounding", "completeness", "formatting", "actionability"},
    "safety": {"factual_grounding", "completeness", "formatting", "actionability"},
    "explanation": {
        "factual_grounding", "completeness", "formatting",
        "scientific_accuracy", "citation_quality",
        "faithfulness", "context_precision", "context_recall",
    },
    "math": {"factual_grounding", "completeness", "formatting"},
}


def get_dimensions_for_bucket(bucket: str) -> List[ScoringDimension]:
    """Return only the scoring dimensions applicable to a given bucket."""
    applicable = BUCKET_APPLICABILITY.get(bucket, {"factual_grounding", "completeness", "formatting"})
    return [d for d in SCORING_DIMENSIONS if d.name in applicable]


def get_dimension(name: str) -> ScoringDimension:
    """Look up a single dimension by name."""
    for d in SCORING_DIMENSIONS:
        if d.name == name:
            return d
    raise ValueError(f"Unknown dimension: {name}")
