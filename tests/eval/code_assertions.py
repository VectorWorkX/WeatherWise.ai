"""Deterministic code-assertion layer for WeatherWise.ai response evaluation.

These checks run BEFORE the LLM judge, catching obvious failures (missing
units, wrong locations, no citations) without burning LLM tokens. Fast, cheap,
and reproducible.
"""

from __future__ import annotations

import re
from typing import List

from pydantic import BaseModel, Field

from golden_responses import GOLDEN_RESPONSES, GoldenResponse


class AssertionResult(BaseModel):
    """Result of a single deterministic assertion."""

    name: str = Field(description="Machine-readable assertion name")
    passed: bool = Field(description="Whether the assertion passed")
    detail: str = Field(description="Human-readable explanation of the result")


# ---------------------------------------------------------------------------
# Individual assertion functions
# ---------------------------------------------------------------------------

UNIT_PATTERN = re.compile(
    r"\d+\.?\d*\s*(?:°[FC]|mph|km/h|knots?|inches?|in\.|cm|mm|mb|hPa|%|kelvin|K\b)",
    re.IGNORECASE,
)

# Derive known locations from golden response metadata so the list stays in
# sync automatically.  When adding a new golden case, just populate its
# `locations` field in golden_responses.py — no changes needed here.
KNOWN_LOCATIONS = sorted(
    {loc for golden in GOLDEN_RESPONSES for loc in golden.locations}
)

# Short names that need word-boundary matching to avoid false positives
# (e.g., "LA" matching "calcuLAte", "NYC" matching random substrings)
_SHORT_LOCATION_ALIASES = {
    "NYC": "New York",
    "LA": "Los Angeles",
}

STEP_PATTERNS = [
    r"[=→]\s*-?\d+",
    r"convert(?:ed|ing)?",
    r"\(-?\d+\.?\d*\s*[-−]\s*\(?-?\d+",
    r"°[FC]\s*→?\s*-?\d+.*°[FC]",
]

CITATION_PATTERN = re.compile(
    r"\[Source\s*#?\d*:?\s*[^\]]+\]|"
    r"\[Based on\s+[^\]]+\]|"
    r"Source:\s*\w+|"
    r"p\.\s*\d+|"
    r"CPO-\d+|"
    r"NCA\d|"
    r"NOAA|NASA|AMS",
    re.IGNORECASE,
)


def check_numeric_with_units(response: str, bucket: str = "") -> AssertionResult:
    """Check that response contains numeric values with weather units.

    Skipped for explanation bucket (science-focused, may not have raw data).
    """
    if bucket == "explanation":
        return AssertionResult(
            name="numeric_with_units",
            passed=True,
            detail="Explanation bucket — numeric data not required (skipped)",
        )
    matches = UNIT_PATTERN.findall(response)
    passed = len(matches) >= 1
    return AssertionResult(
        name="numeric_with_units",
        passed=passed,
        detail=(
            f"Found {len(matches)} numeric value(s) with units"
            if passed
            else "No numeric values with weather units found"
        ),
    )


def _location_in_text(location: str, text: str) -> bool:
    """Check if a location name appears in text with word-boundary awareness."""
    return bool(re.search(r"\b" + re.escape(location) + r"\b", text, re.IGNORECASE))


def check_expected_locations(
    response: str, query: str, bucket: str = ""
) -> AssertionResult:
    """Check that response mentions locations extracted from the query.

    Skipped for lifestyle bucket (agent dynamically selects candidate locations).
    """
    if bucket == "lifestyle":
        return AssertionResult(
            name="expected_locations",
            passed=True,
            detail="Lifestyle bucket — agent selects its own locations (skipped)",
        )
    # Find full-name locations in query
    query_locations = [
        loc for loc in KNOWN_LOCATIONS if _location_in_text(loc, query)
    ]
    # Also resolve short aliases (NYC -> New York, LA -> Los Angeles)
    for alias, full_name in _SHORT_LOCATION_ALIASES.items():
        if re.search(r"\b" + re.escape(alias) + r"\b", query):
            query_locations.append(full_name)

    if not query_locations:
        return AssertionResult(
            name="expected_locations",
            passed=True,
            detail="No specific locations detected in query (skipped)",
        )
    found = [loc for loc in query_locations if _location_in_text(loc, response)]
    missing = set(query_locations) - set(found)
    passed = len(found) == len(query_locations)
    return AssertionResult(
        name="expected_locations",
        passed=passed,
        detail=(
            f"Found {len(found)}/{len(query_locations)} expected locations"
            if passed
            else f"Missing locations: {missing}"
        ),
    )


def check_math_intermediate_steps(response: str, bucket: str) -> AssertionResult:
    """For math bucket: check that intermediate calculation steps are shown."""
    if bucket != "math":
        return AssertionResult(
            name="math_intermediate_steps",
            passed=True,
            detail="Not a math bucket query (skipped)",
        )
    found_steps = sum(
        1 for p in STEP_PATTERNS if re.search(p, response, re.IGNORECASE)
    )
    passed = found_steps >= 2
    return AssertionResult(
        name="math_intermediate_steps",
        passed=passed,
        detail=(
            f"Found {found_steps} intermediate step patterns"
            if passed
            else f"Found {found_steps}/4 step indicator patterns (need >=2)"
        ),
    )


def check_citations(response: str, bucket: str) -> AssertionResult:
    """For explanation bucket: check that citations/sources are present."""
    if bucket != "explanation":
        return AssertionResult(
            name="has_citations",
            passed=True,
            detail="Not an explanation bucket query (skipped)",
        )
    matches = CITATION_PATTERN.findall(response)
    passed = len(matches) >= 1
    return AssertionResult(
        name="has_citations",
        passed=passed,
        detail=(
            f"Found {len(matches)} citation(s)"
            if passed
            else "No citations or source references found in explanation response"
        ),
    )


def check_multiple_locations_comparative(
    response: str, bucket: str
) -> AssertionResult:
    """For comparative bucket: check that multiple distinct locations are mentioned."""
    if bucket != "comparative":
        return AssertionResult(
            name="multiple_locations_comparative",
            passed=True,
            detail="Not a comparative bucket query (skipped)",
        )
    location_candidates = re.findall(
        r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", response
    )
    unique = set(location_candidates)
    non_locations = {
        "The", "This", "That", "However", "Additionally", "Current",
        "Based", "Overall", "Winter", "Storm", "Response", "Summary",
    }
    unique = unique - non_locations
    passed = len(unique) >= 2
    return AssertionResult(
        name="multiple_locations_comparative",
        passed=passed,
        detail=f"Found {len(unique)} potential location references (need >=2)",
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_assertions(
    response: str, golden: GoldenResponse
) -> List[AssertionResult]:
    """Run all applicable deterministic assertions for a response.

    Args:
        response: The actual system response text.
        golden: The golden reference with bucket and query info.

    Returns:
        List of AssertionResult for each check.
    """
    return [
        check_numeric_with_units(response, golden.bucket),
        check_expected_locations(response, golden.query, golden.bucket),
        check_math_intermediate_steps(response, golden.bucket),
        check_citations(response, golden.bucket),
        check_multiple_locations_comparative(response, golden.bucket),
    ]
