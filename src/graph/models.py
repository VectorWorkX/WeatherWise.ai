"""Structured output models for specialist agent responses.

These Pydantic models define the expected shape of each agent's final output,
enabling downstream validation and structured logging.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ForecastResponse(BaseModel):
    """Structured output from the forecast agent (Buckets 1-4)."""

    location: str = Field(description="The queried location name.")
    temperature: str = Field(description="Current temperature with units (e.g., '72°F').")
    conditions: str = Field(description="Current weather conditions (e.g., 'Partly Cloudy').")
    wind: str = Field(description="Wind speed and direction (e.g., '15 mph NW').")
    advisory: Optional[str] = Field(
        default=None,
        description="Any active weather advisory or alert, if applicable.",
    )
    sources: List[str] = Field(
        default_factory=list,
        description="Data source identifiers (e.g., tool names, API endpoints).",
    )


class ResearchResponse(BaseModel):
    """Structured output from the research agent (Bucket 5)."""

    explanation: str = Field(description="The scientific explanation addressing the user's question.")
    key_concepts: List[str] = Field(
        default_factory=list,
        description="Key meteorological terms/concepts referenced (e.g., 'Rex Block', 'pyroconvection').",
    )
    citations: List[str] = Field(
        default_factory=list,
        description="Source citations from RAG retrieval (e.g., '[Source: report_name, p.7]').",
    )


class AnalystResponse(BaseModel):
    """Structured output from the analyst agent (Bucket 6)."""

    fetched_values: Dict[str, str] = Field(
        default_factory=dict,
        description="Weather values fetched for computation (e.g., {'Chicago wind chill': '-8°F'}).",
    )
    expression: str = Field(description="The mathematical expression evaluated.")
    result: str = Field(description="The final computed result with units.")
