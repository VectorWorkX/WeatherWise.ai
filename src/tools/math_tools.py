"""Calculator tool using numexpr for safe math evaluation."""

from __future__ import annotations

import numexpr
from langchain_core.tools import tool


@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression. Input must be a valid numerical expression
    that numexpr can parse (e.g. '15.5 + 20.2', '(72 - 32) * 5/9 + 273.15').
    Use this for any arithmetic on weather data such as unit conversions,
    temperature sums, or averages.
    """
    try:
        result = numexpr.evaluate(expression).item()
        return str(result)
    except Exception as exc:
        return f"Error evaluating '{expression}': {exc}"
