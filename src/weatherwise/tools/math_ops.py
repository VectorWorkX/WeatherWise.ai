"""Math operations tool — safe evaluation of weather-related calculations."""

from __future__ import annotations

import ast
import operator

from langchain_core.tools import tool


# Historical averages for anomaly calculations (mock baseline)
HISTORICAL_AVERAGES: dict[str, dict[str, float]] = {
    "santos": {"temp_c": 25.0, "wind_kph": 12.0, "humidity_pct": 75.0},
    "new york": {"temp_c": 10.5, "wind_kph": 14.0, "humidity_pct": 60.0},
    "london": {"temp_c": 11.0, "wind_kph": 16.0, "humidity_pct": 78.0},
    "tokyo": {"temp_c": 15.8, "wind_kph": 11.0, "humidity_pct": 63.0},
    "mumbai": {"temp_c": 27.5, "wind_kph": 10.0, "humidity_pct": 72.0},
}

_SAFE_OPS: dict[type, object] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node: ast.AST) -> float:
    """Recursively evaluate an AST node containing only arithmetic."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp):
        op_func = _SAFE_OPS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        return op_func(_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        op_func = _SAFE_OPS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        return op_func(_safe_eval(node.operand))
    raise ValueError(f"Unsupported expression element: {type(node).__name__}")


@tool
def calculate_anomaly(
    location: str, field: str, current_value: float
) -> str:
    """Calculate how much a current weather value deviates from the historical average.

    Args:
        location: City name (case-insensitive), e.g. "Santos", "New York".
        field: Weather field — one of "temp_c", "wind_kph", "humidity_pct".
        current_value: The current observed value to compare against the baseline.

    Returns:
        Human-readable anomaly result string.
    """
    key = location.strip().lower()
    averages = HISTORICAL_AVERAGES.get(key)

    if averages is None:
        return f"No historical baseline available for {location}."

    baseline = averages.get(field)
    if baseline is None:
        return f"No historical baseline for '{field}' in {location}."

    anomaly = round(current_value - baseline, 2)
    direction = "above" if anomaly > 0 else "below"
    return (
        f"{location.title()} {field} anomaly: {current_value} vs "
        f"historical avg {baseline} = {anomaly:+.2f} ({abs(anomaly)} {direction} normal)"
    )


@tool
def compute_expression(expression: str) -> str:
    """Evaluate a math expression safely. Input must be a valid numerical expression
    using only numbers and basic operators (+, -, *, /).

    Use this for any arithmetic on weather data such as unit conversions,
    temperature sums, differences, or averages.

    Args:
        expression: A string like "28.5 - 25.0" or "(72 - 32) * 5/9 + 273.15".

    Returns:
        Result string, or error message.
    """
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree)
        return f"{expression} = {round(result, 4)}"
    except (SyntaxError, ValueError, ZeroDivisionError) as e:
        return f"Calculation error: {e}"
