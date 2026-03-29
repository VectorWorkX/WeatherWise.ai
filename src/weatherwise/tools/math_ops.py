"""Math operations tools — safe arithmetic and sandboxed Python execution."""

from __future__ import annotations

import ast
import collections
import datetime
import json
import math
import operator
import re
import signal
import statistics

from langchain_core.tools import tool

from weatherwise.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Simple arithmetic evaluator (for quick expressions like "23.11 - 20.59")
# ---------------------------------------------------------------------------

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
def compute_expression(expression: str) -> str:
    """Evaluate a simple math expression safely. Input must be a valid numerical
    expression using only numbers and basic operators (+, -, *, /).

    Use this ONLY for simple, quick arithmetic where you already have the exact
    numbers — such as unit conversions, temperature differences, or sums of
    2-3 known values.

    Do NOT use this for extracting data from tables or computing statistics
    over many values — use run_calculation for those tasks instead.

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


# ---------------------------------------------------------------------------
# Sandboxed Python execution (for complex computations, table parsing, stats)
# ---------------------------------------------------------------------------

# Maximum code length to prevent memory bombs
_MAX_CODE_LENGTH = 15000

# Execution timeout in seconds
_EXEC_TIMEOUT = 5

# Builtins that are safe for numerical computation
_SAFE_BUILTINS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "len": len,
    "sorted": sorted,
    "enumerate": enumerate,
    "zip": zip,
    "range": range,
    "list": list,
    "dict": dict,
    "set": set,
    "tuple": tuple,
    "float": float,
    "int": int,
    "str": str,
    "bool": bool,
    "True": True,
    "False": False,
    "None": None,
    "isinstance": isinstance,
    "print": print,
    "chr": chr,
    "ord": ord,
    "map": map,
    "filter": filter,
    "any": any,
    "all": all,
    # Exception types — LLMs write try/except blocks that need these
    "ValueError": ValueError,
    "TypeError": TypeError,
    "KeyError": KeyError,
    "IndexError": IndexError,
    "Exception": Exception,
    "ZeroDivisionError": ZeroDivisionError,
}


def _strip_imports(code: str) -> str:
    """Remove import statements from code.

    The sandbox pre-loads math, statistics, and re as globals.
    LLMs write 'import re' or 'import statistics' out of habit — these
    are redundant but harmless. Strip them instead of rejecting the code.
    """
    lines = code.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            logger.info("sandbox_import_stripped", line=stripped)
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def _validate_code(code: str) -> str | None:
    """Validate Python code via AST analysis. Returns error message or None if safe."""
    if len(code) > _MAX_CODE_LENGTH:
        return f"Code too long: {len(code)} chars (max {_MAX_CODE_LENGTH})"

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"Syntax error: {e}"

    for node in ast.walk(tree):
        # Block dunder attribute access (sandbox escape vector)
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return f"Access to __{node.attr.lstrip('_')} is not allowed"

    return None


def _exec_with_timeout(code: str, globals_dict: dict, locals_dict: dict) -> None:
    """Execute code with a timeout to prevent infinite loops."""
    def _timeout_handler(signum, frame):
        raise TimeoutError("Code execution timed out")

    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(_EXEC_TIMEOUT)
    try:
        exec(code, globals_dict, locals_dict)  # noqa: S102
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


@tool
def run_calculation(code: str, data: str = "") -> str:
    """Execute a Python code snippet in a secure sandbox for complex calculations.

    Use this for any computation that involves:
    - Parsing tables or structured text to extract numeric values
    - Statistical operations (mean, std dev, correlation, percentile, regression)
    - Processing many data points (e.g., 30 days of temperatures)
    - Multi-step calculations that are easier to express as code
    - Any math beyond simple arithmetic

    The code runs in a restricted environment. You MUST assign your final
    answer to a variable called `result`. The `data` parameter is available
    as the variable `data` in your code.

    Args:
        code: Python code snippet. Must assign the final answer to `result`.
              Available modules: math, statistics, json, re, datetime, collections.
              Available builtins: abs, round, min, max, sum, len, sorted,
              enumerate, zip, range, list, dict, set, tuple, float, int, str, bool.
        data: Optional data string to process (available as `data` variable in code).
              Pass the raw weather data here so your code can parse it.

    Returns:
        The string representation of the `result` variable, or an error message.

    Example:
        code: '''
        # Parse temperatures from a table
        lines = data.strip().split('\\n')
        temps = []
        for line in lines:
            parts = [p.strip() for p in line.split('|') if p.strip()]
            if len(parts) >= 4 and parts[0].startswith('20'):
                try:
                    temps.append(float(parts[3]))
                except ValueError:
                    pass
        result = {
            'count': len(temps),
            'mean': statistics.mean(temps),
            'std_dev': statistics.stdev(temps),
            'min': min(temps),
            'max': max(temps),
        }
        '''
    """
    # Strip import statements (modules are pre-loaded as globals)
    code = _strip_imports(code)

    # Validate code safety via AST analysis
    validation_error = _validate_code(code)
    if validation_error:
        return f"Code validation error: {validation_error}"

    # Build restricted execution environment
    restricted_globals = {
        "__builtins__": _SAFE_BUILTINS,
        "math": math,
        "statistics": statistics,
        "re": re,
        "json": json,
        "datetime": datetime,
        "collections": collections,
        "data": data,
    }
    local_vars: dict = {}

    try:
        _exec_with_timeout(code, restricted_globals, local_vars)
    except TimeoutError:
        return f"Code execution timed out after {_EXEC_TIMEOUT} seconds"
    except Exception as e:
        return f"Code execution error: {type(e).__name__}: {e}"

    if "result" not in local_vars:
        return "Error: code must assign a value to the `result` variable"

    result = local_vars["result"]
    logger.info("run_calculation_success", result_type=type(result).__name__)
    return str(result)
