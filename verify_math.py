"""Verify the calculator tool works."""

from __future__ import annotations

import sys

sys.path.insert(0, "src")

from tools.math_tools import calculator  # noqa: E402

result = calculator.invoke({"expression": "15.5 + 20.2"})
print(f"15.5 + 20.2 = {result}")
