"""Tests for the math_ops tool — safe evaluation and anomaly calculation."""

from weatherwise.tools.math_ops import calculate_anomaly, compute_expression


class TestComputeExpression:
    def test_simple_addition(self):
        assert "= 5.0" in compute_expression("2 + 3")

    def test_subtraction(self):
        assert "= 3.5" in compute_expression("28.5 - 25.0")

    def test_multiplication(self):
        assert "= 6.0" in compute_expression("2 * 3")

    def test_division(self):
        assert "= 2.5" in compute_expression("5 / 2")

    def test_parentheses(self):
        result = compute_expression("(2 + 3) * 4")
        assert "= 20" in result

    def test_nested_parentheses(self):
        result = compute_expression("((10 + 5) / 3) * 2")
        assert "= 10.0" in result

    def test_negative_unary(self):
        result = compute_expression("-5 + 3")
        assert "= -2" in result

    def test_division_by_zero(self):
        result = compute_expression("1 / 0")
        assert "Calculation error" in result

    def test_invalid_syntax(self):
        result = compute_expression("2 +")
        assert "Calculation error" in result

    def test_rejects_function_calls(self):
        result = compute_expression("__import__('os').system('ls')")
        assert "Calculation error" in result

    def test_rejects_attribute_access(self):
        result = compute_expression("().__class__.__bases__")
        assert "Calculation error" in result

    def test_rejects_string_literals(self):
        result = compute_expression("'hello'")
        assert "Calculation error" in result

    def test_float_precision(self):
        result = compute_expression("22.4 / 12.0")
        assert "= 1.8667" in result


class TestCalculateAnomaly:
    def test_above_normal(self):
        result = calculate_anomaly("Santos", "temp_c", 30.0)
        assert "+5.00" in result
        assert "above" in result

    def test_below_normal(self):
        result = calculate_anomaly("London", "temp_c", 5.0)
        assert "below" in result

    def test_unknown_location(self):
        result = calculate_anomaly("Atlantis", "temp_c", 20.0)
        assert "No historical baseline" in result

    def test_unknown_field(self):
        result = calculate_anomaly("Tokyo", "pressure_hpa", 1013.0)
        assert "No historical baseline" in result

    def test_case_insensitive(self):
        result = calculate_anomaly("NEW YORK", "temp_c", 15.0)
        assert "anomaly" in result
