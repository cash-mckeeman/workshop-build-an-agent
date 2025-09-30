"""Tests for tools.math_tools module."""

from unittest.mock import MagicMock

import pytest

from src.tools.math_tools import add, divide, multiply, power, register_math_tools


@pytest.mark.unit
class TestMathTools:
    """Test cases for math tool functions."""

    def test_add(self):
        """Test the add function."""
        assert add(3, 12) == 15
        assert add(0, 0) == 0
        assert add(-5, 10) == 5
        assert add(-3, -7) == -10

    def test_add_type_annotations(self):
        """Test that add function has proper type annotations."""
        annotations = add.__annotations__
        assert "a" in annotations
        assert "b" in annotations
        assert "return" in annotations

    def test_multiply(self):
        """Test the multiply function."""
        assert multiply(4, 5) == 20
        assert multiply(0, 100) == 0
        assert multiply(-3, 4) == -12
        assert multiply(-2, -6) == 12

    def test_multiply_type_annotations(self):
        """Test that multiply function has proper type annotations."""
        annotations = multiply.__annotations__
        assert "a" in annotations
        assert "b" in annotations
        assert "return" in annotations

    def test_divide(self):
        """Test the divide function."""
        assert divide(15, 3) == 5.0
        assert divide(7, 2) == 3.5
        assert divide(-10, 2) == -5.0
        assert divide(0, 5) == 0.0

    def test_divide_by_zero(self):
        """Test that divide function raises ValueError for division by zero."""
        with pytest.raises(ValueError, match="Cannot divide by zero"):
            divide(10, 0)

        with pytest.raises(ValueError, match="Cannot divide by zero"):
            divide(-5, 0)

    def test_divide_type_annotations(self):
        """Test that divide function has proper type annotations."""
        annotations = divide.__annotations__
        assert "a" in annotations
        assert "b" in annotations
        assert "return" in annotations

    def test_power(self):
        """Test the power function."""
        assert power(2, 3) == 8.0
        assert power(5, 2) == 25.0
        assert power(10, 0) == 1.0
        assert power(4, 0.5) == 2.0  # Square root
        assert power(-2, 3) == -8.0

    def test_power_edge_cases(self):
        """Test power function edge cases."""
        assert power(0, 5) == 0.0
        assert power(1, 100) == 1.0
        assert power(-1, 2) == 1.0
        assert power(-1, 3) == -1.0

    def test_power_type_annotations(self):
        """Test that power function has proper type annotations."""
        annotations = power.__annotations__
        assert "base" in annotations
        assert "exponent" in annotations
        assert "return" in annotations

    def test_register_math_tools(self):
        """Test registering math tools with an agent."""
        # Create a mock agent
        mock_agent = MagicMock()
        mock_agent.tool = MagicMock(return_value=None)

        # Register tools
        result = register_math_tools(mock_agent)

        # Should return the agent
        assert result is mock_agent

        # Should have called agent.tool for each function
        assert mock_agent.tool.call_count == 4

        # Verify the functions were registered
        call_args_list = mock_agent.tool.call_args_list
        registered_functions = [call[0][0] for call in call_args_list]

        assert add in registered_functions
        assert multiply in registered_functions
        assert divide in registered_functions
        assert power in registered_functions

    def test_function_docstrings(self):
        """Test that functions have proper docstrings."""
        assert add.__doc__ is not None
        assert "Add two integers" in add.__doc__

        assert multiply.__doc__ is not None
        assert "Multiply two integers" in multiply.__doc__

        assert divide.__doc__ is not None
        assert "Divide one number" in divide.__doc__

        assert power.__doc__ is not None
        assert "Raise a number to a power" in power.__doc__

    def test_numerical_precision(self):
        """Test numerical precision for floating point operations."""
        # Test divide precision
        result = divide(1, 3)
        assert abs(result - 0.3333333333333333) < 1e-10

        # Test power precision
        result = power(2, 10)
        assert result == 1024.0

    def test_large_numbers(self):
        """Test functions with large numbers."""
        # Test with reasonably large numbers
        assert add(1000000, 2000000) == 3000000
        assert multiply(1000, 2000) == 2000000
        assert divide(1000000, 1000) == 1000.0
        assert power(10, 6) == 1000000.0

    def test_function_parameter_names(self):
        """Test that functions have meaningful parameter names."""
        import inspect

        # Test add parameters
        add_params = list(inspect.signature(add).parameters.keys())
        assert add_params == ["a", "b"]

        # Test multiply parameters
        multiply_params = list(inspect.signature(multiply).parameters.keys())
        assert multiply_params == ["a", "b"]

        # Test divide parameters
        divide_params = list(inspect.signature(divide).parameters.keys())
        assert divide_params == ["a", "b"]

        # Test power parameters
        power_params = list(inspect.signature(power).parameters.keys())
        assert power_params == ["base", "exponent"]
