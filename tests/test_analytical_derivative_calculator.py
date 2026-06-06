import pytest
import numpy as np

from source.functions.function import Function
from source.optimizations.derivative_calculators.analytical_derivative_calculator import AnalyticalDerivativeCalculator


class TestAnalyticalDerivativeCalculator:
    def test_constructor_raises_on_none_gradient_function(self):
        with pytest.raises(ValueError, match="gradient_function"):
            AnalyticalDerivativeCalculator(None)

    def test_returns_result_of_provided_gradient_function(self):
        gradient_function = lambda parameters: parameters * 2.0
        calculator = AnalyticalDerivativeCalculator(gradient_function)
        function = Function(lambda parameters: 0.0, count_of_variables=3)

        result = calculator.calculate_derivative_at(function, np.array([1.0, 2.0, 3.0]))
        np.testing.assert_array_equal(result, [2.0, 4.0, 6.0])

    def test_ignores_function_argument(self):
        gradient_function = lambda parameters: np.array([42.0, 43.0])
        calculator = AnalyticalDerivativeCalculator(gradient_function)

        first = calculator.calculate_derivative_at(
            Function(lambda parameters: 1.0, count_of_variables=2),
            np.array([0.0, 0.0])
        )
        second = calculator.calculate_derivative_at(
            Function(lambda parameters: -999.0, count_of_variables=2),
            np.array([0.0, 0.0])
        )
        np.testing.assert_array_equal(first, second)
