import pytest
import numpy as np

from source.functions.function import Function
from source.optimizations.derivative_calculators.finite_difference_derivative_calculator import FiniteDifferenceDerivativeCalculator


def make_quadratic(coefficients: np.ndarray) -> Function:
    return Function(
        function_predicate=lambda parameters: float(np.sum(coefficients * parameters**2)),
        count_of_variables=coefficients.shape[0]
    )


class TestFiniteDifferenceDerivativeCalculator:
    def test_constructor_raises_on_zero_epsilon(self):
        with pytest.raises(ValueError, match="epsilon"):
            FiniteDifferenceDerivativeCalculator(epsilon=0.0)

    def test_constructor_raises_on_negative_epsilon(self):
        with pytest.raises(ValueError, match="epsilon"):
            FiniteDifferenceDerivativeCalculator(epsilon=-1e-6)

    def test_gradient_of_quadratic_matches_analytical(self):
        coefficients = np.array([1.0, 2.0, 3.0])
        function = make_quadratic(coefficients)
        point = np.array([1.0, 1.0, 1.0])

        calculator = FiniteDifferenceDerivativeCalculator(epsilon=1e-6)
        gradient = calculator.calculate_derivative_at(function, point)
        np.testing.assert_allclose(gradient, 2 * coefficients * point, atol=1e-5)

    def test_gradient_is_zero_at_minimum(self):
        function = make_quadratic(np.array([1.0, 1.0]))
        point = np.array([0.0, 0.0])

        calculator = FiniteDifferenceDerivativeCalculator(epsilon=1e-6)
        gradient = calculator.calculate_derivative_at(function, point)
        np.testing.assert_allclose(gradient, [0.0, 0.0], atol=1e-9)

    def test_gradient_shape_matches_input(self):
        function = make_quadratic(np.ones(7))
        point = np.zeros(7)

        calculator = FiniteDifferenceDerivativeCalculator()
        gradient = calculator.calculate_derivative_at(function, point)
        assert gradient.shape == (7,)

    def test_central_difference_is_more_accurate_than_forward(self):
        function = Function(lambda parameters: float(parameters[0] ** 3), count_of_variables=1)
        point = np.array([2.0])

        calculator = FiniteDifferenceDerivativeCalculator(epsilon=1e-3)
        gradient = calculator.calculate_derivative_at(function, point)
        np.testing.assert_allclose(gradient, [12.0], atol=1e-5)
