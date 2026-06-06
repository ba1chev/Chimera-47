import pytest
import numpy as np

from source.functions.function import Function
from source.optimizations.derivative_calculators.forward_difference_derivative_calculator import ForwardDifferenceDerivativeCalculator


class TestForwardDifferenceDerivativeCalculator:
    def test_constructor_raises_on_zero_step_size(self):
        with pytest.raises(ValueError, match="step_size"):
            ForwardDifferenceDerivativeCalculator(step_size=0.0)

    def test_constructor_raises_on_negative_step_size(self):
        with pytest.raises(ValueError, match="step_size"):
            ForwardDifferenceDerivativeCalculator(step_size=-1.0)

    def test_gradient_of_linear_function_is_constant(self):
        function = Function(
            function_predicate=lambda parameters: float(3.0 * parameters[0] + 5.0 * parameters[1]),
            count_of_variables=2
        )
        calculator = ForwardDifferenceDerivativeCalculator(step_size=1e-6)
        gradient = calculator.calculate_derivative_at(function, np.array([0.0, 0.0]))
        np.testing.assert_allclose(gradient, [3.0, 5.0], atol=1e-4)

    def test_gradient_of_quadratic_matches_analytical(self):
        function = Function(
            function_predicate=lambda parameters: float(np.sum(parameters**2)),
            count_of_variables=3
        )
        calculator = ForwardDifferenceDerivativeCalculator(step_size=1e-6)
        gradient = calculator.calculate_derivative_at(function, np.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(gradient, [2.0, 4.0, 6.0], atol=1e-3)

    def test_gradient_shape_matches_input(self):
        function = Function(lambda parameters: float(parameters.sum()), count_of_variables=5)
        calculator = ForwardDifferenceDerivativeCalculator()
        gradient = calculator.calculate_derivative_at(function, np.zeros(5))
        assert gradient.shape == (5,)
