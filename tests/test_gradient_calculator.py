import pytest
import numpy as np

from source.functions.function import Function
from source.optimizations.gradient_descent.gradient_calculator import GradientCalculator
from source.optimizations.derivative_calculators.analytical_derivative_calculator import AnalyticalDerivativeCalculator


class TestGradientCalculator:
    def test_constructor_raises_on_zero_learning_rate(self):
        with pytest.raises(ValueError, match="learning_rate"):
            GradientCalculator(
                derivative_calculator=AnalyticalDerivativeCalculator(lambda p: p),
                learning_rate=0.0
            )

    def test_constructor_raises_on_zero_max_iterations(self):
        with pytest.raises(ValueError, match="max_iterations"):
            GradientCalculator(
                derivative_calculator=AnalyticalDerivativeCalculator(lambda p: p),
                max_iterations=0
            )

    def test_constructor_raises_on_negative_tolerance(self):
        with pytest.raises(ValueError, match="tolerance"):
            GradientCalculator(
                derivative_calculator=AnalyticalDerivativeCalculator(lambda p: p),
                tolerance=-1.0
            )

    def test_finds_minimum_of_quadratic(self):
        gradient_function = lambda parameters: 2.0 * parameters
        function = Function(lambda parameters: float(np.sum(parameters**2)), count_of_variables=3)

        optimizer = GradientCalculator(
            derivative_calculator=AnalyticalDerivativeCalculator(gradient_function),
            learning_rate=0.1,
            max_iterations=1000,
            tolerance=1e-8
        )
        result = optimizer.minimize(function, np.array([5.0, -3.0, 2.0]))
        np.testing.assert_allclose(result, [0.0, 0.0, 0.0], atol=1e-4)

    def test_finds_shifted_minimum(self):
        target = np.array([3.0, -2.0, 7.0])
        gradient_function = lambda parameters: 2.0 * (parameters - target)
        function = Function(
            lambda parameters: float(np.sum((parameters - target) ** 2)),
            count_of_variables=3
        )

        optimizer = GradientCalculator(
            derivative_calculator=AnalyticalDerivativeCalculator(gradient_function),
            learning_rate=0.1,
            max_iterations=1000,
            tolerance=1e-9
        )
        result = optimizer.minimize(function, np.zeros(3))
        np.testing.assert_allclose(result, target, atol=1e-4)

    def test_does_not_modify_initial_point(self):
        initial = np.array([5.0, 5.0])
        before = initial.copy()
        optimizer = GradientCalculator(
            derivative_calculator=AnalyticalDerivativeCalculator(lambda p: 2.0 * p),
            learning_rate=0.1,
            max_iterations=10
        )
        optimizer.minimize(
            Function(lambda parameters: float(np.sum(parameters**2)), count_of_variables=2),
            initial
        )
        np.testing.assert_array_equal(initial, before)

    def test_stops_early_when_gradient_below_tolerance(self):
        call_count = {"value": 0}

        def gradient_function(parameters):
            call_count["value"] += 1
            return 2.0 * parameters

        optimizer = GradientCalculator(
            derivative_calculator=AnalyticalDerivativeCalculator(gradient_function),
            learning_rate=0.1,
            max_iterations=10000,
            tolerance=1e-6
        )
        optimizer.minimize(
            Function(lambda parameters: float(np.sum(parameters**2)), count_of_variables=2),
            np.array([1.0, 1.0])
        )
        assert call_count["value"] < 10000
