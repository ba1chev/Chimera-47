import numpy as np
from numpy.typing import NDArray

from source.functions.function import Function
from source.optimizations.derivative_calculators.derivative_calculator import DerivativeCalculator


class GradientCalculator:
    """Plain gradient descent: x_{k+1} = x_k - learning_rate * grad(f)(x_k)."""

    def __init__(self, derivative_calculator: DerivativeCalculator,
        learning_rate: float = 0.01, max_iterations: int = 1000,
        tolerance: float = 1e-6) -> None:
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}.")
        if max_iterations < 1:
            raise ValueError(f"max_iterations must be at least 1, got {max_iterations}.")
        if tolerance <= 0:
            raise ValueError(f"tolerance must be positive, got {tolerance}.")

        self._derivative_calculator: DerivativeCalculator = derivative_calculator
        self._learning_rate = learning_rate
        self._max_iterations = max_iterations
        self._tolerance = tolerance

    def minimize(self, function: Function, initial_point: NDArray) -> NDArray:
        current_point = np.asarray(initial_point, dtype=np.float64).copy()

        for _ in range(self._max_iterations):
            gradient: NDArray = self._derivative_calculator.calculate_derivative_at(function, current_point)
            if float(np.linalg.norm(gradient)) < self._tolerance:
                break
            current_point = current_point - self._learning_rate * gradient

        return current_point
