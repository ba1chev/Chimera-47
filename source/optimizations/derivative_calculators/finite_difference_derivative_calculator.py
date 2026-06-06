import numpy as np
from numpy.typing import NDArray

from source.functions.function import Function
from source.optimizations.derivative_calculators.derivative_calculator import DerivativeCalculator


class FiniteDifferenceDerivativeCalculator(DerivativeCalculator):
    """Numerical gradient via central finite differences: (f(x + e) - f(x - e)) / (2e), per coordinate."""

    def __init__(self, epsilon: float = 1e-6) -> None:
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}.")
        self._epsilon: float = epsilon

    def calculate_derivative_at(self, function: Function, parameters: NDArray) -> NDArray:
        parameters_float = np.asarray(parameters, dtype=np.float64)
        gradient = np.empty_like(parameters_float)

        for coordinate_index in range(parameters_float.shape[0]):
            shifted_up = parameters_float.copy()
            shifted_down = parameters_float.copy()
            shifted_up[coordinate_index] += self._epsilon
            shifted_down[coordinate_index] -= self._epsilon

            value_up = float(function(shifted_up))
            value_down = float(function(shifted_down))
            gradient[coordinate_index] = (value_up - value_down) / (2.0 * self._epsilon)

        return gradient
