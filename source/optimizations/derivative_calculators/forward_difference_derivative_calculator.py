import numpy as np
from numpy.typing import NDArray

from source.functions.function import Function
from source.optimizations.derivative_calculators.derivative_calculator import DerivativeCalculator


class ForwardDifferenceDerivativeCalculator(DerivativeCalculator):
    """Numerical gradient via forward finite differences: (f(x + h) - f(x)) / h, per coordinate."""

    def __init__(self, step_size: float = 1e-6) -> None:
        if step_size <= 0:
            raise ValueError(f"step_size must be positive, got {step_size}.")
        self._step_size: float = step_size

    def calculate_derivative_at(self, function: Function, parameters: NDArray) -> NDArray:
        parameters_float: NDArray = np.asarray(parameters, dtype=np.float64)
        gradient: NDArray = np.empty_like(parameters_float)

        value_at_point: float = float(function(parameters_float))

        for coordinate_index in range(parameters_float.shape[0]):
            shifted: NDArray = parameters_float.copy()
            shifted[coordinate_index] += self._step_size
            value_shifted: float = float(function(shifted))
            gradient[coordinate_index] = (value_shifted - value_at_point) / self._step_size

        return gradient
