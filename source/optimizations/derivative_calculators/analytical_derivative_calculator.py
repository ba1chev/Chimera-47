from typing import Callable
from numpy.typing import NDArray

from source.functions.function import Function
from source.optimizations.derivative_calculators.derivative_calculator import DerivativeCalculator


class AnalyticalDerivativeCalculator(DerivativeCalculator):
    """Wraps a hand-written gradient closure into the DerivativeCalculator contract."""

    def __init__(self, gradient_function: Callable[[NDArray], NDArray]) -> None:
        if gradient_function is None:
            raise ValueError("gradient_function must not be None.")
        self._gradient_function = gradient_function

    def calculate_derivative_at(self, function: Function, parameters: NDArray) -> NDArray:
        return self._gradient_function(parameters)
