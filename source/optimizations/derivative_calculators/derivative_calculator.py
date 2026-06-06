from numpy.typing import NDArray
from abc import ABC, abstractmethod

from source.functions.function import Function


class DerivativeCalculator(ABC):
    """Abstract base for gradient calculators: maps (function, point) -> gradient vector."""

    @abstractmethod
    def calculate_derivative_at(self, function: Function, parameters: NDArray) -> NDArray:
        raise NotImplementedError("Must be implemented")
