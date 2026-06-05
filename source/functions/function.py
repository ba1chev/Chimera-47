import numpy as np
from numpy.typing import NDArray
from typing import Callable, Generic, TypeVar

TDomainDType = TypeVar("TDomainDType", bound=np.generic)
TRange = TypeVar("TRange")


class Function(Generic[TDomainDType, TRange]):
    def __init__(self, function_predicate: Callable[[NDArray], TRange], count_of_variables: int) -> None:
        if function_predicate is None:
            raise ValueError("function_predicate must not be None.")
        if count_of_variables < 0:
            raise ValueError(f"count_of_variables must be non-negative, got {count_of_variables}.")

        self._function_predicate = function_predicate
        self._count_of_variables = count_of_variables

    @property
    def arity(self) -> int:
        return self._count_of_variables

    def _execution_validation(self, parameters: NDArray) -> bool:
        return parameters.shape[0] == self._count_of_variables

    def __call__(self, parameters: NDArray) -> TRange:
        if not self._execution_validation(parameters):
            raise ValueError(
                f"Expected {self._count_of_variables} parameters, "
                f"got {parameters.shape[0]}."
            )
        return self._function_predicate(parameters)
