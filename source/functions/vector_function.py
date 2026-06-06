from typing import Callable
from numpy.typing import NDArray


class VectorFunction:
    """Wraps a vector-to-vector predicate. Validates the trailing axis equals the declared input width."""

    def __init__(self, function_predicate: Callable[[NDArray], NDArray], count_of_input_features: int) -> None:
        if function_predicate is None:
            raise ValueError("function_predicate must not be None.")
        if count_of_input_features <= 0:
            raise ValueError(f"count_of_input_features must be positive, got {count_of_input_features}.")

        self._function_predicate = function_predicate
        self._count_of_input_features = count_of_input_features

    @property
    def count_of_input_features(self) -> int:
        return self._count_of_input_features

    def _execution_validation(self, parameters: NDArray) -> bool:
        if parameters.ndim < 1:
            return False
        return parameters.shape[-1] == self._count_of_input_features

    def __call__(self, parameters: NDArray) -> NDArray:
        if not self._execution_validation(parameters):
            raise ValueError(
                f"Expected trailing axis of size {self._count_of_input_features}, "
                f"got shape {tuple(parameters.shape)}."
            )
        return self._function_predicate(parameters)
