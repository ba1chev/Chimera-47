import numpy as np
from numpy.typing import NDArray


def softmax_function_predicate(parameters: NDArray) -> NDArray:
    shifted: NDArray = parameters - parameters.max(axis=-1, keepdims=True)
    exponentials: NDArray = np.exp(shifted)
    return exponentials / exponentials.sum(axis=-1, keepdims=True)
