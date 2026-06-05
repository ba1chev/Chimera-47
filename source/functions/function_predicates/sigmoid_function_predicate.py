import numpy as np
from numpy.typing import NDArray


def sigmoid_function_predicate(parameters: NDArray) -> float:
    n_total = parameters.shape[0]
    if n_total < 3 or n_total % 2 == 0:
        raise ValueError(f"Expected parameters of shape (1 + 2*n_features,), got length {n_total}.")

    n_features = (n_total - 1) // 2
    bias = float(parameters[0])
    weights = parameters[1 : 1 + n_features]
    inputs = parameters[1 + n_features :]

    exp_sum = bias + float(np.dot(weights, inputs))
    return float(1.0 / (1.0 + np.exp(-exp_sum)))
