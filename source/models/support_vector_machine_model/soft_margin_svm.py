import numpy as np
from typing import Tuple
from numpy.typing import ArrayLike, NDArray

from source.functions.function import Function
from source.data.encoders.binary_label_encoder import BinaryLabelEncoder
from source.models.support_vector_machine_model.svm_model import SVMModel
from source.optimizations.gradient_descent.gradient_calculator import GradientCalculator
from source.optimizations.derivative_calculators.analytical_derivative_calculator import AnalyticalDerivativeCalculator


class SoftMarginSVM(SVMModel):
    """Soft-margin linear SVM trained by gradient descent on hinge loss + L2 regularization."""

    def __init__(self, classes: ArrayLike, binary_label_encoder: BinaryLabelEncoder, regularization_strength: float = 1.0,
        learning_rate: float = 0.01, max_iterations: int = 2000, tolerance: float = 1e-6) -> None:
        super().__init__(classes, binary_label_encoder, learning_rate, max_iterations, tolerance)
        if regularization_strength <= 0:
            raise ValueError(f"regularization_strength must be positive, got {regularization_strength}.")
        self._regularization_strength: float = regularization_strength

    def fit(self, X: ArrayLike, y: ArrayLike) -> "SoftMarginSVM":
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = self._validate_y(y)
        y_signed = self._binary_label_encoder.encode(y_arr)

        count_of_features = X_arr.shape[1]
        gradient_closure = self._build_gradient_closure(X_arr, y_signed, count_of_features)
        cost_function: Function = Function(
            function_predicate=lambda parameters: 0.0,
            count_of_variables=count_of_features + 1
        )

        optimizer = GradientCalculator(
            derivative_calculator=AnalyticalDerivativeCalculator(gradient_closure),
            learning_rate=self._learning_rate,
            max_iterations=self._max_iterations,
            tolerance=self._tolerance
        )

        flat_initial = np.zeros(count_of_features + 1, dtype=np.float64)
        flat_optimal = optimizer.minimize(cost_function, flat_initial)
        self._weights, self._bias = self._unflatten(flat_optimal, count_of_features)
        self._is_fitted = True
        return self

    def _build_gradient_closure(self, X: NDArray, y_signed: NDArray, count_of_features: int):
        regularization_strength = self._regularization_strength

        def gradient_at(flat_parameters: NDArray) -> NDArray:
            weights, bias = self._unflatten(flat_parameters, count_of_features)
            margins = y_signed * (X @ weights + bias)
            violator_mask = margins < 1.0

            grad_weights = weights.copy()
            grad_bias = 0.0
            if np.any(violator_mask):
                violators_X = X[violator_mask]
                violators_y = y_signed[violator_mask]
                grad_weights -= regularization_strength * (violators_y @ violators_X)
                grad_bias = -regularization_strength * float(violators_y.sum())

            return np.concatenate([grad_weights, np.array([grad_bias])])

        return gradient_at

    @staticmethod
    def _unflatten(flat_parameters: NDArray, count_of_features: int) -> Tuple[NDArray, float]:
        weights = flat_parameters[:count_of_features]
        bias = float(flat_parameters[count_of_features])
        return weights, bias
