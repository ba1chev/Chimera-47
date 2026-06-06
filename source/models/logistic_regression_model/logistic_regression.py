import numpy as np
from typing import Tuple
from numpy.typing import ArrayLike, NDArray

from source.functions.function import Function
from source.functions.vector_function import VectorFunction
from source.data.encoders.one_hot_encoder import OneHotEncoder
from source.models.supervised_learning_model import SupervisedLearningModel
from source.optimizations.gradient_descent.gradient_calculator import GradientCalculator
from source.functions.vector_function_predicates.softmax_function_predicate import softmax_function_predicate
from source.optimizations.derivative_calculators.analytical_derivative_calculator import AnalyticalDerivativeCalculator


class LogisticRegression(SupervisedLearningModel):
    """Multinomial logistic regression trained by gradient descent on cross-entropy loss."""

    def __init__(self, classes: ArrayLike, one_hot_encoder: OneHotEncoder,
        learning_rate: float = 0.1, max_iterations: int = 2000, tolerance: float = 1e-6) -> None:
        super().__init__(classes)
        if one_hot_encoder is None:
            raise ValueError("one_hot_encoder must not be None.")
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}.")
        if max_iterations < 1:
            raise ValueError(f"max_iterations must be at least 1, got {max_iterations}.")
        if tolerance <= 0:
            raise ValueError(f"tolerance must be positive, got {tolerance}.")

        self._one_hot_encoder: OneHotEncoder = one_hot_encoder
        self._learning_rate = learning_rate
        self._max_iterations = max_iterations
        self._tolerance = tolerance
        self._weights = None
        self._bias = None
        self._softmax_function: VectorFunction | None = None

    def fit(self, X: ArrayLike, y: ArrayLike) -> "LogisticRegression":
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = self._validate_y(y)

        count_of_features = X_arr.shape[1]
        count_of_classes = self.count_of_classes()
        y_one_hot = self._one_hot_encoder.encode(y_arr)

        softmax = VectorFunction(softmax_function_predicate, count_of_classes)
        gradient_closure = self._build_gradient_closure(X_arr, y_one_hot, softmax, count_of_features, count_of_classes)
        cost_function: Function = Function(
            function_predicate=lambda parameters: 0.0,
            count_of_variables=count_of_features * count_of_classes + count_of_classes
        )

        optimizer = GradientCalculator(
            derivative_calculator=AnalyticalDerivativeCalculator(gradient_closure),
            learning_rate=self._learning_rate,
            max_iterations=self._max_iterations,
            tolerance=self._tolerance
        )

        flat_initial = np.zeros(
            count_of_features * count_of_classes + count_of_classes, dtype=np.float64
        )
        flat_optimal = optimizer.minimize(cost_function, flat_initial)
        self._weights, self._bias = self._unflatten(flat_optimal, count_of_features, count_of_classes)
        self._softmax_function = softmax
        self._is_fitted = True
        return self

    def predict(self, X: ArrayLike) -> NDArray:
        probabilities = self.predict_proba(X)
        return self._one_hot_encoder.decode(probabilities)

    def predict_proba(self, X: ArrayLike) -> NDArray:
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before predict_proba().")
        X_arr = np.asarray(X, dtype=np.float64)
        logits = X_arr @ self._weights + self._bias
        return self._softmax_function(logits)

    def _build_gradient_closure(self, X: NDArray, y_one_hot: NDArray, softmax: VectorFunction,
        count_of_features: int, count_of_classes: int):
        count_of_samples = X.shape[0]

        def gradient_at(flat_parameters: NDArray) -> NDArray:
            weights, bias = self._unflatten(flat_parameters, count_of_features, count_of_classes)
            logits = X @ weights + bias
            probabilities = softmax(logits)
            error = (probabilities - y_one_hot) / count_of_samples
            grad_weights = X.T @ error
            grad_bias = error.sum(axis=0)
            return np.concatenate([grad_weights.ravel(), grad_bias])

        return gradient_at

    @staticmethod
    def _unflatten(flat_parameters: NDArray, count_of_features: int,
        count_of_classes: int) -> Tuple[NDArray, NDArray]:
        weight_size = count_of_features * count_of_classes
        weights = flat_parameters[:weight_size].reshape(count_of_features, count_of_classes)
        bias = flat_parameters[weight_size:]
        return weights, bias
