import numpy as np
from typing import Tuple
from numpy.typing import ArrayLike, NDArray

from source.functions.function import Function
from source.models.supervised_learning_model import SupervisedLearningModel
from source.optimizations.gradient_descent.gradient_calculator import GradientCalculator
from source.optimizations.derivative_calculators.analytical_derivative_calculator import AnalyticalDerivativeCalculator


class LogisticRegression(SupervisedLearningModel):
    """Multinomial logistic regression trained by gradient descent on cross-entropy loss."""

    def __init__(self, classes: ArrayLike, learning_rate: float = 0.1,
        max_iterations: int = 2000, tolerance: float = 1e-6) -> None:
        super().__init__(classes)
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}.")
        if max_iterations < 1:
            raise ValueError(f"max_iterations must be at least 1, got {max_iterations}.")
        if tolerance <= 0:
            raise ValueError(f"tolerance must be positive, got {tolerance}.")

        self._learning_rate = learning_rate
        self._max_iterations = max_iterations
        self._tolerance = tolerance
        self._weights = None
        self._bias = None

    def fit(self, X: ArrayLike, y: ArrayLike) -> "LogisticRegression":
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = self._validate_y(y)

        count_of_features = X_arr.shape[1]
        count_of_classes = self.count_of_classes()
        y_one_hot = self._encode_one_hot(y_arr)

        gradient_closure = self._build_gradient_closure(X_arr, y_one_hot, count_of_features, count_of_classes)
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
        self._is_fitted = True
        return self

    def predict(self, X: ArrayLike) -> NDArray:
        probabilities = self.predict_proba(X)
        return self._classes[np.argmax(probabilities, axis=1)]

    def predict_proba(self, X: ArrayLike) -> NDArray:
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before predict_proba().")
        X_arr = np.asarray(X, dtype=np.float64)
        logits = X_arr @ self._weights + self._bias
        return self._softmax(logits)

    def _build_gradient_closure(self, X: NDArray, y_one_hot: NDArray,
        count_of_features: int, count_of_classes: int):
        count_of_samples = X.shape[0]

        def gradient_at(flat_parameters: NDArray) -> NDArray:
            weights, bias = self._unflatten(flat_parameters, count_of_features, count_of_classes)
            logits = X @ weights + bias
            probabilities = self._softmax(logits)
            error = (probabilities - y_one_hot) / count_of_samples
            grad_weights = X.T @ error
            grad_bias = error.sum(axis=0)
            return np.concatenate([grad_weights.ravel(), grad_bias])

        return gradient_at

    def _encode_one_hot(self, y: NDArray) -> NDArray:
        count_of_samples = y.shape[0]
        count_of_classes = self.count_of_classes()
        one_hot = np.zeros((count_of_samples, count_of_classes), dtype=np.float64)
        for class_index, class_label in enumerate(self._classes):
            one_hot[y == class_label, class_index] = 1.0
        return one_hot

    @staticmethod
    def _softmax(logits: NDArray) -> NDArray:
        shifted = logits - logits.max(axis=1, keepdims=True)
        exponentials = np.exp(shifted)
        return exponentials / exponentials.sum(axis=1, keepdims=True)

    @staticmethod
    def _unflatten(flat_parameters: NDArray, count_of_features: int,
        count_of_classes: int) -> Tuple[NDArray, NDArray]:
        weight_size = count_of_features * count_of_classes
        weights = flat_parameters[:weight_size].reshape(count_of_features, count_of_classes)
        bias = flat_parameters[weight_size:]
        return weights, bias
