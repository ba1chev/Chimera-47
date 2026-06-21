import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.linear_model import LogisticRegression

from source.models.supervised_learning_model import SupervisedLearningModel
from source.constants import DEFAULT_REGULARIZATION_STRENGTH, DEFAULT_MAX_ITERATIONS, DEFAULT_TOLERANCE


class MultinomialLogisticRegression(SupervisedLearningModel):
    """Native multi-class linear classifier. Wraps sklearn LogisticRegression with the lbfgs solver."""

    def __init__(self, classes: ArrayLike, regularization_strength: float = DEFAULT_REGULARIZATION_STRENGTH,
        max_iterations: int = DEFAULT_MAX_ITERATIONS, tolerance: float = DEFAULT_TOLERANCE) -> None:
        super().__init__(classes)
        if regularization_strength <= 0:
            raise ValueError(f"regularization_strength must be positive, got {regularization_strength}.")
        if max_iterations < 1:
            raise ValueError(f"max_iterations must be at least 1, got {max_iterations}.")
        if tolerance <= 0:
            raise ValueError(f"tolerance must be positive, got {tolerance}.")

        self._regularization_strength = regularization_strength
        self._max_iterations = max_iterations
        self._tolerance = tolerance
        self._estimator = None

    def fit(self, X: ArrayLike, y: ArrayLike) -> "MultinomialLogisticRegression":
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = self._validate_y(y)

        self._estimator = LogisticRegression(
            C=self._regularization_strength,
            # lbfgs is sklearn's default multinomial solver — quasi-Newton, handles the full softmax natively.
            solver="lbfgs",
            tol=self._tolerance,
            max_iter=self._max_iterations,
            fit_intercept=True
        )
        self._estimator.fit(X_arr, y_arr)
        self._is_fitted = True
        return self

    def predict(self, X: ArrayLike) -> NDArray:
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before predict().")
        X_arr = np.asarray(X, dtype=np.float64)
        return self._estimator.predict(X_arr)

    def predict_proba(self, X: ArrayLike) -> NDArray:
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before predict_proba().")
        X_arr = np.asarray(X, dtype=np.float64)
        return self._estimator.predict_proba(X_arr)

    @property
    def weights(self) -> NDArray:
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before accessing weights.")
        return self._estimator.coef_

    @property
    def biases(self) -> NDArray:
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before accessing biases.")
        return self._estimator.intercept_
