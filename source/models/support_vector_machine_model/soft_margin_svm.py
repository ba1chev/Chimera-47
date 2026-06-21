import numpy as np
from sklearn.svm import LinearSVC
from numpy.typing import ArrayLike

from source.data.encoders.binary_label_encoder import BinaryLabelEncoder
from source.models.support_vector_machine_model.svm_model import SVMModel
from source.constants import (
    DEFAULT_REGULARIZATION_STRENGTH, DEFAULT_SVM_LEARNING_RATE,
    DEFAULT_MAX_ITERATIONS, DEFAULT_TOLERANCE
)


class SoftMarginSVM(SVMModel):
    """Soft-margin linear SVM. Wraps sklearn.svm.LinearSVC (hinge loss + L2 regularization)."""

    def __init__(self, classes: ArrayLike, binary_label_encoder: BinaryLabelEncoder,
        regularization_strength: float = DEFAULT_REGULARIZATION_STRENGTH,
        learning_rate: float = DEFAULT_SVM_LEARNING_RATE,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        tolerance: float = DEFAULT_TOLERANCE) -> None:
        super().__init__(classes, binary_label_encoder, learning_rate, max_iterations, tolerance)
        if regularization_strength <= 0:
            raise ValueError(f"regularization_strength must be positive, got {regularization_strength}.")
        self._regularization_strength = regularization_strength

    def fit(self, X: ArrayLike, y: ArrayLike) -> "SoftMarginSVM":
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = self._validate_y(y)
        y_signed = self._binary_label_encoder.encode(y_arr)

        estimator = LinearSVC(
            C=self._regularization_strength,
            loss="hinge",
            penalty="l2",
            tol=self._tolerance,
            max_iter=self._max_iterations,
            fit_intercept=True,
            dual=True
        )
        estimator.fit(X_arr, y_signed)

        self._weights = estimator.coef_.ravel().astype(np.float64)
        self._bias = float(estimator.intercept_[0])
        self._is_fitted = True
        return self
