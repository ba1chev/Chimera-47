import numpy as np
from abc import abstractmethod
from numpy.typing import ArrayLike, NDArray

from source.constants import DEFAULT_SVM_LEARNING_RATE, DEFAULT_MAX_ITERATIONS, DEFAULT_TOLERANCE
from source.data.encoders.binary_label_encoder import BinaryLabelEncoder
from source.models.supervised_learning_model import SupervisedLearningModel


class SVMModel(SupervisedLearningModel):
    """Abstract linear binary SVM. Subclasses define how the weight vector and bias are fit."""

    def __init__(self, classes: ArrayLike, binary_label_encoder: BinaryLabelEncoder,
        learning_rate: float = DEFAULT_SVM_LEARNING_RATE, max_iterations: int = DEFAULT_MAX_ITERATIONS,
        tolerance: float = DEFAULT_TOLERANCE) -> None:
        super().__init__(classes)
        if self.count_of_classes() != 2:
            raise ValueError(f"SVMModel requires exactly 2 classes, got {self.count_of_classes()}.")
        if binary_label_encoder is None:
            raise ValueError("binary_label_encoder must not be None.")
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}.")
        if max_iterations < 1:
            raise ValueError(f"max_iterations must be at least 1, got {max_iterations}.")
        if tolerance <= 0:
            raise ValueError(f"tolerance must be positive, got {tolerance}.")

        self._binary_label_encoder = binary_label_encoder
        self._learning_rate = learning_rate
        self._max_iterations  = max_iterations
        self._tolerance = tolerance
        self._weights = None
        self._bias = None

    @abstractmethod
    def fit(self, X: ArrayLike, y: ArrayLike) -> "SVMModel":
        raise NotImplementedError("Must be implemented")

    def predict(self, X: ArrayLike) -> NDArray:
        scores = self.decision_function(X)
        return self._binary_label_encoder.decode(np.sign(scores))

    def decision_function(self, X: ArrayLike) -> NDArray:
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before decision_function().")
        X_arr = np.asarray(X, dtype=np.float64)
        return X_arr @ self._weights + self._bias

    @property
    def weights(self) -> NDArray:
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before accessing weights.")
        return self._weights

    @property
    def bias(self) -> float:
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before accessing bias.")
        return self._bias
