import numpy as np
from abc import abstractmethod
from numpy.typing import ArrayLike, NDArray

from source.models.learning_model import LearningModel


class SupervisedLearningModel(LearningModel):
    """Abstract supervised model. Classes are declared upfront via the constructor."""

    def __init__(self, classes: ArrayLike) -> None:
        super().__init__()
        if classes is None or len(classes) < 2:
            raise ValueError("SupervisedLearningModel requires at least 2 classes.")
        self._classes: NDArray = np.asarray(classes)

    @abstractmethod
    def fit(self, X: ArrayLike, y: ArrayLike) -> "SupervisedLearningModel":
        raise NotImplementedError("Must be implemented")

    @abstractmethod
    def predict(self, X: ArrayLike) -> NDArray:
        raise NotImplementedError("Must be implemented")

    def _validate_y(self, y: ArrayLike) -> NDArray:
        y_arr: NDArray = np.asarray(y)
        unseen: NDArray = np.setdiff1d(np.unique(y_arr), self._classes)
        if len(unseen) > 0:
            raise ValueError(
                f"y contains classes not declared at init: {unseen.tolist()}. "
                f"Declared classes: {self._classes.tolist()}"
            )
        return y_arr

    @property
    def classes(self) -> NDArray:
        return self._classes

    def count_of_classes(self) -> int:
        return len(self._classes)
