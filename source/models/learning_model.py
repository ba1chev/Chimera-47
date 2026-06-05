from typing import Optional
from abc import ABC, abstractmethod
from numpy.typing import ArrayLike, NDArray


class LearningModel(ABC):
    """Root abstract base class for every learning model in Chimera-47"""

    def __init__(self) -> None:
        self._is_fitted: bool = False

    @abstractmethod
    def fit(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> "LearningModel":
        raise NotImplementedError("Must be implemented")

    @abstractmethod
    def predict(self, X: ArrayLike) -> NDArray:
        raise NotImplementedError("Must be implemented")

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted
