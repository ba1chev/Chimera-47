from abc import ABC, abstractmethod
from numpy.typing import ArrayLike, NDArray


class Normalizer(ABC):
    """Abstract base for feature normalizers. Stateless or stateful"""

    def __init__(self) -> None:
        self._is_fitted: bool = False

    @abstractmethod
    def fit(self, X: ArrayLike) -> "Normalizer":
        raise NotImplementedError("Must be implemented")

    @abstractmethod
    def transform(self, X: ArrayLike) -> NDArray:
        raise NotImplementedError("Must be implemented")

    def fit_transform(self, X: ArrayLike) -> NDArray:
        return self.fit(X).transform(X)

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted
