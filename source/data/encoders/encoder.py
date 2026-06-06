from abc import ABC, abstractmethod
from numpy.typing import ArrayLike, NDArray


class Encoder(ABC):
    """Abstract base for stateless encoders that map labels to a numeric representation and back."""

    @abstractmethod
    def encode(self, values: ArrayLike) -> NDArray:
        raise NotImplementedError("Must be implemented")

    @abstractmethod
    def decode(self, encoded: ArrayLike) -> NDArray:
        raise NotImplementedError("Must be implemented")
