from numpy.typing import ArrayLike
from abc import ABC, abstractmethod


class Metric(ABC):
    """Abstract base for scalar classification metrics computed from true and predicted labels."""

    @abstractmethod
    def evaluate(self, y_true: ArrayLike, y_predicted: ArrayLike) -> float:
        raise NotImplementedError("Must be implemented")
