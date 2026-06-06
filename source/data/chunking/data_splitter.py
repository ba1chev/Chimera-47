from numpy.typing import ArrayLike
from abc import ABC, abstractmethod

from source.data.chunking.data_split import DataSplit


class DataSplitter(ABC):
    """Abstract base for one-shot train/test splitters."""

    @abstractmethod
    def split(self, X: ArrayLike, y: ArrayLike) -> DataSplit:
        raise NotImplementedError("Must be implemented")
